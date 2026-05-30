#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path
from typing import Optional
import base64
import importlib
import mimetypes
import threading

import cv2
try:
    import rospy
except ImportError:
    rospy = None

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    from ros_langchain_node.srv import DescribeImage, DescribeImageResponse
except Exception:
    DescribeImage = None
    DescribeImageResponse = None

try:
    import rospkg
except ImportError:
    rospkg = None


def _ensure_package_src_on_path() -> None:
    candidate_paths = []

    if rospkg is not None:
        try:
            package_root = Path(rospkg.RosPack().get_path("ros_langchain_node"))
            candidate_paths.append(package_root / "src")
        except Exception:
            pass

    current_file = Path(__file__).resolve()
    candidate_paths.extend(
        [
            current_file.parent,
            current_file.parent / "src",
            current_file.parent.parent / "src",
            current_file.parent.parent.parent / "src",
        ]
    )

    for path in candidate_paths:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_ensure_package_src_on_path()

try:
    from ros_AIUI_node.srv import textToSpeak, textToSpeakRequest

    _TTS_AVAILABLE = True
except ImportError:
    textToSpeak = None
    textToSpeakRequest = None
    _TTS_AVAILABLE = False

try:
    from model_manager import ModelManagerLM, image_to_data_url
    from config import (
        BASE_LOCAL_SERVER,
        BASE_VISION_MODEL_IDENTIFIER,
        LOAD_ENDPOINT,
        UNLOAD_ENDPOINT,
        CHAT_ENDPOINT,
        CONTEXT_LENGTH,
        EVAL_BATCH_SIZE,
        FLASH_ATTENTION,
        OFFLOAD_KV_CACHE_TO_GPU,
        SYSTEM_PROMPT,
        TEMPERATURE,
        MAX_OUTPUT_TOKENS,
        STORE,
    )
except ImportError as exc:  # pragma: no cover - depends on ROS / workspace path
    ModelManagerLM = None
    image_to_data_url = None
    BASE_LOCAL_SERVER = None
    BASE_VISION_MODEL_IDENTIFIER = None
    LOAD_ENDPOINT = UNLOAD_ENDPOINT = CHAT_ENDPOINT = None
    CONTEXT_LENGTH = EVAL_BATCH_SIZE = None
    FLASH_ATTENTION = OFFLOAD_KV_CACHE_TO_GPU = None
    SYSTEM_PROMPT = None
    TEMPERATURE = None
    MAX_OUTPUT_TOKENS = None
    STORE = None
    _AGENT_IMPORT_ERROR = exc
else:
    _AGENT_IMPORT_ERROR = None


class LangchainNode:
    def __init__(self):
        rospy.init_node("langchain_node")
        self.bridge = CvBridge()
        self._manager_lock = threading.Lock()

        self.service_name = rospy.get_param("~service_name", "/langchain/describe_image")
        self.image_encoding = rospy.get_param("~image_encoding", "bgr8")
        self.enable_tts = bool(rospy.get_param("~enable_tts", True))
        self.tts_service_name = rospy.get_param("~tts_service_name", "/aiui/text_to_speak")
        self.tts_wait_timeout = float(rospy.get_param("~tts_wait_timeout", 3.0))

        self.base_local_server = rospy.get_param("~base_local_server", BASE_LOCAL_SERVER)
        self.base_vision_model_identifier = rospy.get_param(
            "~model_identifier", BASE_VISION_MODEL_IDENTIFIER
        )
        self.temperature = float(rospy.get_param("~temperature", TEMPERATURE))
        self.max_output_tokens = int(rospy.get_param("~max_output_tokens", MAX_OUTPUT_TOKENS))

        self.manager = None

        self.tts_cli = None
        if self.enable_tts and _TTS_AVAILABLE:
            try:
                rospy.wait_for_service(self.tts_service_name, timeout=self.tts_wait_timeout)
                self.tts_cli = rospy.ServiceProxy(self.tts_service_name, textToSpeak)
                rospy.loginfo("TTS service ready: %s", self.tts_service_name)
            except rospy.ROSException:
                rospy.logwarn(
                    "TTS service %s unavailable, speak() will be skipped",
                    self.tts_service_name,
                )
        elif self.enable_tts and not _TTS_AVAILABLE:
            rospy.logwarn("ros_AIUI_node 不可导入，TTS 功能已禁用")

        svc_cls = DescribeImage
        resp_cls = DescribeImageResponse
        if svc_cls is None or resp_cls is None:
            try:
                mod = importlib.import_module("ros_langchain_node.srv")
                svc_cls = getattr(mod, "DescribeImage", None)
                resp_cls = getattr(mod, "DescribeImageResponse", None)
            except Exception as e:
                svc_cls = None
                resp_cls = None
                if rospy is not None:
                    rospy.logwarn("DescribeImage srv not importable: %s", e)
                else:
                    print("DescribeImage srv not importable:", e)

        if svc_cls is not None and resp_cls is not None and rospy is not None:
            self.response_cls = resp_cls
            self.server = rospy.Service(self.service_name, svc_cls, self.handle_describe)
            rospy.loginfo("Service ready: %s", self.service_name)
        else:
            if rospy is not None:
                rospy.logwarn("DescribeImage service type not found; service not created")
            else:
                print("DescribeImage service type not found; service not created")

        rospy.on_shutdown(self._shutdown)

    def _agent_status(self) -> str:
        if _AGENT_IMPORT_ERROR is not None:
            return f"无法导入 model_manager/config：{_AGENT_IMPORT_ERROR}"
        return "未配置可用的模型后端或 model_manager 未就绪"

    def _get_manager(self) -> ModelManagerLM:
        if self.manager is not None:
            return self.manager

        with self._manager_lock:
            if self.manager is None:
                self.manager = self._build_and_load_manager()
        return self.manager

    def _build_and_load_manager(self) -> Optional[ModelManagerLM]:
        if ModelManagerLM is None:
            raise RuntimeError(self._agent_status())

        try:
            manager = ModelManagerLM(
                base_local_server=self.base_local_server,
                load_endpoint=LOAD_ENDPOINT,
                unload_endpoint=UNLOAD_ENDPOINT,
                chat_endpoint=CHAT_ENDPOINT,
            )
        except Exception as exc:
            raise RuntimeError(f"ModelManager 初始化失败：{exc}") from exc

        try:
            manager.load(
                model_identifier=self.base_vision_model_identifier,
                context_length=CONTEXT_LENGTH,
                eval_batch_size=EVAL_BATCH_SIZE,
                flash_attention=FLASH_ATTENTION,
                offload_kv_cache_to_gpu=OFFLOAD_KV_CACHE_TO_GPU,
            )
            rospy.loginfo("模型已加载：%s", self.base_vision_model_identifier)
            return manager
        except Exception as exc:
            raise RuntimeError(f"模型加载失败：{exc}") from exc

    def _data_url_from_cv_image(self, cv_img) -> str:
        ok, encoded = cv2.imencode(".png", cv_img)
        if not ok:
            raise RuntimeError("图像编码失败")
        return self._data_url_from_bytes(encoded.tobytes(), "image/png")

    def _data_url_from_bytes(self, image_bytes: bytes, mime: str) -> str:
        if not image_bytes:
            raise ValueError("输入图像字节为空")
        data_b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime or 'image/png'};base64,{data_b64}"

    def _data_url_from_path(self, path: Path) -> str:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        return self._data_url_from_bytes(path.read_bytes(), mime)

    def _data_url_from_request(self, req) -> str:
        image_path = getattr(req, "image_path", "")
        if image_path:
            path = Path(image_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"图片路径不存在：{path}")
            return self._data_url_from_path(path)

        image_data = bytes(getattr(req, "image_data", []) or [])
        if image_data:
            return self._data_url_from_bytes(image_data, getattr(req, "image_mime", "image/png"))

        image_msg = getattr(req, "image", None)
        if image_msg is not None and getattr(image_msg, "data", None):
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding=self.image_encoding)
            if cv_img is None:
                raise ValueError("输入 ROS Image 为空")
            return self._data_url_from_cv_image(cv_img)

        raise ValueError("请求中没有可用图像：请传入 image、image_path 或 image_data")

    def call_agent_with_image(self, data_url: str) -> str:
        manager = self._get_manager()

        input_prompt = [
            {"type": "text", "content": "描述这张图片"},
            {"type": "image", "data_url": data_url},
        ]

        try:
            response = manager.chat(
                prompt=input_prompt,
                temperature=self.temperature,
                stream=False,
                max_output_tokens=self.max_output_tokens,
                system_prompt=SYSTEM_PROMPT,
                store=STORE,
            )
        except Exception as exc:
            raise RuntimeError(f"模型调用失败：{exc}") from exc

        try:
            text = response.get("output", [])[0].get("content", "")
        except Exception:
            text = str(response)

        if not isinstance(text, str):
            text = str(text)

        return text.strip()

    def _shutdown(self):
        try:
            if self.manager is not None:
                self.manager.unload()
                rospy.loginfo("已卸载模型实例")
        except Exception as exc:
            rospy.logwarn("卸载模型时出错：%s", exc)

    def speak(self, text):
        if not self.enable_tts or self.tts_cli is None or not _TTS_AVAILABLE:
            rospy.loginfo("TTS 跳过：%s", text)
            return False

        try:
            req = textToSpeakRequest(text=text)
            resp = self.tts_cli(req)
            return bool(getattr(resp, "is_success", False))
        except rospy.ServiceException as exc:
            rospy.logwarn("TTS 调用失败：%s", exc)
            return False

    def handle_describe(self, req):
        try:
            data_url = self._data_url_from_request(req)
            text = self.call_agent_with_image(data_url)

            if req.speak and text.strip():
                ok = self.speak(text)
                if not ok:
                    rospy.logwarn("TTS 调用未成功，但图像描述已生成")

            return self.response_cls(True, text, "")
        except Exception as e:
            rospy.logerr("/langchain/describe_image 处理失败：%s", e)
            return self.response_cls(False, "", str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ROS/local LM image description node")
    parser.add_argument(
        "--standalone-test",
        action="store_true",
        help="load the model once and describe --image without starting the ROS service",
    )
    parser.add_argument(
        "--image",
        default="1.png",
        help="image path used with --standalone-test",
    )
    args = parser.parse_args()

    if not args.standalone_test:
        if rospy is None or CvBridge is None:
            print("rospy/cv_bridge is not available, cannot run as a ROS node.")
            raise SystemExit(1)
        LangchainNode()
        rospy.spin()
    else:
        print("Running standalone test mode (no ROS service).")
        if ModelManagerLM is None:
            print("ModelManagerLM not available (could not import). Exiting.")
            raise SystemExit(1)

        test_image_path = Path(args.image)
        if not test_image_path.is_absolute():
            test_image_path = Path(__file__).parent / test_image_path

        if not test_image_path.exists():
            print(f"Test image not found: {test_image_path}")
            raise SystemExit(1)

        mgr = ModelManagerLM(
            base_local_server=BASE_LOCAL_SERVER,
            load_endpoint=LOAD_ENDPOINT,
            unload_endpoint=UNLOAD_ENDPOINT,
            chat_endpoint=CHAT_ENDPOINT,
        )

        try:
            print(f"Loading model: {BASE_VISION_MODEL_IDENTIFIER}")
            mgr.load(
                model_identifier=BASE_VISION_MODEL_IDENTIFIER,
                context_length=CONTEXT_LENGTH,
                eval_batch_size=EVAL_BATCH_SIZE,
                flash_attention=FLASH_ATTENTION,
                offload_kv_cache_to_gpu=OFFLOAD_KV_CACHE_TO_GPU,
            )

            data_url = image_to_data_url(str(test_image_path))
            prompt = [
                {"type": "text", "content": "描述这张图片"},
                {"type": "image", "data_url": data_url},
            ]

            print(f"Calling chat with image: {test_image_path}")
            resp = mgr.chat(
                prompt=prompt,
                temperature=TEMPERATURE,
                stream=False,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                system_prompt=SYSTEM_PROMPT,
                store=STORE,
            )

            try:
                out = resp.get("output", [])[0].get("content", "")
            except Exception:
                out = str(resp)

            print("Model response:\n", out)
        finally:
            try:
                mgr.unload()
            except Exception:
                pass
