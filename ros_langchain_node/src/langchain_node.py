#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import base64
import importlib
import mimetypes
import os
import threading

import cv2

try:
    basestring
except NameError:
    basestring = str

try:
    unicode
except NameError:
    unicode = str

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


def _ensure_package_src_on_path():
    candidate_paths = []

    if rospkg is not None:
        try:
            package_root = rospkg.RosPack().get_path("ros_langchain_node")
            candidate_paths.append(os.path.join(package_root, "src"))
        except Exception:
            pass

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    candidate_paths.extend(
        [
            current_dir,
            os.path.join(current_dir, "src"),
            os.path.join(parent_dir, "src"),
            os.path.join(grandparent_dir, "src"),
        ]
    )

    for path in candidate_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


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
except ImportError as exc:
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
    @staticmethod
    def _to_ros_string(text):
        if isinstance(text, unicode):
            return text.encode("utf-8")
        return text

    def __init__(self):
        rospy.init_node("langchain_node")

        if CvBridge is None:
            raise RuntimeError("cv_bridge 不可用，无法启动 ROS 图像服务")

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

        # 新增参数：
        # True 表示每次 service 请求结束后自动卸载模型。
        # 这样 LM Studio 端模型会在调用完成后 eject/unload。
        self.auto_unload_after_request = bool(
            rospy.get_param("~auto_unload_after_request", True)
        )

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
                rospy.logwarn("DescribeImage srv not importable: %s", e)

        if svc_cls is not None and resp_cls is not None:
            self.response_cls = resp_cls
            self.server = rospy.Service(self.service_name, svc_cls, self.handle_describe)
            rospy.loginfo("Service ready: %s", self.service_name)
            rospy.loginfo("auto_unload_after_request: %s", self.auto_unload_after_request)
        else:
            rospy.logwarn("DescribeImage service type not found; service not created")

        rospy.on_shutdown(self._shutdown)

    def _agent_status(self):
        if _AGENT_IMPORT_ERROR is not None:
            return "无法导入 model_manager/config：%s" % _AGENT_IMPORT_ERROR
        return "未配置可用的模型后端或 model_manager 未就绪"

    def _get_manager(self):
        """
        获取模型管理器。
        如果当前没有 manager，则创建并加载模型。
        """
        if self.manager is not None:
            return self.manager

        with self._manager_lock:
            if self.manager is None:
                self.manager = self._build_and_load_manager()

        return self.manager

    def _build_and_load_manager(self):
        """
        创建 ModelManagerLM，并加载视觉模型。
        """
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
            raise RuntimeError("ModelManager 初始化失败：%s" % exc)

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
            raise RuntimeError("模型加载失败：%s" % exc)

    def _unload_manager(self):
        """
        卸载当前已加载的模型。

        这个函数会调用 self.manager.unload()。
        如果 model_manager.py 里的 unload() 对应 LM Studio 的 unload/eject 接口，
        那么这里执行后，LM Studio 侧模型就会被卸载。
        """
        with self._manager_lock:
            if self.manager is None:
                return

            try:
                rospy.loginfo("正在卸载模型：%s", self.base_vision_model_identifier)
                self.manager.unload()
                rospy.loginfo("模型已卸载：%s", self.base_vision_model_identifier)
            except Exception as exc:
                rospy.logwarn("卸载模型时出错：%s", exc)
            finally:
                self.manager = None

    def _data_url_from_cv_image(self, cv_img):
        ok, encoded = cv2.imencode(".png", cv_img)
        if not ok:
            raise RuntimeError("图像编码失败")

        return self._data_url_from_bytes(encoded.tobytes(), "image/png")

    def _data_url_from_bytes(self, image_bytes, mime):
        if not image_bytes:
            raise ValueError("输入图像字节为空")

        data_b64 = base64.b64encode(image_bytes).decode("ascii")
        return "data:%s;base64,%s" % (mime or "image/png", data_b64)

    def _data_url_from_path(self, path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as image_file:
            return self._data_url_from_bytes(image_file.read(), mime)

    def _data_url_from_request(self, req):
        """
        从 service 请求中提取图像。

        优先级：
        1. image_path
        2. image_data
        3. sensor_msgs/Image image
        """
        image_path = getattr(req, "image_path", "")

        if image_path:
            path = os.path.expanduser(image_path)
            if not os.path.exists(path):
                raise IOError("图片路径不存在：%s" % path)
            return self._data_url_from_path(path)

        image_data = bytearray(getattr(req, "image_data", []) or [])
        if image_data:
            return self._data_url_from_bytes(
                image_data,
                getattr(req, "image_mime", "image/png"),
            )

        image_msg = getattr(req, "image", None)
        if image_msg is not None and getattr(image_msg, "data", None):
            cv_img = self.bridge.imgmsg_to_cv2(
                image_msg,
                desired_encoding=self.image_encoding,
            )
            if cv_img is None:
                raise ValueError("输入 ROS Image 为空")

            return self._data_url_from_cv_image(cv_img)

        raise ValueError("请求中没有可用图像：请传入 image、image_path 或 image_data")

    def extract_message_text(self, response):
        """
        只提取模型返回中的 message 内容，过滤 reasoning。
        """
        messages = []

        for item in response.get("output", []):
            if item.get("type") == "message":
                content = item.get("content", "")
                if isinstance(content, basestring):
                    messages.append(content.strip())

        return "\n".join(messages).strip()

    def call_agent_with_image(self, data_url):
        """
        调用本地视觉语言模型，生成图像描述。
        只返回 type == message 的内容，不返回 reasoning。
        """
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
            raise RuntimeError("模型调用失败：%s" % exc)

        text = self.extract_message_text(response)

        if not text:
            rospy.logwarn("未提取到 message 内容，原始 response=%s", response)
            text = ""

        return text

    def speak(self, text):
        """
        调用外部 TTS service 朗读文本。

        注意：
        如果 /aiui/text_to_speak 是同步 service，
        那么这个函数返回时可以认为朗读已经完成。

        如果 /aiui/text_to_speak 内部只是把文本加入播放队列然后立即返回，
        那么这里无法保证真实语音播放已经结束。
        """
        if not self.enable_tts or self.tts_cli is None or not _TTS_AVAILABLE:
            rospy.loginfo("TTS 跳过：%s", text)
            return False

        try:
            req = textToSpeakRequest(text=self._to_ros_string(text))
            resp = self.tts_cli(req)
            return bool(getattr(resp, "is_success", False))
        except rospy.ServiceException as exc:
            rospy.logwarn("TTS 调用失败：%s", exc)
            return False
        except Exception as exc:
            rospy.logwarn("TTS 调用异常：%s", exc)
            return False

    def handle_describe(self, req):
        """
        /langchain/describe_image 的 service 回调。

        修改点：
        - 原来模型只在节点关闭时卸载。
        - 现在如果 auto_unload_after_request=True，
          每次 service 请求结束后都会自动调用 self._unload_manager()。
        """
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

        finally:
            if self.auto_unload_after_request:
                self._unload_manager()

    def _shutdown(self):
        """
        节点关闭时兜底卸载模型。
        """
        self._unload_manager()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROS/local LM image description node"
    )

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

    args, unknown = parser.parse_known_args()

    if unknown:
        print("Ignored ROS args:", unknown)

    if not args.standalone_test:
        if rospy is None or CvBridge is None:
            print("rospy/cv_bridge is not available, cannot run as a ROS node.")
            raise SystemExit(1)

        LangchainNode()
        rospy.spin()

    else:
        print("Running standalone test mode no ROS service.")

        if ModelManagerLM is None:
            print("ModelManagerLM not available could not import. Exiting.")
            raise SystemExit(1)

        test_image_path = os.path.expanduser(args.image)

        if not os.path.isabs(test_image_path):
            test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_image_path)

        if not os.path.exists(test_image_path):
            print("Test image not found: %s" % test_image_path)
            raise SystemExit(1)

        mgr = ModelManagerLM(
            base_local_server=BASE_LOCAL_SERVER,
            load_endpoint=LOAD_ENDPOINT,
            unload_endpoint=UNLOAD_ENDPOINT,
            chat_endpoint=CHAT_ENDPOINT,
        )

        try:
            print("Loading model: %s" % BASE_VISION_MODEL_IDENTIFIER)

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

            print("Calling chat with image: %s" % test_image_path)

            resp = mgr.chat(
                prompt=prompt,
                temperature=TEMPERATURE,
                stream=False,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                system_prompt=SYSTEM_PROMPT,
                store=STORE,
            )

            messages = []

            try:
                for item in resp.get("output", []):
                    if item.get("type") == "message":
                        content = item.get("content", "")

                        if isinstance(content, basestring):
                            messages.append(content.strip())

                out = "\n".join(messages).strip()

            except Exception:
                out = str(resp)

            print("Model response:\n", out)

        finally:
            try:
                print("Unloading model...")
                mgr.unload()
                print("Model unloaded.")

            except Exception as exc:
                print("Unload failed: %s" % exc)
