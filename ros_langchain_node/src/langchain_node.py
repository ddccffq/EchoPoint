#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ros_langchain_node.srv import DescribeImage, DescribeImageResponse

try:
    import rospkg
except ImportError:
    rospkg = None


def _ensure_local_agents_on_path() -> None:
    """兼容 `rosrun` / `catkin_install_python`，确保可导入本包下的 `agents`。"""

    candidate_paths = []

    if rospkg is not None:
        try:
            package_root = Path(rospkg.RosPack().get_path("ros_langchain_node"))
            candidate_paths.extend([package_root / "src", package_root / "src" / "agents"])
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


_ensure_local_agents_on_path()

try:
    from ros_AIUI_node.srv import textToSpeak, textToSpeakRequest

    _TTS_AVAILABLE = True
except ImportError:
    textToSpeak = None
    textToSpeakRequest = None
    _TTS_AVAILABLE = False

try:
    from agents.agent import Agent
    from agents.prompts import IMAGE_TO_TEXT_PROMPT
except ImportError as exc:  # pragma: no cover - depends on ROS / workspace path
    Agent = None
    IMAGE_TO_TEXT_PROMPT = None
    _AGENT_IMPORT_ERROR = exc
else:
    _AGENT_IMPORT_ERROR = None


def _build_chat_model(provider: str, model_name: str, temperature: float):
    """按 ROS 参数/环境变量构建一个 LangChain chat model。"""

    provider = (provider or "").strip().lower()

    if provider in {"", "openai"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "未安装 langchain_openai，无法使用 openai 后端。"
            ) from exc

        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "未安装 langchain_community，无法使用 ollama 后端。"
            ) from exc

        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }

        base_url = os.environ.get("OLLAMA_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOllama(**kwargs)

    raise RuntimeError(f"不支持的模型后端：{provider}")

class LangchainNode:
    def __init__(self):
        rospy.init_node("langchain_node")
        self.bridge = CvBridge()

        self.service_name = rospy.get_param("~service_name", "/langchain/describe_image")
        self.image_encoding = rospy.get_param("~image_encoding", "bgr8")
        self.enable_tts = bool(rospy.get_param("~enable_tts", True))
        self.tts_service_name = rospy.get_param("~tts_service_name", "/aiui/text_to_speak")
        self.tts_wait_timeout = float(rospy.get_param("~tts_wait_timeout", 3.0))

        self.model_provider = rospy.get_param(
            "~model_provider", os.environ.get("LANGCHAIN_NODE_MODEL_PROVIDER", "openai")
        )
        self.model_name = rospy.get_param(
            "~model_name",
            os.environ.get("LANGCHAIN_NODE_MODEL_NAME", "gpt-image-2-codex"),
        )
        self.temperature = float(rospy.get_param("~temperature", 0.0))
        self.verbose_agent = bool(rospy.get_param("~verbose_agent", False))

        self.agent = self._build_agent()

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

        self.server = rospy.Service(self.service_name, DescribeImage, self.handle_describe)
        rospy.loginfo("Service ready: %s", self.service_name)

        if self.agent is None:
            rospy.logwarn("LangChain agent 未启用：%s", self._agent_status())

    def _agent_status(self) -> str:
        if _AGENT_IMPORT_ERROR is not None:
            return f"无法导入 agents.Agent / prompt：{_AGENT_IMPORT_ERROR}"
        return "未配置可用的模型后端"

    def _build_agent(self) -> Optional[Agent]:
        if Agent is None or IMAGE_TO_TEXT_PROMPT is None:
            return None

        try:
            model = _build_chat_model(self.model_provider, self.model_name, self.temperature)
        except Exception as exc:
            rospy.logwarn("模型后端初始化失败：%s", exc)
            return None

        try:
            return Agent(
                model=model,
                prompt=IMAGE_TO_TEXT_PROMPT,
                verbose=self.verbose_agent,
            )
        except Exception as exc:
            rospy.logwarn("Agent 初始化失败：%s", exc)
            return None

    def call_agent_with_image(self, cv_img):
        if self.agent is None:
            raise RuntimeError(self._agent_status())

        if cv_img is None:
            raise ValueError("输入图像为空")

        ok, encoded = cv2.imencode(".png", cv_img)
        if not ok:
            raise RuntimeError("图像编码失败，无法发送给 LangChain 模型")

        image_bytes = encoded.tobytes()
        text = self.agent.invoke(
            image_bytes=image_bytes,
            mime_type="image/png",
        )

        if not isinstance(text, str):
            text = str(text)

        return text.strip()

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
            cv_img = self.bridge.imgmsg_to_cv2(req.image, desired_encoding=self.image_encoding)
            text = self.call_agent_with_image(cv_img)

            if req.speak and text.strip():
                ok = self.speak(text)
                if not ok:
                    rospy.logwarn("TTS 调用未成功，但图像描述已生成")

            return DescribeImageResponse(True, text, "")
        except Exception as e:
            rospy.logerr("/langchain/describe_image 处理失败：%s", e)
            return DescribeImageResponse(False, "", str(e))

if __name__ == '__main__':
    LangchainNode()
    rospy.spin()
