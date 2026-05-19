# agent.py
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


ImageBlockFormat = Literal["openai", "langchain"]
ImagePath = Union[str, Path]


class InvokeableModel(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> Any:
        ...


class Agent:
    """
    图像输入 + 云端多模态模型 + 纯文本输出的轻量 Agent。

    支持以下图片输入方式，调用时四选一即可：
    1. image_path: 本地图片路径
    2. image_bytes: 图片二进制内容
    3. image_base64: 图片 base64 字符串
    4. image_url: 远程图片 URL 或 data URL
    """

    def __init__(
        self,
        model: InvokeableModel,
        prompt: ChatPromptTemplate,
        verbose: bool = False,
        image_block_format: ImageBlockFormat = "openai",
        max_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.verbose = verbose
        self.image_block_format = image_block_format
        self.max_image_bytes = max_image_bytes

    def invoke(
        self,
        *,
        image_path: Optional[ImagePath] = None,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        mime_type: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        detail: Optional[Literal["auto", "low", "high"]] = None,
        **model_kwargs: Any,
    ) -> str:
        """
        调用多模态模型识别图片内容。

        参数说明：
        - image_path: 本地图片路径
        - image_bytes: 图片二进制内容
        - image_base64: 图片 base64 字符串，不包含 data:image/... 前缀
        - image_url: 图片 URL，支持 https://... 或 data:image/...;base64,...
        - mime_type: 图片 MIME 类型，例如 image/jpeg、image/png
        - prompt_variables: prompt 模板变量；当前 prompt 无变量时可不传
        - detail: 图片细节等级，部分模型支持 auto / low / high
        - model_kwargs: 透传给底层模型的参数，例如 temperature、max_tokens 等

        返回：
        - 模型输出的纯文本字符串
        """

        image_url_value = self._build_image_url_value(
            image_path=image_path,
            image_bytes=image_bytes,
            image_base64=image_base64,
            image_url=image_url,
            mime_type=mime_type,
        )

        messages = self._build_multimodal_messages(
            image_url_value=image_url_value,
            prompt_variables=prompt_variables or {},
            detail=detail,
        )

        if self.verbose:
            self._print_safe_messages(messages)

        response = self.model.invoke(messages, **model_kwargs)
        return self._extract_text(response)

    def _build_image_url_value(
        self,
        *,
        image_path: Optional[ImagePath],
        image_bytes: Optional[bytes],
        image_base64: Optional[str],
        image_url: Optional[str],
        mime_type: Optional[str],
    ) -> str:
        provided = [
            image_path is not None,
            image_bytes is not None,
            image_base64 is not None,
            image_url is not None,
        ]

        if sum(provided) != 1:
            raise ValueError(
                "必须且只能提供一种图片输入方式：image_path、image_bytes、image_base64 或 image_url。"
            )

        if image_url is not None:
            return image_url

        if image_path is not None:
            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(f"图片文件不存在：{path}")

            if not path.is_file():
                raise ValueError(f"不是有效的图片文件：{path}")

            data = path.read_bytes()
            self._check_image_size(data)

            guessed_mime_type = mime_type or self._guess_mime_type(path)
            encoded = base64.b64encode(data).decode("utf-8")
            return self._to_data_url(encoded, guessed_mime_type)

        if image_bytes is not None:
            self._check_image_size(image_bytes)

            if not mime_type:
                raise ValueError("使用 image_bytes 时必须提供 mime_type，例如 image/jpeg 或 image/png。")

            encoded = base64.b64encode(image_bytes).decode("utf-8")
            return self._to_data_url(encoded, mime_type)

        if image_base64 is not None:
            if not mime_type:
                raise ValueError("使用 image_base64 时必须提供 mime_type，例如 image/jpeg 或 image/png。")

            clean_base64 = self._clean_base64(image_base64)
            return self._to_data_url(clean_base64, mime_type)

        raise ValueError("未提供图片输入。")

    def _build_multimodal_messages(
        self,
        *,
        image_url_value: str,
        prompt_variables: Dict[str, Any],
        detail: Optional[Literal["auto", "low", "high"]],
    ) -> List[BaseMessage]:
        prompt_value = self.prompt.invoke(prompt_variables)
        messages = list(prompt_value.to_messages())

        image_block = self._build_image_block(image_url_value=image_url_value, detail=detail)

        last_human_index = self._find_last_human_message_index(messages)

        if last_human_index is None:
            messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "请根据随附图像完成任务，并只输出纯文本结果。",
                        },
                        image_block,
                    ]
                )
            )
            return messages

        old_human_message = messages[last_human_index]
        old_content = old_human_message.content

        new_content = self._content_to_blocks(old_content)
        new_content.append(image_block)

        messages[last_human_index] = HumanMessage(content=new_content)

        return messages

    def _build_image_block(
        self,
        *,
        image_url_value: str,
        detail: Optional[Literal["auto", "low", "high"]],
    ) -> Dict[str, Any]:
        if self.image_block_format == "openai":
            image_url_payload: Dict[str, Any] = {"url": image_url_value}

            if detail is not None:
                image_url_payload["detail"] = detail

            return {
                "type": "image_url",
                "image_url": image_url_payload,
            }

        if self.image_block_format == "langchain":
            block: Dict[str, Any] = {
                "type": "image",
                "url": image_url_value,
            }

            if detail is not None:
                block["extras"] = {"detail": detail}

            return block

        raise ValueError(f"不支持的 image_block_format：{self.image_block_format}")

    @staticmethod
    def _find_last_human_message_index(messages: List[BaseMessage]) -> Optional[int]:
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return index
        return None

    @staticmethod
    def _content_to_blocks(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []

            for item in content:
                if isinstance(item, str):
                    blocks.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    blocks.append(item)
                else:
                    blocks.append({"type": "text", "text": str(item)})

            return blocks

        return [{"type": "text", "text": str(content)}]

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        兼容常见 LangChain 返回：
        - AIMessage(content="...")
        - AIMessage(content=[{"type": "text", "text": "..."}])
        - 普通字符串
        - 其他带 content 字段的对象
        """

        if isinstance(response, str):
            return response.strip()

        if isinstance(response, AIMessage):
            return Agent._extract_content_text(response.content)

        if hasattr(response, "content"):
            return Agent._extract_content_text(response.content)

        return str(response).strip()

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            texts: List[str] = []

            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") in {"text", "output_text"} and "text" in item:
                        texts.append(str(item["text"]))
                    elif "content" in item:
                        texts.append(str(item["content"]))

            return "\n".join(texts).strip()

        return str(content).strip()

    def _check_image_size(self, data: bytes) -> None:
        if len(data) > self.max_image_bytes:
            raise ValueError(
                f"图片大小超过限制：{len(data)} bytes，最大允许 {self.max_image_bytes} bytes。"
            )

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type is None:
            raise ValueError(
                f"无法根据文件后缀判断图片 MIME 类型，请手动传入 mime_type。文件：{path}"
            )

        if not mime_type.startswith("image/"):
            raise ValueError(f"文件 MIME 类型不是图片：{mime_type}")

        return mime_type

    @staticmethod
    def _to_data_url(image_base64: str, mime_type: str) -> str:
        if not mime_type.startswith("image/"):
            raise ValueError(f"mime_type 必须是 image/* 类型，当前为：{mime_type}")

        return f"data:{mime_type};base64,{image_base64}"

    @staticmethod
    def _clean_base64(image_base64: str) -> str:
        text = image_base64.strip()

        if text.startswith("data:"):
            _, _, text = text.partition(",")

        return text.strip()

    @staticmethod
    def _print_safe_messages(messages: List[BaseMessage]) -> None:
        """
        verbose 模式下打印消息结构，但不完整打印 base64，避免终端刷屏。
        """

        safe_messages: List[Dict[str, Any]] = []

        for message in messages:
            content = message.content

            if isinstance(content, list):
                safe_content = []

                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"image_url", "image"}:
                        safe_content.append(
                            {
                                "type": block.get("type"),
                                "image": "<image data omitted>",
                            }
                        )
                    else:
                        safe_content.append(block)

                content = safe_content

            safe_messages.append(
                {
                    "type": message.__class__.__name__,
                    "content": content,
                }
            )

        print(safe_messages)


