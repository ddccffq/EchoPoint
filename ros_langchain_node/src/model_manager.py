# -*- coding: utf-8 -*-

import requests
import base64

def image_to_data_url(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode()

    return (
        "data:image/png;base64,"
        + encoded
    )

class ModelManagerLM:

    def __init__(
        self,
        base_local_server,
        load_endpoint,
        unload_endpoint,
        chat_endpoint
    ):
        self.load_url = "%s%s" % (base_local_server.strip("/"), load_endpoint)
        self.unload_url = "%s%s" % (base_local_server.strip("/"), unload_endpoint)
        self.chat_url = "%s%s" % (base_local_server.strip("/"), chat_endpoint)
        self.instance_id = None

    def load(
        self,
        model_identifier,
        context_length,
        eval_batch_size,
        flash_attention,
        offload_kv_cache_to_gpu
    ):
        payload = {
                "model": model_identifier,
                "context_length": context_length,
                "eval_batch_size": eval_batch_size,
                "flash_attention": flash_attention,
                "offload_kv_cache_to_gpu": offload_kv_cache_to_gpu,
        }
        r = requests.post(
            url=self.load_url,
            json=payload
        )

        r.raise_for_status()

        data = r.json()
        self.instance_id = data["instance_id"]

        return data

    def chat(
        self,
        prompt,
        temperature,
        stream,
        max_output_tokens,
        system_prompt=None,
        store=False,
    ):
        if not self.instance_id:
            raise RuntimeError(
                "model not loaded"
            )

        payload = {
            "model": self.instance_id,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "stream": stream,
            "store": store

        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        r = requests.post(
            url=self.chat_url,
            json=payload,
            timeout=300

        )
        r.raise_for_status()

        return r.json()

    def unload(self):
        if not self.instance_id:
            return

        requests.post(
            self.unload_url,
            json={
                "instance_id": self.instance_id
            }
        )
        self.instance_id = None

def main():
    from config import (
        BASE_LOCAL_SERVER,
        BASE_VISION_MODEL_IDENTIFIER,
        LOAD_ENDPOINT,
        CONTEXT_LENGTH,
        EVAL_BATCH_SIZE,
        FLASH_ATTENTION,
        OFFLOAD_KV_CACHE_TO_GPU,

        UNLOAD_ENDPOINT,

        CHAT_ENDPOINT,
        STREAM,
        SYSTEM_PROMPT,
        TEMPERATURE,
        MAX_OUTPUT_TOKENS,
        STORE,
    )

    manager = ModelManagerLM(
        BASE_LOCAL_SERVER,
        LOAD_ENDPOINT,
        UNLOAD_ENDPOINT,
        CHAT_ENDPOINT
    )
    manager.load(
        BASE_VISION_MODEL_IDENTIFIER,
        CONTEXT_LENGTH,
        EVAL_BATCH_SIZE,
        FLASH_ATTENTION,
        OFFLOAD_KV_CACHE_TO_GPU
    )

    input_prompt = [
        {
            "type": "text",
            "content": "描述这张图片"
        },
        {
            "type": "image",
            "data_url": image_to_data_url("1.png")
        }
    ]

    response = manager.chat(
        prompt=input_prompt,
        temperature=TEMPERATURE,
        stream=STREAM,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=STORE
    )

    messages = []

    for item in response["output"]:
        if item["type"] == "message":
            messages.append(item["content"])

    final_text = "\n".join(messages)

    print(final_text)

    manager.unload()

if __name__ == "__main__":
    main()
