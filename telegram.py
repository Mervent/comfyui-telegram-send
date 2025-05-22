from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from torch import Tensor

ImageCtx = Tuple[Tensor, Any]  # what ComfyUI gives us
MediaList = List[Dict[str, str]]
FileDict = Dict[str, io.BytesIO]
TelegramMedia = Tuple[MediaList, FileDict]


class TelegramSend:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "bot_token": ("STRING",),
                "channel_id": ("STRING",),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "caption": ("STRING",),
                "as_document": ("BOOLEAN", {"default": False, "forceInput": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("message_id",)
    FUNCTION = "run"
    CATEGORY = "api/telegram"

    def run(
        self,
        bot_token: str,
        channel_id: str,
        image_1: Optional[ImageCtx] = None,
        image_2: Optional[ImageCtx] = None,
        image_3: Optional[ImageCtx] = None,
        image_4: Optional[ImageCtx] = None,
        image_5: Optional[ImageCtx] = None,
        caption: str = "",
        as_document: bool = False,
    ) -> Tuple[int]:
        context_list = (image_1, image_2, image_3, image_4, image_5)
        tensors: List[Any] = [x[0] for x in context_list if x]

        if not tensors:
            raise ValueError("TelegramSend: Nothing to send")

        media, files = self._tensors_to_media_group(tensors, caption, as_document)
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
            data={
                "chat_id": channel_id,
                "media": json.dumps(media, ensure_ascii=False),
            },
            files=files,
            timeout=60,
        )
        resp.raise_for_status()
        return (resp.json()["result"][0]["message_id"],)

    def _tensors_to_media_group(
        self,
        images: List[Any],
        caption: str,
        as_document: bool,
    ) -> TelegramMedia:
        media: MediaList = []
        files: FileDict = {}

        for idx, tensor in enumerate(images):
            buf = self._tensor_to_buffer(tensor)
            fname = f"img{idx}.png"

            media_item = {
                "type": "document" if as_document else "photo",
                "media": f"attach://{fname}",
            }
            if idx == 0 and caption:
                media_item["caption"] = caption
                media_item["parse_mode"] = "HTML"

            media.append(media_item)
            files[fname] = buf

        return media, files

    @staticmethod
    def _tensor_to_buffer(tensor: Tensor) -> io.BytesIO:
        arr = (tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)
        return buf

    # ------------------------------------------------------ cache break ——
    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:  # pragma: no cover
        return time.time()


class TelegramReply(TelegramSend):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "reply_to": ("INT",),
                "bot_token": ("STRING", {"default": "", "forceInput": False}),
                "chat_id": ("STRING", {"default": "", "forceInput": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "text": ("STRING",),
                "as_document": ("BOOLEAN", {"default": False, "forceInput": False}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "api/telegram"

    def run(
        self,
        bot_token: str,
        chat_id: str,
        reply_to: int,
        image_1: Optional[ImageCtx] = None,
        image_2: Optional[ImageCtx] = None,
        image_3: Optional[ImageCtx] = None,
        image_4: Optional[ImageCtx] = None,
        image_5: Optional[ImageCtx] = None,
        text: str = "",
        as_document: bool = False,
    ) -> Dict[str, Any]:
        context_list = (image_1, image_2, image_3, image_4, image_5)
        tensors: List[Any] = [x[0] for x in context_list if x]

        if tensors:
            media, files = self._tensors_to_media_group(tensors, text, as_document)
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
                data={
                    "chat_id": chat_id,
                    "reply_to_message_id": reply_to,
                    "allow_sending_without_reply": True,
                    "media": json.dumps(media, ensure_ascii=False),
                },
                files=files,
                timeout=60,
            )
            resp.raise_for_status()
            return {}

        if text.strip():
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={
                    "chat_id": chat_id,
                    "reply_to_message_id": reply_to,
                    "allow_sending_without_reply": True,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=60,
            )
            resp.raise_for_status()
            return {}

        raise ValueError("TelegramReply: Nothing to send")

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "TelegramSend": TelegramSend,
    "TelegramReply": TelegramReply,
}
