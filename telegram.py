import io
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image


class TelegramSend:
    @classmethod
    def INPUT_TYPES(cls):
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

    # ------------------------------------------------------------
    def run(
        self,
        bot_token: str,
        channel_id: str,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        caption: str = "",
        as_document: bool = False,
    ) -> Tuple[int]:
        images = [
            ctx[0] for ctx in (image_1, image_2, image_3, image_4, image_5) if ctx
        ]

        if images:
            media, files = self._tensors_to_media_group(images, caption, as_document)
            r = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
                data={
                    "chat_id": channel_id,
                    "media": json.dumps(media, ensure_ascii=False),
                },
                files=files,
                timeout=60,
            )
            r.raise_for_status()
            return (r.json()["result"][0]["message_id"],)

        raise ValueError("Nothing to send")

    # ------------------------------------------------------------
    def _tensors_to_media_group(self, images, caption: str, as_document: bool):
        media: List[Dict] = []
        files: Dict[str, io.BytesIO] = {}

        for idx, tensor in enumerate(images):
            buffer = self._tensor_to_buffer(tensor)
            fname = f"img{idx}.png"
            item = {
                "type": "document" if as_document else "photo",
                "media": f"attach://{fname}",
            }
            if idx == 0 and caption:
                item["caption"] = caption
                item["parse_mode"] = "HTML"
            media.append(item)
            files[fname] = buffer

        return media, files

    def _tensor_to_buffer(self, tensor) -> io.BytesIO:
        i = 255.0 * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    @classmethod
    def IS_CHANGED(cls, *_, **__):
        return time.time()


class TelegramReply(TelegramSend):
    """Reply to an existing Telegram message."""

    @classmethod
    def INPUT_TYPES(cls):
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

    # ------------------------------------------------------------
    def run(
        self,
        bot_token: str,
        chat_id: str,
        reply_to: int,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        text: str = "",
        as_document: bool = False,
    ) -> Dict:
        images = [
            ctx[0] for ctx in (image_1, image_2, image_3, image_4, image_5) if ctx
        ]

        # ── images present → sendMediaGroup ---------------------
        if images:
            media, files = self._tensors_to_media_group(images, text, as_document)
            r = requests.post(
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
            r.raise_for_status()
            return {}

        # ── no images → plain text reply ------------------------
        if text.strip():
            r = requests.post(
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
            r.raise_for_status()
            return {}

        raise ValueError("Nothing to send")

    @classmethod
    def IS_CHANGED(cls, *_, **__):
        return time.time()


# ────────────────────────────────────────────────────────────────
#  COMFYUI REGISTRATION
# ────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "TelegramSend": TelegramSend,
    "TelegramReply": TelegramReply,
}
