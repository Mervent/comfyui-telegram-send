from __future__ import annotations

import io
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from torch import Tensor

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
                "use_async": ("BOOLEAN", {"default": True, "forceInput": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("message_id",)
    FUNCTION = "run"
    CATEGORY = "api/telegram"
    OUTPUT_NODE = True

    def run(
        self,
        bot_token: str,
        channel_id: str,
        image_1: Optional[Tensor] = None,
        image_2: Optional[Tensor] = None,
        image_3: Optional[Tensor] = None,
        image_4: Optional[Tensor] = None,
        image_5: Optional[Tensor] = None,
        caption: str = "",
        as_document: bool = False,
        use_async: bool = True,
    ) -> Tuple[int]:
        tensors = self._get_tensors(image_1, image_2, image_3, image_4, image_5)
        media, files = self._tensors_to_media_group(tensors, caption, as_document)

        data = {
            "chat_id": channel_id,
            "media": json.dumps(media, ensure_ascii=False),
        }

        if use_async:
            resp = self.send_async(bot_token, data, files)
            return (-1,)
        else:
            resp = self.send(bot_token, data, files)
            return (resp.json()["result"][0]["message_id"],)

    def send_async(
        self,
        bot_token: str,
        data: dict,
        files: FileDict,
    ) -> threading.Thread:
        args = (bot_token, data, files)
        thread = threading.Thread(target=self.send, args=args, daemon=True)
        thread.start()
        return thread

    def send(self, bot_token: str, data: dict, files: FileDict):
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
            data=data,
            files=files,
            timeout=60,
        )
        return resp.json()

    def _get_tensors(self, *args: Optional[Tensor]) -> List[Tensor]:
        tensors = [x[0] for x in args if x is not None]
        if not tensors:
            raise ValueError("TelegramSend: Nothing to send")
        return tensors

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

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
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
                "reply_to_message_id": ("INT",),
                "as_document": ("BOOLEAN", {"default": False, "forceInput": False}),
                "use_async": ("BOOLEAN", {"default": True, "forceInput": False}),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("reply_to_message_id",)
    FUNCTION = "run"
    CATEGORY = "api/telegram"
    OUTPUT_NODE = True

    def run(
        self,
        bot_token: str,
        chat_id: str,
        reply_to: int,
        reply_to_message_id: Optional[int] = None,
        image_1: Optional[Tensor] = None,
        image_2: Optional[Tensor] = None,
        image_3: Optional[Tensor] = None,
        image_4: Optional[Tensor] = None,
        image_5: Optional[Tensor] = None,
        text: str = "",
        as_document: bool = False,
        use_async: bool = True,
    ) -> Tuple[int]:
        if not reply_to_message_id:
            reply_to_message_id = self._find_reply_to_message_id(bot_token, reply_to)

        if not reply_to_message_id:
            raise ValueError(f"Telegram: Could not find reply to message {reply_to}")

        tensors = self._get_tensors(image_1, image_2, image_3, image_4, image_5)
        if tensors:
            media, files = self._tensors_to_media_group(tensors, text, as_document)
            data = {
                "chat_id": chat_id,
                "reply_to_message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
                "media": json.dumps(media, ensure_ascii=False),
            }
        elif text.strip():
            data = {
                "chat_id": chat_id,
                "reply_to_message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
                "text": text,
                "parse_mode": "HTML",
            }
        else:
            raise ValueError("Telegram: Nothing to send")

        if use_async:
            self.send_async(bot_token, data, files)
            return (reply_to_message_id,)
        else:
            self.send(bot_token, data, files)
            return (reply_to_message_id,)

    def _find_reply_to_message_id(self, bot_token: str, reply_to: int) -> Optional[int]:
        offset = -1
        for _ in range(30):
            r = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getUpdates",
                params={"offset": offset},
            )
            for upd in r.json()["result"]:
                msg = upd.get("message", {})
                if msg.get("forward_from_message_id") == reply_to:
                    return msg["message_id"]
                offset = max(offset, upd["update_id"])
            time.sleep(1)

        return None

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()
