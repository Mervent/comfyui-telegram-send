from __future__ import annotations

import io
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from torch import Tensor

MediaList = List[Dict[str, str]]
FileDict = Dict[str, io.BytesIO]
TelegramMedia = Tuple[MediaList, FileDict]


class TelegramSend:
    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu

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
        image_1: Optional[List[Tensor]] = None,
        image_2: Optional[List[Tensor]] = None,
        image_3: Optional[List[Tensor]] = None,
        image_4: Optional[List[Tensor]] = None,
        image_5: Optional[List[Tensor]] = None,
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
            self.call_async(self.send_media_group, args=(bot_token, data, files))
            return (-1,)
        else:
            resp = self.send_media_group(bot_token, data, files)
            return (resp["result"][0]["message_id"],)

    def call_async(
        self,
        callable: Callable[..., Any],
        args: Tuple[Any, ...],
    ) -> threading.Thread:
        thread = threading.Thread(target=callable, args=args, daemon=True)
        thread.start()
        return thread

    def send_media_group(
        self, bot_token: str, data: Dict[str, Any], files: FileDict
    ) -> Dict[str, Any]:
        return self._make_request(
            f"https://api.telegram.org/bot{bot_token}/sendMediaGroup",
            data=data,
            files=files,
            timeout=60,
        )

    def send_message(self, bot_token: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage", data=data
        )

    def _make_request(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        resp = requests.post(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _get_tensors(self, *args: Optional[List[Tensor]]) -> List[Tensor]:
        return [x[0] for x in args if x is not None]

    def _tensors_to_media_group(
        self,
        images: List[Tensor],
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

    def _tensor_to_buffer(self, t: Tensor) -> io.BytesIO:
        # 1)  Quantise on-GPU, keep CHW
        u8 = t.mul(255).clamp_(0, 255).to(torch.uint8)
        if u8.shape[0] not in (1, 3):  # CHW guarantee
            u8 = u8.permute(2, 0, 1).contiguous()

        # 2)  Asynchronous copy into a pinned host buffer
        h = torch.empty_like(u8, device="cpu", pin_memory=True)
        h.copy_(u8, non_blocking=True)  # no stream-wide sync

        # 3)  ZERO-COPY wrap: Image.frombuffer just *views* the memory
        w, h_px = h.shape[2], h.shape[1]

        # Handle MPS tensor issue on Mac by moving to CPU before any operations if needed
        if self.force_cpu:
            h = h.cpu()

        im = Image.frombuffer(
            "RGB",
            (w, h_px),
            h.permute(1, 2, 0).contiguous().numpy(),  # expose a C-contiguous view
            "raw",
            "RGB",
            0,
            1,  # <-- no extra copy
        )

        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1, optimize=False)
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

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("reply_to_message_id", "reply_id")
    FUNCTION = "run"
    CATEGORY = "api/telegram"
    OUTPUT_NODE = True

    def run(
        self,
        bot_token: str,
        chat_id: str,
        reply_to: int,
        reply_to_message_id: Optional[int] = None,
        image_1: Optional[List[Tensor]] = None,
        image_2: Optional[List[Tensor]] = None,
        image_3: Optional[List[Tensor]] = None,
        image_4: Optional[List[Tensor]] = None,
        image_5: Optional[List[Tensor]] = None,
        text: str = "",
        as_document: bool = False,
        use_async: bool = True,
    ) -> Tuple[int, int]:
        if not reply_to_message_id:
            reply_to_message_id = self._find_reply_to_message_id(bot_token, reply_to)

        if not reply_to_message_id:
            raise ValueError(f"Telegram: Could not find reply to message {reply_to}")

        tensors = self._get_tensors(image_1, image_2, image_3, image_4, image_5)
        media, files = self._tensors_to_media_group(tensors, text, as_document)
        if tensors:
            data = {
                "chat_id": chat_id,
                "reply_to_message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
                "media": json.dumps(media, ensure_ascii=False),
            }
            if use_async:
                self.call_async(self.send_media_group, args=(bot_token, data, files))
                return (reply_to_message_id, -1)
            else:
                resp = self.send_media_group(bot_token, data, files)
                return (reply_to_message_id, resp["result"][0]["message_id"])

        data = {
            "chat_id": chat_id,
            "reply_to_message_id": reply_to_message_id,
            "allow_sending_without_reply": True,
            "text": text,
            "parse_mode": "HTML",
        }
        if use_async:
            self.call_async(self.send_message, args=(bot_token, data))
            return (reply_to_message_id, -1)
        else:
            resp = self.send_message(bot_token, data)
            return (reply_to_message_id, resp["result"]["message_id"])

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
