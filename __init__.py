try:
    from .comfyui_telegram.nodes import TelegramReply, TelegramSend, TelegramSendDocument  # for comfyui
except ImportError:
    from comfyui_telegram.nodes import TelegramReply, TelegramSend, TelegramSendDocument  # for pytest

NODE_CLASS_MAPPINGS = {
    "TelegramSend": TelegramSend,
    "TelegramSendDocument": TelegramSendDocument,
    "TelegramReply": TelegramReply,
}
