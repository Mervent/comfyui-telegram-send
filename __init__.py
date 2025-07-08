try:
    from .comfyui_telegram.nodes import TelegramReply, TelegramSend  # for comfyui
except ImportError:
    from comfyui_telegram.nodes import TelegramReply, TelegramSend  # for pytest

NODE_CLASS_MAPPINGS = {
    "TelegramSend": TelegramSend,
    "TelegramReply": TelegramReply,
}
