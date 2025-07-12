import io
import json
import time
from unittest.mock import Mock

import torch
from PIL import Image

from ..nodes import TelegramSend


class TestTelegramSend:
    def test_input_types(self):
        """Test INPUT_TYPES returns correct structure."""
        input_types = TelegramSend.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types
        assert "bot_token" in input_types["required"]
        assert "channel_id" in input_types["required"]
        assert "image_1" in input_types["optional"]
        assert "caption" in input_types["optional"]

    def test_get_tensors_filters_none_values(self, sample_tensor):
        """Test _get_tensors filters out None values."""
        telegram_send = TelegramSend(force_cpu=True)
        tensor1 = [sample_tensor]
        tensor2 = [sample_tensor]

        result = telegram_send._get_tensors(tensor1, None, tensor2, None, None)

        assert len(result) == 2
        assert torch.equal(result[0], sample_tensor)
        assert torch.equal(result[1], sample_tensor)

    def test_get_tensors_empty_when_all_none(self):
        """Test _get_tensors returns empty list when all inputs are None."""
        telegram_send = TelegramSend(force_cpu=True)

        result = telegram_send._get_tensors(None, None, None, None, None)

        assert result == []

    def test_tensor_to_buffer_creates_png(self, sample_tensor):
        """Test _tensor_to_buffer creates PNG buffer."""
        telegram_send = TelegramSend(force_cpu=True)

        buffer = telegram_send._tensor_to_buffer(sample_tensor)

        assert isinstance(buffer, io.BytesIO)
        buffer.seek(0)
        image = Image.open(buffer)
        assert image.format == "PNG"
        assert image.size == (64, 64)

    def test_tensors_to_media_group_creates_media_and_files(self, sample_tensor):
        """Test _tensors_to_media_group creates media list and files dict."""
        telegram_send = TelegramSend(force_cpu=True)
        tensors = [sample_tensor, sample_tensor]
        caption = "Test caption"

        media, files = telegram_send._tensors_to_media_group(tensors, caption, False)

        assert len(media) == 2
        assert len(files) == 2
        assert media[0]["type"] == "photo"
        assert media[0]["media"] == "attach://img0.png"
        assert media[0]["caption"] == caption
        assert media[0]["parse_mode"] == "HTML"
        assert media[1]["type"] == "photo"
        assert media[1]["media"] == "attach://img1.png"
        assert "caption" not in media[1]
        assert "img0.png" in files
        assert "img1.png" in files

    def test_tensors_to_media_group_as_document(self, sample_tensor):
        """Test _tensors_to_media_group with as_document=True."""
        telegram_send = TelegramSend(force_cpu=True)
        tensors = [sample_tensor]

        media, files = telegram_send._tensors_to_media_group(tensors, "", True)

        assert media[0]["type"] == "document"

    def test_make_request_calls_post_and_returns_json(self, mock_requests):
        """Test _make_request makes POST request and returns JSON."""
        telegram_send = TelegramSend(force_cpu=True)
        url = "https://api.telegram.org/bot123/sendMessage"
        data = {"chat_id": "test", "text": "hello"}

        result = telegram_send._make_request(url, data=data)

        mock_requests.post.assert_called_once_with(url, data=data)
        assert result == {"ok": True, "result": {"message_id": 123}}

    def test_send_media_group_calls_make_request(self, mock_requests):
        """Test send_media_group calls _make_request with correct URL."""
        telegram_send = TelegramSend(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        data = {"chat_id": "test"}
        files = {"img0.png": io.BytesIO(b"test")}

        result = telegram_send.send_media_group(bot_token, data, files)

        expected_url = f"https://api.telegram.org/bot{bot_token}/sendMediaGroup"
        mock_requests.post.assert_called_once_with(
            expected_url, data=data, files=files, timeout=60
        )
        assert result == {"ok": True, "result": [{"message_id": 123}]}

    def test_send_message_calls_make_request(self, mock_requests):
        """Test send_message calls _make_request with correct URL."""
        telegram_send = TelegramSend(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        data = {"chat_id": "test", "text": "hello"}

        result = telegram_send.send_message(bot_token, data)

        expected_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        mock_requests.post.assert_called_once_with(expected_url, data=data)
        assert result == {"ok": True, "result": {"message_id": 123}}

    def test_call_async_submits_to_executor(self):
        """Test call_async submits to thread pool executor."""
        telegram_send = TelegramSend(force_cpu=True)
        mock_callable = Mock()
        args = ("arg1", "arg2")

        telegram_send.call_async(mock_callable, args)

        time.sleep(0.1)
        mock_callable.assert_called_once_with("arg1", "arg2")

    def test_call_async_with_keep_order_true(self):
        """Test call_async with keep_order=True uses ordered executor."""
        telegram_send = TelegramSend(force_cpu=True)
        mock_callable = Mock()
        args = ("arg1", "arg2")

        telegram_send.call_async(mock_callable, args, keep_order=True)

        time.sleep(0.1)
        mock_callable.assert_called_once_with("arg1", "arg2")

    def test_call_async_with_keep_order_false(self):
        """Test call_async with keep_order=False uses parallel executor."""
        telegram_send = TelegramSend(force_cpu=True)
        mock_callable = Mock()
        args = ("arg1", "arg2")

        telegram_send.call_async(mock_callable, args, keep_order=False)

        time.sleep(0.1)
        mock_callable.assert_called_once_with("arg1", "arg2")

    def test_run_sync_returns_message_id(self, mock_requests, sample_tensor):
        """Test run method in sync mode returns message_id."""
        telegram_send = TelegramSend(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        channel_id = "-1001234567890"

        result = telegram_send.run(
            bot_token=bot_token,
            channel_id=channel_id,
            image_1=[sample_tensor],
            caption="Test",
            use_async=False,
        )

        assert result == (123,)
        mock_requests.post.assert_called_once()

    def test_run_async_returns_negative_one(self, mock_requests, sample_tensor):
        """Test run method in async mode returns -1."""
        telegram_send = TelegramSend(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        channel_id = "-1001234567890"

        result = telegram_send.run(
            bot_token=bot_token,
            channel_id=channel_id,
            image_1=[sample_tensor],
            caption="Test",
            use_async=True,
        )

        assert result == (-1,)

    def test_run_without_images(self, mock_requests):
        """Test run method without any images."""
        telegram_send = TelegramSend(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        channel_id = "-1001234567890"

        result = telegram_send.run(
            bot_token=bot_token, channel_id=channel_id, caption="Test", use_async=False
        )

        assert result == (123,)
        mock_requests.post.assert_called_once()

        # Check that media group is empty
        call_args = mock_requests.post.call_args
        assert "media" in call_args[1]["data"]
        media = json.loads(call_args[1]["data"]["media"])
        assert len(media) == 0

    def test_is_changed_returns_float(self):
        """Test IS_CHANGED returns a float timestamp."""
        result = TelegramSend.IS_CHANGED()

        assert isinstance(result, float)
        assert result > 0
