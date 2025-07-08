from unittest.mock import Mock

from comfyu_telegram.nodes import TelegramReply


class TestTelegramReply:
    def test_input_types(self):
        """Test INPUT_TYPES returns correct structure."""
        input_types = TelegramReply.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types
        assert "reply_to" in input_types["required"]
        assert "bot_token" in input_types["required"]
        assert "chat_id" in input_types["required"]
        assert "text" in input_types["optional"]
        assert "reply_to_message_id" in input_types["optional"]

    def test_return_types(self):
        """Test RETURN_TYPES and RETURN_NAMES are correct."""
        assert TelegramReply.RETURN_TYPES == ("INT", "INT")
        assert TelegramReply.RETURN_NAMES == ("reply_to_message_id", "reply_id")

    def test_find_reply_to_message_id_success(
        self, mock_requests, telegram_updates_response
    ):
        """Test _find_reply_to_message_id finds the correct message."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        reply_to = 789
        mock_requests.get.return_value.json.return_value = telegram_updates_response

        result = telegram_reply._find_reply_to_message_id(bot_token, reply_to)

        assert result == 456
        mock_requests.get.assert_called_with(
            f"https://api.telegram.org/bot{bot_token}/getUpdates", params={"offset": -1}
        )

    def test_run_with_images_sync(self, mock_requests, sample_tensor):
        """Test run method with images in sync mode."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789
        reply_to_message_id = 456

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_message_id=reply_to_message_id,
            image_1=[sample_tensor],
            text="Test reply",
            use_async=False,
        )

        assert result == (456, 123)
        mock_requests.post.assert_called_once()

        # Check that reply_to_message_id is in the data
        call_args = mock_requests.post.call_args
        assert call_args[1]["data"]["reply_to_message_id"] == 456
        assert call_args[1]["data"]["allow_sending_without_reply"] is True

    def test_run_with_images_async(self, mock_requests, sample_tensor):
        """Test run method with images in async mode."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789
        reply_to_message_id = 456

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_message_id=reply_to_message_id,
            image_1=[sample_tensor],
            text="Test reply",
            use_async=True,
        )

        assert result == (456, -1)

    def test_run_text_only_sync(self, mock_requests):
        """Test run method with text only in sync mode."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789
        reply_to_message_id = 456

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_message_id=reply_to_message_id,
            text="Test reply",
            use_async=False,
        )

        assert result == (456, 123)
        mock_requests.post.assert_called_once()

        call_args = mock_requests.post.call_args
        assert "sendMessage" in call_args[0][0]
        assert call_args[1]["data"]["text"] == "Test reply"
        assert call_args[1]["data"]["parse_mode"] == "HTML"

    def test_run_text_only_async(self, mock_requests):
        """Test run method with text only in async mode."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789
        reply_to_message_id = 456

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_message_id=reply_to_message_id,
            text="Test reply",
            use_async=True,
        )

        assert result == (456, -1)

    def test_run_finds_reply_to_message_id_when_not_provided(
        self, mock_requests, telegram_updates_response
    ):
        """Test run method finds reply_to_message_id when not provided."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789

        # Mock GET response for getUpdates
        mock_get_response = Mock()
        mock_get_response.json.return_value = telegram_updates_response
        mock_requests.get.return_value = mock_get_response

        # Mock POST response for sendMessage (keep existing structure)
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "ok": True,
            "result": {"message_id": 123},
        }
        mock_post_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_post_response

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            text="Test reply",
            use_async=False,
        )

        assert result == (456, 123)
        mock_requests.get.assert_called_with(
            f"https://api.telegram.org/bot{bot_token}/getUpdates", params={"offset": -1}
        )

    def test_run_with_multiple_images(self, mock_requests, sample_tensor):
        """Test run method with multiple images."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789
        reply_to_message_id = 456

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_message_id=reply_to_message_id,
            image_1=[sample_tensor],
            image_2=[sample_tensor],
            image_3=[sample_tensor],
            text="Test with multiple images",
            use_async=False,
        )

        assert result == (456, 123)
        mock_requests.post.assert_called_once()

        call_args = mock_requests.post.call_args
        assert "sendMediaGroup" in call_args[0][0]
        assert "files" in call_args[1]
        assert len(call_args[1]["files"]) == 3

    def test_run_as_document(self, mock_requests, sample_tensor):
        """Test run method with as_document=True."""
        telegram_reply = TelegramReply(force_cpu=True)
        bot_token = "123456:ABC-DEF"
        chat_id = "-1001234567890"
        reply_to = 789
        reply_to_message_id = 456

        result = telegram_reply.run(
            bot_token=bot_token,
            chat_id=chat_id,
            reply_to=reply_to,
            reply_to_message_id=reply_to_message_id,
            image_1=[sample_tensor],
            as_document=True,
            use_async=False,
        )

        assert result == (456, 123)
        mock_requests.post.assert_called_once()

    def test_is_changed_returns_float(self):
        """Test IS_CHANGED returns a float timestamp."""
        result = TelegramReply.IS_CHANGED()

        assert isinstance(result, float)
        assert result > 0
