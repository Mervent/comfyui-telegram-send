import io
from unittest.mock import Mock

import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests module for HTTP calls."""
    mock = Mock()

    def mock_post(url, *args, **kwargs):
        response = Mock()
        response.raise_for_status.return_value = None
        # Check if this is a sendMediaGroup call
        if "sendMediaGroup" in url:
            response.json.return_value = {"ok": True, "result": [{"message_id": 123}]}
        else:  # sendMessage
            response.json.return_value = {"ok": True, "result": {"message_id": 123}}
        return response

    mock_get_response = Mock()
    mock_get_response.json.return_value = {"ok": True, "result": [{"message_id": 123}]}
    mock_get_response.raise_for_status.return_value = None

    mock.post.side_effect = mock_post
    mock.get.return_value = mock_get_response

    monkeypatch.setattr("comfyu_telegram.nodes.requests", mock)
    return mock


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.rand(3, 64, 64)


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (64, 64), color="red")


@pytest.fixture
def sample_buffer():
    """Create a sample BytesIO buffer with PNG data."""
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def telegram_api_response():
    """Sample Telegram API response."""
    return {
        "ok": True,
        "result": [
            {
                "message_id": 123,
                "from": {"id": 123456789, "is_bot": True},
                "chat": {"id": -1001234567890, "type": "channel"},
                "date": 1234567890,
                "photo": [{"file_id": "test_file_id"}],
            }
        ],
    }


@pytest.fixture
def telegram_updates_response():
    """Sample Telegram getUpdates response."""
    return {
        "ok": True,
        "result": [
            {
                "update_id": 123456789,
                "message": {
                    "message_id": 456,
                    "from": {"id": 987654321, "is_bot": False},
                    "chat": {"id": -1001234567890, "type": "channel"},
                    "date": 1234567890,
                    "forward_from_message_id": 789,
                    "text": "Test message",
                },
            }
        ],
    }


@pytest.fixture
def bot_token():
    """Sample bot token."""
    return "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"


@pytest.fixture
def channel_id():
    """Sample channel ID."""
    return "-1001234567890"
