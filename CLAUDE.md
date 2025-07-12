# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom node plugin that provides Telegram integration for sending images and messages. The plugin allows ComfyUI workflows to send generated images and text to Telegram channels via bot API.

## Development Commands

### Environment Setup
```bash
make venv  # Create virtual environment and install dependencies with uv
```

### Testing
```bash
make test      # Run all tests with pytest
pytest -v ./tests/test_telegram_send.py  # Run single test file
pytest -v ./tests/test_telegram_reply.py  # Run reply tests
```

### Code Quality
```bash
make lint      # Check code with ruff and ty
make format    # Format code with ruff
```

## Architecture

The codebase structure:
- `comfyui_telegram/nodes.py`: Core functionality with TelegramSend and TelegramReply classes
- `__init__.py`: ComfyUI node registration and exports
- `comfyui_telegram/tests/`: Comprehensive test suite with mocked HTTP requests
- `pyproject.toml`: Python project configuration with dependencies and pytest settings

## Key Components

### TelegramSend Class
- ComfyUI node for sending up to 5 images as media groups
- Supports both synchronous and asynchronous sending (via `use_async` parameter)
- Can send images as photos or documents
- Handles tensor-to-image conversion with GPU optimization
- Returns message_id for successful sends (-1 for async operations)

### TelegramReply Class
- Extends TelegramSend to reply to specific messages
- Can find the original message to reply to by searching recent updates
- Supports both image and text replies
- Maintains reply chain functionality

## Testing Framework

- Uses pytest with comprehensive fixtures in `comfyui_telegram/tests/conftest.py`
- Mock objects for HTTP requests to avoid actual Telegram API calls
- Fixtures for sample tensors, images, and API responses
- Tests cover both synchronous and asynchronous operations
- Test configuration in `pyproject.toml` specifies test paths and options

## Dependencies

Core dependencies from pyproject.toml:
- `requests>=2.25.0`: HTTP requests to Telegram Bot API
- `torch>=1.9.0`: Tensor operations and GPU optimization
- `Pillow>=8.0.0`: Image processing
- `numpy>=2.2.0,<2.3.0`: Numerical operations and array handling
- `pytest>=8.3.5`: Testing framework
- `pytest-mock>=3.14.1`: Mock objects for testing
- `ruff>=0.12.2`: Code linting and formatting
- `ty>=0.0.1a14`: Type checking

## Image Processing
- Uses optimized tensor-to-image conversion with GPU acceleration
- Implements zero-copy memory operations for performance
- Converts tensors to PNG format with minimal compression

## Async Operations
- Uses dual ThreadPoolExecutor system for non-blocking Telegram API calls
- Two executors: `_executor_ordered` (1 thread) and `_executor_parallel` (5 threads)
- `keep_order` parameter (default: False) controls which executor to use:
  - `keep_order=True`: Uses single-thread executor for sequential message sending
  - `keep_order=False`: Uses 5-thread executor for parallel processing
- Async operations return -1 as message_id placeholder

## API Integration
- Uses Telegram Bot API endpoints: `sendMediaGroup` and `sendMessage`
- Requires bot_token and chat_id/channel_id for operations
- Supports HTML parsing for message formatting
- Implements error handling with HTTP status checks
- Reply functionality uses `getUpdates` endpoint to find messages to reply to

## ComfyUI Node Registration
- Nodes are registered in `__init__.py` via `NODE_CLASS_MAPPINGS`
- Both nodes are output nodes (`OUTPUT_NODE = True`) for triggering workflows
- Uses `IS_CHANGED` method returning `time.time()` to ensure re-execution