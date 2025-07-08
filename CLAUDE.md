# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom node plugin that provides Telegram integration for sending images and messages. The plugin allows ComfyUI workflows to send generated images and text to Telegram channels via bot API.

## Architecture

The codebase consists of two main Python files:

- `telegram.py`: Contains the core functionality with two main classes:
  - `TelegramSend`: Sends images and media groups to Telegram channels
  - `TelegramReply`: Replies to specific messages with images or text
- `__init__.py`: Exports the node classes for ComfyUI integration

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

## Development Notes

### Testing
- No formal test framework is present in this codebase
- Testing likely requires integration with ComfyUI and actual Telegram bot tokens

### Dependencies
- `requests`: HTTP requests to Telegram Bot API
- `torch`: Tensor operations and GPU optimization
- `PIL (Pillow)`: Image processing
- ComfyUI framework (implicit dependency)

### Image Processing
- Uses optimized tensor-to-image conversion with GPU acceleration
- Implements zero-copy memory operations for performance
- Converts tensors to PNG format with minimal compression

### Async Operations
- Uses Python threading for non-blocking Telegram API calls
- Async operations return -1 as message_id placeholder
- Daemon threads are used to prevent hanging on exit

## API Integration
- Uses Telegram Bot API endpoints: `sendMediaGroup` and `sendMessage`
- Requires bot_token and chat_id/channel_id for operations
- Supports HTML parsing for message formatting
- Implements error handling with HTTP status checks