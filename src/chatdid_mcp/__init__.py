"""
ChatDiD MCP Server

A Model Context Protocol server for interactive Difference-in-Differences analysis.
Enables users to perform robust DID analysis through natural language conversations.
"""

__version__ = "1.0.0"
__author__ = "ChatDiD Team"
__email__ = "team@chatdid.com"

from .server import main

__all__ = ["main"]
