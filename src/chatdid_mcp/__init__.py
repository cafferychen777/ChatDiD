"""
ChatDiD MCP Server

A Model Context Protocol server for interactive Difference-in-Differences analysis.
Enables users to perform robust DID analysis through natural language conversations.
"""

__version__ = "1.0.0"
__author__ = "ChatDiD Team"
__email__ = "team@chatdid.com"

# CRITICAL: Configure matplotlib backend BEFORE any imports to prevent macOS Dock icon
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI, no Dock icon on macOS)

# Disable R graphics devices (prevent GUI windows)
import os
os.environ['R_DEFAULT_DEVICE'] = 'pdf'  # Use non-interactive device for R

# For FastMCP 2.0, the mcp instance is exported from server.py
# FastMCP CLI will automatically discover and use it
from .server import mcp

__all__ = ["mcp"]
