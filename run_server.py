#!/usr/bin/env python3
"""
Simple script to run the ChatDiD MCP server.
Uses the virtual environment and official MCP SDK.
"""

import sys
import os
from pathlib import Path

# Ensure we're using the virtual environment
venv_path = Path(__file__).parent / "chatdid_env"
if venv_path.exists():
    # Add virtual environment to path
    venv_site_packages = venv_path / "lib" / "python3.12" / "site-packages"
    if venv_site_packages.exists():
        sys.path.insert(0, str(venv_site_packages))

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatdid_mcp.server import main

if __name__ == "__main__":
    main()
