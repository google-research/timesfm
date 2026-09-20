# -*- coding: utf-8 -*-
"""Compatibility launcher for Archer's TimesFM MCP connector."""

from timesfm_forecast_service.mcp_server import main, mcp

__all__ = ["main", "mcp"]


if __name__ == "__main__":
  main()
