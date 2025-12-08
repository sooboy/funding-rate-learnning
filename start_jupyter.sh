#!/bin/bash
# 启动 Jupyter Lab 的便捷脚本

cd "$(dirname "$0")"
source venv/bin/activate
jupyter lab --no-browser
