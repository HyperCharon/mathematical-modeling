#!/bin/bash
cd "$(dirname "$0")"
sphinx-build -b html source build/html
echo "文档已生成: docs/build/html/index.html"
