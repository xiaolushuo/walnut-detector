#!/bin/bash

# 核桃检测程序一键安装脚本

echo "=== 核桃检测程序安装脚本 ==="
echo

# 检查Python版本
echo "检查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查pip
echo "检查pip..."
python3 -m pip --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到pip，请先安装pip"
    exit 1
fi

# 安装依赖包
echo "安装依赖包..."
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ 依赖包安装成功"
else
    echo "❌ 依赖包安装失败"
    echo "尝试使用 --user 参数安装..."
    python3 -m pip install --user -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ 依赖包安装成功（用户模式）"
    else
        echo "❌ 依赖包安装失败，请手动安装"
        exit 1
    fi
fi

echo
echo "=== 安装完成 ==="
echo "使用方法："
echo "1. 确保核桃图片文件名为 test_output.jpg"
echo "2. 运行: python3 walnut_detector.py"
echo "3. 查看生成的检测结果文件"
echo