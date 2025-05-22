@echo off
REM 设置代码页为UTF-8
chcp 65001 > nul

REM 设置工作目录
cd /d "%~dp0"

REM 启用命令回显以便查看执行过程
@echo on

REM 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python未安装，请先安装Python。
    pause
    exit /b
)

REM 设置UV链接模式为复制模式
set UV_LINK_MODE=copy

REM 检查uv是否已安装
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo uv未安装，请先安装uv。
    pause
    exit /b
)

REM 检查虚拟环境是否存在
if not exist "venv\" (
    echo 虚拟环境不存在，正在创建...
    uv venv venv
)

REM 激活虚拟环境并使用虚拟环境中的Python
call venv\Scripts\activate.bat
set PYTHON=%cd%\venv\Scripts\python.exe

REM 使用uv安装依赖
uv pip install -r requirements.txt

REM 运行主程序（使用虚拟环境中的Python）
"%PYTHON%" down.py

REM 保持窗口打开以便查看输出
pause