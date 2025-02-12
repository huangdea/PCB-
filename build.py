import PyInstaller.__main__
import os

PyInstaller.__main__.run([
    'main.py',                # 主程序文件
    '--name=PCB图片转化v1.1',    # 生成的exe名称
    '--windowed',             # 使用GUI模式，不显示控制台
    '--onefile',              # 打包成单个exe文件
    '--clean',                # 清理临时文件
    '--noconfirm',            # 不询问确认
    '--icon=icons/ico.png',   # 添加图标
]) 
