# PCB图片转化

**PCB图片转化**是一个将图片转换为立创EDA PCB文件的工具，支持多种图片格式（PNG、JPG、BMP等）。该工具提供丰富的图像二值化处理功能，帮助用户轻松生成个性化的二值化图片以及符合嘉立创EDA文件格式的pcb文件和库文件。

## 项目背景

本项目源自 [PIC2LCEDA](https://github.com/KnightSin/PIC2LCEDA)，并进行了许多优化和功能扩展，添加更多功能。根据原项目GPL-3.0 许可证的要求，本项目将延续使用GPL-3.0 许可证。

## 安装说明

1. **确保环境**：请确保已安装 Python 3.7 或更高版本。
2. **克隆项目**：通过 git 克隆本项目。
3. **安装依赖**：
    在项目目录下执行以下命令以安装所有必要的依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 功能亮点

- **多格式支持**：支持 PNG、JPG、BMP 等常见格式，处理灵活。
- **自定义PCB尺寸与线宽**：可以轻松调整 PCB 尺寸（mm）和线宽（mil），完全符合个人需求。
- **多层设计支持**：支持顶层、底层、丝印层等多层设计，适应复杂项目需求。
- **图像增强与预处理**：
  - 调整图像的 **亮度、对比度、锐化** 等参数，优化效果。
  - **阈值调整**：根据不同需要选择固定或自适应阈值，优化图像转换效果。
  - **图像降噪**：通过平滑处理去除噪点，提高图像质量。
- **实时预览**：支持即时预览效果，帮助用户快速调整设置。
- **支持文件拖放**：简单便捷，支持直接拖放图片文件进行处理。


## 效果图

![zhu.png](https://github.com/huangdea/PCB-/blob/main/img/zhu.png)

<table>
  <tr>
    <td>原图</td>
    <td>转换图</td>
  </tr>
  <tr>
    <td><img src="https://github.com/huangdea/PCB-/blob/main/img/1.png" width="400"></td>
    <td><img src="https://github.com/huangdea/PCB-/blob/main/img/1.%E8%BD%AC%E5%8C%96%E5%90%8E.png" width="400"></td>
  </tr>
</table>

<table>
  <tr>
    <td>原图</td>
    <td>转换图</td>
  </tr>
  <tr>
    <td><img src="https://github.com/huangdea/PCB-/blob/main/img/2.png" width="400"></td>
    <td><img src="https://github.com/huangdea/PCB-/blob/main/img/2%E8%BD%AC%E5%8C%96%E5%90%8E.png" width="400"></td>
  </tr>
</table>

<table>
  <tr>
    <td>原图</td>
    <td>转换图</td>
  </tr>
  <tr>
    <td><img src="https://github.com/huangdea/PCB-/blob/main/img/3.png" width="400"></td>
    <td><img src="https://github.com/huangdea/PCB-/blob/main/img/3%E8%BD%AC%E5%8C%96%E5%90%8E.png" width="400"></td>
  </tr>
</table>


注：图片来源于网络，侵权联系删除。


## 参数说明

- **X/Y最大尺寸**：设定PCB的物理尺寸（单位：mm）。
- **线宽**：设置PCB线条的宽度（单位：mil）。
- **层级选择**：选择所需的PCB层，如顶层、底层等。
- **图像预处理**：
  - **伽马校正**：调整图像的亮度曲线。
  - **平滑处理**：去除图像噪点，提升图像质量。
  - **曝光度**：调节整体亮度。
  - **对比度**：调整图像的明暗对比度。
  - **锐化**：增强图像边缘细节。
- **图像增强**：
  - **直方图均衡化**：优化图像的整体对比度。
  - **CLAHE增强**：局部对比度增强，突出细节。
  - **细节增强**：提升图像细节的清晰度。
  - **边缘增强**：增强图像的边缘效果，突出轮廓。
  - **局部对比度**：增强图像局部区域的对比度。
- **图像阈值**：设定黑白转换的阈值。
- **降噪方法**：多种降噪算法供选择，以减少噪点影响。

## 注意事项

- 在处理大尺寸图片时，建议先进行小尺寸图片的测试。
- **DPI模式**下无法直接生成PCB文件。
- 生成的文件将自动保存在指定目录中。
- 可以通过预览窗口实时查看处理效果，调整设置以获得最佳结果。

## 已知问题

1. PCB文件无法在最新的嘉立创EDA上导入，因为嘉立创EDA更新了PCB文件格式，但库文件可以正常导入并显示。
2. 生成PCB文件的时间可能较长，尤其是在处理较大的图片时。
3. 保存图片时名称中有中文则报错文件过大。

## 开源许可证

本项目采用 **GPL-3.0 许可证**，你可以自由使用、修改和分发本项目的代码。


## 贡献

欢迎提交问题、建议和贡献代码！如果您希望为项目贡献代码， 请提交 **Pull Request**。
