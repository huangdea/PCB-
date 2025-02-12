from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLineEdit, QWidget
from PyQt5.QtGui import QIcon, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt
import sys
import ui
import PIC2LCEDA
import logging
import os
import cv2

class DragDropLineEdit(QLineEdit):
    """支持拖放的输入框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖入文件时"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """放下文件时"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.setText(file_path)
            # 触发文本改变信号
            self.textChanged.emit(file_path)

class MainWindow(QMainWindow):
    # 添加类常量
    DEFAULT_ICON = "img/ico.png"
    DEFAULT_X_SIZE = 50.0
    DEFAULT_Y_SIZE = 50.0
    DEFAULT_WIDTH = 5
    DEFAULT_THRESHOLD = 150  # 修改默认阈值
    DEFAULT_PREVIEW_WIDTH = 800  # 默认预览窗口宽度
    DEFAULT_PREVIEW_HEIGHT = 600
    MIN_WINDOW_SIZE = 200  # 最小窗口尺寸
    MAX_IMAGE_DIMENSION = 3840  # 限制最大分辨率为4K
    
    def __init__(self):
        super().__init__()
        # 初始化成员变量
        self._image_loaded = False
        self._preview_window_created = False  # 添加预览窗口状态标记
        
        # 添加新的成员变量
        self._denoise_method = 0
        self._equalize_hist = False
        self._clahe = False
        
        self._setup_logging()
        self.ui = ui.Ui_ui()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 设置UI
        self.ui.setupUi(central_widget)
        
        # 替换原有的输入框为支持拖放的输入框
        self.drag_drop_input = DragDropLineEdit(self)
        # 获取原输入框的布局位置和大小
        original_geometry = self.ui.sourcefullpath.geometry()
        self.drag_drop_input.setGeometry(original_geometry)
        self.drag_drop_input.setText(self.ui.sourcefullpath.text())
        
        # 获取原输入框的父级布局
        parent_layout = self.ui.sourcefullpath.parent().layout()
        # 获取水平布局
        horizontal_layout = self.ui.horizontalLayout
        # 移除原输入框
        horizontal_layout.removeWidget(self.ui.sourcefullpath)
        self.ui.sourcefullpath.setParent(None)
        # 在水平布局中添加新的输入框
        horizontal_layout.insertWidget(0, self.drag_drop_input)
        # 更新引用
        self.ui.sourcefullpath = self.drag_drop_input
        
        # 设置输出路径的拖放支持
        self.output_drag_drop_input = DragDropLineEdit(self)
        original_geometry = self.ui.outputpath.geometry()
        self.output_drag_drop_input.setGeometry(original_geometry)
        
        # 替换输出路径输入框
        horizontal_layout_output = self.ui.horizontalLayout_output
        horizontal_layout_output.removeWidget(self.ui.outputpath)
        self.ui.outputpath.setParent(None)
        horizontal_layout_output.insertWidget(0, self.output_drag_drop_input)
        self.ui.outputpath = self.output_drag_drop_input
        
        self.setWindowIcon(QIcon(self.DEFAULT_ICON))
        self._connect_signals()
        self._setup_ui()  # 确保在初始化时调用
        
        # 设置接受拖放
        self.setAcceptDrops(True)
        
    def _setup_logging(self):
        """配置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='pic2lceda.log'
        )
        
    def _connect_signals(self):
        """连接所有信号槽"""
        self.ui.btn_convert.clicked.connect(self.on_clicked_btn_convert)
        self.ui.btn_getfile.clicked.connect(self.on_clicked_btn_getfile)
        self.ui.thresholdSlider.valueChanged.connect(self.thresholdSlider_valueChanged)
        self.ui.denoiseSlider.valueChanged.connect(self.denoiseSlider_valueChanged)
        self.ui.color_invert_f.clicked.connect(self.refreshImg)
        self.ui.x_invert_f.clicked.connect(self.refreshImg)
        self.ui.y_invert_f.clicked.connect(self.refreshImg)
        self.ui.width.textChanged.connect(self.refreshImg)
        self.ui.x_size.textChanged.connect(self.refreshImg)
        self.ui.y_size.textChanged.connect(self.refreshImg)
        
        # 修改文件路径输入框的信号连接
        self.ui.sourcefullpath.textChanged.connect(self._on_sourcepath_changed)
        self.ui.btn_getoutputpath.clicked.connect(self.on_clicked_btn_getoutputpath)
        self.ui.btn_dpi_mode.clicked.connect(self._on_dpi_mode_changed)
        self.ui.dpi_select.currentIndexChanged.connect(self.refreshImg)
        self.ui.btn_save_image.clicked.connect(self.on_clicked_btn_save_image)
        self.ui.exposure_slider.valueChanged.connect(self._on_preprocess_changed)
        self.ui.contrast_slider.valueChanged.connect(self._on_preprocess_changed)
        self.ui.sharpen_slider.valueChanged.connect(self._on_preprocess_changed)
        self.ui.gamma_slider.valueChanged.connect(self._on_preprocess_changed)
        self.ui.smooth_slider.valueChanged.connect(self._on_preprocess_changed)
        
        # 添加新的信号连接
        self.ui.denoise_method.currentIndexChanged.connect(self._on_denoise_method_changed)
        self.ui.equalize_hist.stateChanged.connect(self._on_equalize_changed)
        self.ui.clahe.stateChanged.connect(self._on_clahe_changed)
        
        # 确保降噪滑块变化时使用新的降噪方法
        self.ui.denoiseSlider.valueChanged.disconnect(self.denoiseSlider_valueChanged)
        self.ui.denoiseSlider.valueChanged.connect(self._on_denoise_changed)
        
        # 添加新的信号连接
        self.ui.detail_slider.valueChanged.connect(self._on_enhance_changed)
        self.ui.edge_slider.valueChanged.connect(self._on_enhance_changed)
        self.ui.local_contrast_slider.valueChanged.connect(self._on_enhance_changed)

    def _setup_ui(self):
        """设置UI初始状态"""
        # 设置默认值
        self.ui.x_size.setValue(self.DEFAULT_X_SIZE)
        self.ui.y_size.setValue(self.DEFAULT_Y_SIZE)
        self.ui.width.setValue(self.DEFAULT_WIDTH)
        self.ui.thresholdSlider.setValue(self.DEFAULT_THRESHOLD)
        self.ui.threshold_label.setText(str(self.DEFAULT_THRESHOLD))
        self.ui.denoiseSlider.setValue(0)
        self.ui.dpi_select.setCurrentIndex(12)  # 12对应20000 DPI
        
        # 设置阈值方法为自适应阈值
        self.ui.threshold_method.setCurrentIndex(1)  # 1 表示自适应阈值
        
        # DPI模式默认关闭
        self.ui.btn_dpi_mode.setChecked(False)
        self.ui.dpi_select.setEnabled(False)
        
        # 设置默认层级
        self.ui.layer.setCurrentIndex(6)  # 默认选择顶层阻焊层
        
        # 设置图像加载标志
        self._image_loaded = False
        self._preview_window_created = False

    def _get_layer(self):
        """获取当前层号"""
        layer_index = int(self.ui.layer.currentIndex())
        return layer_index + 1 if layer_index < 8 else layer_index + 2

    def _show_message(self, title, text, info_text):
        """显示消息框"""
        msg = QMessageBox()
        msg.setWindowIcon(QIcon("img/ico.png"))
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setInformativeText(info_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.show()
        return msg.exec_()

    def _get_current_params(self):
        """获取当前所有参数"""
        try:
            params = {
                'sourcefullpath': self.ui.sourcefullpath.text().strip(),
                'x_size': float(self.ui.x_size.text()),
                'y_size': float(self.ui.y_size.text()),
                'layer': self._get_layer(),
                'color_invert_f': int(self.ui.color_invert_f.checkState()),
                'x_invert_f': int(self.ui.x_invert_f.checkState()),
                'y_invert_f': int(self.ui.y_invert_f.checkState()),
                'copper_f': int(self.ui.copper_f.checkState()),  # 添加铜皮参数
                'threshold': self.ui.thresholdSlider.value(),
                'denoise': self.ui.denoiseSlider.value(),
                'threshold_method': self.ui.threshold_method.currentIndex(),
                # 预处理参数
                'exposure': self.ui.exposure_slider.value(),
                'contrast': self.ui.contrast_slider.value(),
                'sharpen': self.ui.sharpen_slider.value(),
                'gamma': self.ui.gamma_slider.value() / 100.0,
                'smooth': self.ui.smooth_slider.value(),
                'output_path': self.ui.outputpath.text().strip(),
                # 添加新的参数
                'denoise_method': self._denoise_method,
                'denoise_strength': self.ui.denoiseSlider.value(),
                'equalize': self._equalize_hist,
                'clahe': self._clahe,
                'detail_enhance': self.ui.detail_slider.value(),
                'edge_enhance': self.ui.edge_slider.value(),
                'local_contrast': self.ui.local_contrast_slider.value(),
            }
            
            # 如果没有选择阈值方法，使用自适应阈值
            if 'threshold_method' not in params:
                params['threshold_method'] = 1
            
            # 根据模式设置宽度参数
            if self.ui.btn_dpi_mode.isChecked():
                dpi_values = ['original', 125, 300, 600, 1200, 2400, 3600, 5000, 7000, 9000, 12000, 15000, 20000]
                dpi_index = self.ui.dpi_select.currentIndex()
                if dpi_index == 0:  # 原图模式
                    params['use_original_size'] = True
                    params['width'] = 1  # 默认线宽
                else:
                    params['use_original_size'] = False
                    params['width'] = int(25400 / dpi_values[dpi_index])  # 将DPI转换为mil
            else:
                params['use_original_size'] = False
                params['width'] = int(self.ui.width.text())
            
            return params
        except (ValueError, TypeError) as e:
            self._show_message("错误", "参数格式错误", str(e))
            return None

    def _create_preview_window(self, img):
        """创建或更新预览窗口"""
        # 确保窗口存在
        if not self._preview_window_created:
            try:
                cv2.destroyWindow('preview')  # 关闭可能存在的窗口
                cv2.waitKey(1)
            except cv2.error:
                pass
        
            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
            self._preview_window_created = True
            cv2.waitKey(1)  # 确保窗口创建完成
        
        # 获取图像尺寸
        img_height, img_width = img.shape[:2]
        
        # 计算合适的显示尺寸
        scale = self._calculate_display_scale(img_width, img_height)
        display_size = self._calculate_display_size(img_width, img_height, scale)
        
        try:
            # 设置窗口大小
            cv2.resizeWindow('preview', *display_size)
            cv2.waitKey(1)  # 确保窗口大小设置完成
        except cv2.error:
            # 如果调整大小失败，重新创建窗口
            self._preview_window_created = False
            return self._create_preview_window(img)
        
        return scale

    def _calculate_display_scale(self, img_width, img_height):
        """计算显示缩放比例"""
        width_ratio = self.DEFAULT_PREVIEW_WIDTH / img_width
        height_ratio = self.DEFAULT_PREVIEW_HEIGHT / img_height
        return min(width_ratio, height_ratio)

    def _calculate_display_size(self, img_width, img_height, scale):
        """计算显示尺寸"""
        width = max(self.MIN_WINDOW_SIZE, int(img_width * scale))
        height = max(self.MIN_WINDOW_SIZE, int(img_height * scale))
        return width, height

    def refreshImg(self):
        """刷新预览图片"""
        if not self._image_loaded:
            return
            
        preview_params = self._get_current_params()
        if not preview_params:
            return
            
        # 移除不需要的参数
        preview_params.pop('output_path', None)
        preview_params.pop('copper_f', None)  # 预览时不需要铜皮参数
        
        # 调用图像转换函数
        result = PIC2LCEDA.transformpic(**preview_params)
        
        if result == -1:
            self._show_message("温馨提示", "图片不存在", 
                              "可能还没有选择图片，\n或者选择的图片神秘消失了Σ(ﾟдﾟ;)")
            return
            
        img = result[0]
        
        # 创建预览窗口
        scale = self._create_preview_window(img)
        
        # 显示预览图
        PIC2LCEDA.showImg(img, scale)

    def thresholdSlider_valueChanged(self):
        """阈值滑块值变化处理"""
        self.ui.threshold_label.setText(str(self.ui.thresholdSlider.value()))
        self.refreshImg()

    def denoiseSlider_valueChanged(self):
        """降噪滑块值变化处理"""
        self.ui.denoise_label.setText(str(self.ui.denoiseSlider.value()))
        self.refreshImg()

    def on_clicked_btn_getfile(self):
        """选择文件按钮处理"""
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.bmp);;所有文件 (*.*)"
        )
        if fileName:
            self.ui.sourcefullpath.setText(fileName)
            self._image_loaded = True
            self.refreshImg()

    def on_clicked_btn_convert(self):
        """生成文件按钮处理"""
        params = self._get_current_params()
        if not params:
            return
        
        # 复制参数并移除预览专用的参数
        convert_params = params.copy()
        convert_params.pop('use_original_size', None)  # 移除原图尺寸标记
        
        # 调用转换函数
        if PIC2LCEDA.makepcb(**convert_params) == -1:
            self._show_message("错误", "转换失败", 
                              "可能是因为图片不存在，\n或者选择的图片神秘消失了")
        else:
            self._show_message("成功", "转换完成", 
                              "文件已保存到指定目录")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """窗口拖入文件时"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """窗口放下文件时"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.ui.sourcefullpath.setText(file_path)
            self._image_loaded = True
            self.refreshImg()

    def _on_sourcepath_changed(self, text):
        """文件路径改变时"""
        if text and os.path.isfile(text):
            self._image_loaded = True
            # 自动设置输出路径为图片所在目录
            source_dir = os.path.dirname(text)
            self.ui.outputpath.setText(source_dir)
            self.refreshImg()
            
    def on_clicked_btn_getoutputpath(self):
        """选择输出目录按钮"""
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.ui.outputpath.text() if self.ui.outputpath.text() else ""
        )
        if output_dir:
            self.ui.outputpath.setText(output_dir)

    def _on_dpi_mode_changed(self):
        """DPI模式切换"""
        self._update_width_mode()
        self.refreshImg()

    def _update_width_mode(self):
        """更新宽度模式"""
        is_dpi_mode = self.ui.btn_dpi_mode.isChecked()
        self.ui.width.setEnabled(not is_dpi_mode)
        self.ui.dpi_select.setEnabled(is_dpi_mode)
        # 在DPI模式下禁用生成文件按钮
        self.ui.btn_convert.setEnabled(not is_dpi_mode)
        
        # 根据模式更新按钮文本
        if is_dpi_mode:
            self.ui.label_4.setText("DPI")
            self.ui.btn_dpi_mode.setText("切换线宽模式")
        else:
            self.ui.label_4.setText("线宽/mil")
            self.ui.btn_dpi_mode.setText("切换DPI模式")

    def on_clicked_btn_save_image(self):
        """生成图片按钮处理"""
        try:
            params = self._get_current_params()
            if not params:
                return
            
            # 获取预览图像
            preview_params = params.copy()
            preview_params.pop('output_path', None)
            preview_params.pop('copper_f', None)  # 移除铜皮参数
            result = PIC2LCEDA.transformpic(**preview_params)
            
            if result == -1:
                self._show_message("错误", "图片不存在", 
                                  "可能还没有选择图片，\n或者选择的图片神秘消失了")
                return
            
            img = result[0]
            
            # 如果是DPI模式，使用不同的处理方法
            if self.ui.btn_dpi_mode.isChecked():
                dpi_values = ['original', 125, 300, 600, 1200, 2400, 3600, 5000, 7000, 9000, 12000, 15000, 20000]
                dpi_index = self.ui.dpi_select.currentIndex()
                if dpi_index > 0:  # 非原图模式
                    try:
                        dpi = int(dpi_values[dpi_index])
                    except (ValueError, IndexError):
                        self._show_message("错误", "DPI值无效", 
                                         "请选择有效的DPI值")
                        return
                    
                    try:
                        # 根据DPI调整图像大小
                        target_width = int(params['x_size'] / 25.4 * dpi)
                        target_height = int(params['y_size'] / 25.4 * dpi)
                        
                        # 检查尺寸是否为0或负数
                        if target_width <= 0 or target_height <= 0:
                            self._show_message("错误", "图像尺寸无效", 
                                             "计算得到的图像尺寸必须大于0")
                            return
                            
                        # 检查尺寸是否过小
                        if target_width < 10 or target_height < 10:
                            self._show_message("错误", "图像尺寸过小", 
                                             "请增加尺寸或提高DPI")
                            return
                        
                        # 检查并限制图像尺寸
                        if target_width > self.MAX_IMAGE_DIMENSION or target_height > self.MAX_IMAGE_DIMENSION:
                            # 计算缩放比例
                            scale = min(self.MAX_IMAGE_DIMENSION / target_width, 
                                      self.MAX_IMAGE_DIMENSION / target_height)
                            target_width = int(target_width * scale)
                            target_height = int(target_height * scale)
                        
                        # 检查内存是否足够
                        required_memory = target_width * target_height * 4  # 估算所需内存(RGBA)
                        if required_memory > 2 * 1024 * 1024 * 1024:  # 2GB
                            self._show_message("错误", "内存不足", 
                                             "预计所需内存过大，请降低DPI或减小尺寸")
                            return
                        
                        img = cv2.resize(img, (target_width, target_height), 
                                       interpolation=cv2.INTER_NEAREST)
                        
                    except cv2.error as e:
                        self._show_message("错误", "图像处理失败", 
                                         "调整图像大小时出错，请尝试降低DPI或减小尺寸")
                        return
                    except ValueError as e:
                        self._show_message("错误", "参数错误", 
                                         f"计算图像尺寸时出错: {str(e)}")
                        return
                    except Exception as e:
                        self._show_message("错误", "未知错误", 
                                         f"处理图像时发生错误: {str(e)}")
                        return
            
            # 保存图片对话框
            try:
                fileName, fileType = QFileDialog.getSaveFileName(
                    self,
                    "保存图片",
                    os.path.join(params['output_path'], "output"),
                    "PNG图片 (*.png);;BMP图片 (*.bmp);;JPEG图片 (*.jpg);;TIFF图片 (*.tiff);;所有文件 (*.*)"
                )
                
                if fileName:
                    # 确保文件名有正确的扩展名
                    if fileType == "PNG图片 (*.png)" and not fileName.lower().endswith('.png'):
                        fileName += '.png'
                    elif fileType == "BMP图片 (*.bmp)" and not fileName.lower().endswith('.bmp'):
                        fileName += '.bmp'
                    elif fileType == "JPEG图片 (*.jpg)" and not fileName.lower().endswith(('.jpg', '.jpeg')):
                        fileName += '.jpg'
                    elif fileType == "TIFF图片 (*.tiff)" and not fileName.lower().endswith(('.tiff', '.tif')):
                        fileName += '.tiff'
                    
                    # 检查输出目录是否存在
                    output_dir = os.path.dirname(fileName)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # 检查文件是否可写
                    if os.path.exists(fileName):
                        try:
                            with open(fileName, 'a'):
                                pass
                        except IOError:
                            self._show_message("错误", "文件访问被拒绝", 
                                             "无法写入选择的文件，请检查文件权限或选择其他位置")
                            return
                    
                    # 保存图片
                    success = cv2.imwrite(fileName, img)
                    
                    if success:
                        self._show_message("成功", "图片已保存", f"保存路径：{fileName}")
                    else:
                        self._show_message("错误", "保存失败", 
                                         "图片保存失败，请检查磁盘空间或选择其他格式")
                
            except Exception as e:
                self._show_message("错误", "保存失败", 
                                 f"保存图片时发生错误: {str(e)}")
            
        except Exception as e:
            self._show_message("错误", "程序错误", 
                              f"发生未预期的错误: {str(e)}")

    def _on_preprocess_changed(self):
        """预处理参数变化处理"""
        # 更新标签显示
        self.ui.exposure_label.setText(str(self.ui.exposure_slider.value()))
        self.ui.contrast_label.setText(str(self.ui.contrast_slider.value()))
        self.ui.sharpen_label.setText(str(self.ui.sharpen_slider.value()))
        gamma_value = self.ui.gamma_slider.value() / 100.0
        self.ui.gamma_label.setText(f"{gamma_value:.1f}")
        self.ui.smooth_label.setText(str(self.ui.smooth_slider.value()))
        # 刷新预览
        self.refreshImg()

    def _on_denoise_method_changed(self, index):
        """降噪方法改变时"""
        self._denoise_method = index
        self.refreshImg()

    def _on_equalize_changed(self, state):
        """直方图均衡化状态改变时"""
        self._equalize_hist = (state == Qt.Checked)
        self.refreshImg()

    def _on_clahe_changed(self, state):
        """CLAHE状态改变时"""
        self._clahe = (state == Qt.Checked)
        self.refreshImg()

    def _on_denoise_changed(self):
        """降噪强度改变时"""
        self.ui.denoise_label.setText(str(self.ui.denoiseSlider.value()))
        self.refreshImg()

    def _on_enhance_changed(self):
        """增强参数变化"""
        # 更新标签显示
        self.ui.detail_label.setText(str(self.ui.detail_slider.value()))
        self.ui.edge_label.setText(str(self.ui.edge_slider.value()))
        self.ui.local_contrast_label.setText(str(self.ui.local_contrast_slider.value()))
        # 刷新预览
        self.refreshImg()

    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            if self._preview_window_created:
                cv2.destroyWindow('preview')
                cv2.waitKey(1)
                self._preview_window_created = False
        except cv2.error:
            pass  # 忽略窗口已经关闭的错误
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

# 将.ui文件转化成/.py文件
# python -m PyQt5.uic.pyuic form.ui -o ui.py
