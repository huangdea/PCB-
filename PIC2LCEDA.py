import cv2
import numpy as np
import datetime
import os
import chardet

# 显示图片


def showImg(img, k):
    """显示预览图片，支持缩放"""
    img = cv2.resize(img, (0, 0), fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)  # 设置窗口为可调整大小
    cv2.imshow('preview', img)

# 拆分文件路径


def getpath(sourcefullpath):
    if os.path.isfile(sourcefullpath):
        (sourcepath, sourcename) = os.path.split(sourcefullpath)
        #print((sourcepath, sourcename))
        return sourcepath, sourcename
    else:
        return -1  # 图片文件不存在！！！

# 图片转换


def _apply_threshold(img, threshold_method=1, threshold_value=150):
    """阈值处理"""
    if threshold_method == 0:  # 固定阈值
        return cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]
    elif threshold_method == 1:  # 自适应阈值
        # 计算合适的块大小
        block_size = min(img.shape) // 8
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, 51))
        
        # 优化自适应阈值参数
        C = max(0, threshold_value / 10 - 10)
        return cv2.adaptiveThreshold(img, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 
                                   block_size, C)
    elif threshold_method == 2:  # Otsu阈值
        return cv2.threshold(img, 0, 255, 
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif threshold_method == 3:  # Sauvola阈值
        # 计算窗口大小
        window = min(img.shape) // 8
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, 51))
        
        # 计算局部均值和标准差
        mean = cv2.boxFilter(img.astype(float), -1, (window, window))
        mean_square = cv2.boxFilter(img.astype(float)**2, -1, (window, window))
        std = np.sqrt(mean_square - mean**2)
        
        # Sauvola参数
        k = 0.2
        R = 128
        threshold = mean * (1 + k * ((std / R) - 1))
        
        return np.where(img >= threshold, 255, 0).astype(np.uint8)
    elif threshold_method == 4:  # Wolf阈值
        # 计算窗口大小
        window = min(img.shape) // 8
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, 51))
        
        # 计算局部均值和标准差
        mean = cv2.boxFilter(img.astype(float), -1, (window, window))
        mean_square = cv2.boxFilter(img.astype(float)**2, -1, (window, window))
        std = np.sqrt(mean_square - mean**2)
        
        # Wolf参数
        k = 0.5
        R = 128
        min_std = 2
        threshold = mean - k * std * (1 - std/(R * np.clip(std, min_std, None)))
        
        return np.where(img >= threshold, 255, 0).astype(np.uint8)
    elif threshold_method == 5:  # Nick阈值
        # 计算窗口大小
        window = min(img.shape) // 8
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, 51))
        
        # 计算局部均值和标准差
        mean = cv2.boxFilter(img.astype(float), -1, (window, window))
        mean_square = cv2.boxFilter(img.astype(float)**2, -1, (window, window))
        std = np.sqrt(mean_square - mean**2)
        
        # Nick参数
        k = -0.1
        threshold = mean + k * std
        
        return np.where(img >= threshold, 255, 0).astype(np.uint8)
    elif threshold_method == 6:  # Bernsen阈值
        # 计算窗口大小
        window = min(img.shape) // 8
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, 51))
        
        # 计算局部最大值和最小值
        kernel = np.ones((window, window), np.uint8)
        local_max = cv2.dilate(img, kernel)
        local_min = cv2.erode(img, kernel)
        
        # Bernsen参数
        contrast_threshold = 15
        local_contrast = local_max - local_min
        local_mean = (local_max + local_min) / 2
        
        # 对比度太低的区域使用全局阈值
        global_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[0]
        mask = local_contrast < contrast_threshold
        threshold = np.where(mask, global_threshold, local_mean)
        
        return np.where(img >= threshold, 255, 0).astype(np.uint8)
    else:  # 默认使用固定阈值
        return cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]

class ImageEnhancer:
    """图像增强处理类"""
    
    @staticmethod
    def apply_histogram_enhancement(img, equalize=False, clahe=False):
        """直方图增强"""
        if equalize:
            img = cv2.equalizeHist(img.astype(np.uint8)).astype(np.float32)
        if clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img.astype(np.uint8)).astype(np.float32)
        return img
    
    @staticmethod
    def apply_local_enhancement(img, local_contrast=0, detail_enhance=0, edge_enhance=0):
        """局部增强"""
        # 局部对比度增强
        if local_contrast > 0:
            sigma = local_contrast * 0.5
            gaussian = cv2.GaussianBlur(img, (0, 0), sigma)
            img = cv2.addWeighted(img, 1 + local_contrast/50, gaussian, -local_contrast/50, 0)
        
        # 细节增强
        if detail_enhance > 0:
            blur = cv2.GaussianBlur(img, (0, 0), 3)
            detail = cv2.addWeighted(img, 1.0 + detail_enhance/50, blur, -detail_enhance/50, 0)
            img = cv2.addWeighted(img, 0.7, detail, 0.3, 0)
        
        # 边缘增强
        if edge_enhance > 0:
            sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            gradient = cv2.magnitude(sobelx, sobely)
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.addWeighted(img, 1.0, gradient, edge_enhance/100.0, 0)
            
        return img

    @staticmethod
    def apply_denoise(img, method=0, strength=0):
        """降噪"""
        if strength <= 0:
            return img
            
        if method == 0:  # 高斯降噪
            kernel_size = int(strength / 10) * 2 + 3
            sigma = strength / 20.0
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        elif method == 1:  # 中值滤波
            kernel_size = int(strength / 10) * 2 + 3
            return cv2.medianBlur(img.astype(np.uint8), kernel_size).astype(np.float32)
        elif method == 2:  # 双边滤波
            d = int(strength)
            return cv2.bilateralFilter(img.astype(np.uint8), d, 75, 75).astype(np.float32)
        elif method == 3:  # NLMeans降噪
            h = strength * 2
            return cv2.fastNlMeansDenoising(img.astype(np.uint8), 
                                          h=h,
                                          templateWindowSize=7,
                                          searchWindowSize=21).astype(np.float32)
        return img

    @staticmethod
    def apply_basic_adjustments(img, exposure=0, contrast=0, gamma=1.0):
        """基础调整"""
        # 伽马校正
        if gamma != 1.0:
            img = np.power(img / 255.0, gamma) * 255.0
        
        # 曝光度调整
        if exposure != 0:
            factor = 1.0 + (exposure / 50.0)
            img = cv2.multiply(img, factor)
        
        # 对比度调整
        if contrast != 0:
            factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
            img = factor * (img - 128.0) + 128.0
            
        return img

    @staticmethod
    def apply_sharpening(img, sharpen=0):
        """锐化"""
        if sharpen <= 0:
            return img
            
        # 根据锐化强度选择不同的核
        if sharpen < 33:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * (sharpen / 100.0)
        elif sharpen < 66:
            kernel = np.array([[-2,-2,-2],
                             [-2,17,-2],
                             [-2,-2,-2]]) * (sharpen / 100.0)
        else:
            kernel = np.array([[-3,-3,-3],
                             [-3,25,-3],
                             [-3,-3,-3]]) * (sharpen / 100.0)
                             
        return cv2.filter2D(img, -1, kernel)

def _apply_preprocess(img, **kwargs):
    """图像预处理"""
    enhancer = ImageEnhancer()
    
    img = img.astype(np.float32)
    
    # 直方图增强
    img = enhancer.apply_histogram_enhancement(
        img, 
        equalize=kwargs.get('equalize', False),
        clahe=kwargs.get('clahe', False)
    )
    
    # 局部增强
    img = enhancer.apply_local_enhancement(
        img,
        local_contrast=kwargs.get('local_contrast', 0),
        detail_enhance=kwargs.get('detail_enhance', 0),
        edge_enhance=kwargs.get('edge_enhance', 0)
    )
    
    # 降噪
    img = enhancer.apply_denoise(
        img,
        method=kwargs.get('denoise_method', 0),
        strength=kwargs.get('denoise_strength', 0)
    )
    
    # 基础调整
    img = enhancer.apply_basic_adjustments(
        img,
        exposure=kwargs.get('exposure', 0),
        contrast=kwargs.get('contrast', 0),
        gamma=kwargs.get('gamma', 1.0)
    )
    
    # 锐化
    img = enhancer.apply_sharpening(
        img,
        sharpen=kwargs.get('sharpen', 0)
    )
    
    # 裁剪到有效范围
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

class ImageProcessor:
    """图像处理类"""
    
    @staticmethod
    def read_image(sourcefullpath):
        """读取图像"""
        try:
            img = cv2.imdecode(np.fromfile(sourcefullpath, dtype=np.uint8), -1)
            if img is None:
                return None
            return img
        except:
            return None
    
    @staticmethod
    def preprocess_image(img, border=1, to_gray=True):
        """图像预处理"""
        # 拓展边缘
        img = cv2.copyMakeBorder(img, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 转换为灰度图
        if to_gray and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        return img
    
    @staticmethod
    def resize_image(img, target_width, target_height, width):
        """调整图像大小"""
        if img is None:
            return None
            
        im_height, im_width = img.shape[:2]
        
        # 计算缩放比例
        if im_height / target_height > im_width / target_width:
            scale = target_height / im_height
            new_width = int(im_width * scale)
            new_height = target_height
        else:
            scale = target_width / im_width
            new_width = target_width
            new_height = int(im_height * scale)
            
        # 缩放图像
        return cv2.resize(img, 
                         (int(new_width / width), int(new_height / width)),
                         interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def apply_filters(img, denoise=0, threshold=150, threshold_method=1):
        """图像滤波"""
        if denoise > 0:
            # 降噪处理
            sigma = denoise / 60.0 * 2
            d = int(denoise / 60.0 * 12) * 2 + 3
            
            if denoise > 20:
                img = cv2.GaussianBlur(img, (3, 3), sigma)
            img = cv2.bilateralFilter(img, d, sigmaColor=50, sigmaSpace=0.5 * d)
        
        # 二值化处理
        return _apply_threshold(img, threshold_method, threshold)
    
    @staticmethod
    def apply_transforms(img, layer, color_invert_f, x_invert_f, y_invert_f):
        """图像变换"""
        # 层级翻转
        if layer in [2, 4, 6, 8]:
            img = cv2.flip(img, 1)
            
        # 用户指定的翻转
        if x_invert_f:
            img = cv2.flip(img, 1)
        if y_invert_f:
            img = cv2.flip(img, 0)
        if color_invert_f:
            img = cv2.bitwise_not(img)
            
        return img

def transformpic(sourcefullpath, x_size, y_size, width, layer, color_invert_f, 
                x_invert_f, y_invert_f, threshold, denoise=0, use_original_size=False, 
                threshold_method=1, exposure=0, contrast=0, sharpen=0, gamma=1.0, smooth=0,
                denoise_method=0, denoise_strength=0, equalize=False, clahe=False,
                detail_enhance=0, edge_enhance=0, local_contrast=0):
    """转换图片"""
    # 检查文件是否存在
    if getpath(sourcefullpath) == -1:
        return -1

    # 单位转换
    x_size_mil = int(x_size / 2.54 * 100)
    y_size_mil = int(y_size / 2.54 * 100)

    # 读取并处理图像
    processor = ImageProcessor()
    img = processor.read_image(sourcefullpath)
    if img is None:
        return -1
        
    # 基础预处理
    img = processor.preprocess_image(img)
    
    # 获取原始尺寸
    im_raw, im_col = img.shape[:2]
    
    # 调整图像大小
    if not use_original_size:
        img = processor.resize_image(img, x_size_mil, y_size_mil, width)
        if img is None:
            return -1
    
    # 图像预处理
    preprocess_params = {
        'exposure': exposure,
        'contrast': contrast,
        'sharpen': sharpen,
        'gamma': gamma,
        'smooth': smooth,
        'equalize': equalize,
        'clahe': clahe,
        'denoise_method': denoise_method,
        'denoise_strength': denoise_strength,
        'detail_enhance': detail_enhance,
        'edge_enhance': edge_enhance,
        'local_contrast': local_contrast
    }
    
    img = _apply_preprocess(img, **preprocess_params)
    
    # 滤波
    img = processor.apply_filters(img, denoise, threshold, threshold_method)
    
    # 变换
    img = processor.apply_transforms(img, layer, color_invert_f, x_invert_f, y_invert_f)
    
    return img, x_size_mil, y_size_mil, im_raw, im_col

# 转换生成库文件，正常转换返回0，图片不存在返回-1


def makepcb(sourcefullpath, x_size, y_size, width, layer, color_invert_f, 
            x_invert_f, y_invert_f, threshold, denoise=0, threshold_method=1, 
            exposure=0, contrast=0, sharpen=0, gamma=1.0, smooth=0,
            denoise_method=0, denoise_strength=0, equalize=False, clahe=False,
            detail_enhance=0, edge_enhance=0, local_contrast=0,
            output_path="", copper_f=0):
    """生成PCB文件"""
    # 由文件绝对路径分离出路径和文件名
    if getpath(sourcefullpath) == -1:
        return -1
    (sourcepath, sourcename) = getpath(sourcefullpath)
    
    # 使用指定的输出路径或默认路径
    if output_path:
        path = os.path.join(
            output_path,
            'LCEDA_' + sourcename.split('.')[0] + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")))
    else:
        path = os.path.join(
            sourcepath,
            'LCEDA_' + sourcename.split('.')[0] + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")))
    
    # 创建输出目录
    if not os.path.exists(path):
        os.makedirs(path)
        
    lib_filename = 'LIB_' + sourcename.split('.')[0] + '.json'  # Lib文件名
    pcb_filename = 'PCB_' + sourcename.split('.')[0] + '.json'  # PCB文件名
    
    # 备份原图
    img2 = cv2.imdecode(np.fromfile(sourcefullpath, dtype=np.uint8), -1)
    cv2.imencode('.jpg', img2)[1].tofile(os.path.join(path, 'pre.jpg'))
    
    # 转换图片，传递所有预处理参数
    result = transformpic(sourcefullpath, x_size, y_size, width, layer, 
                         color_invert_f, x_invert_f, y_invert_f, threshold,
                         denoise=denoise, threshold_method=threshold_method,
                         exposure=exposure, contrast=contrast, sharpen=sharpen,
                         gamma=gamma, smooth=smooth)
    
    if result == -1:
        return -1
        
    img, x_size_mil, y_size_mil, im_raw, im_col = result
    
    # 备份处理后的图片
    cv2.imencode('.jpg', img)[1].tofile(os.path.join(path, 'after.jpg'))
    
    # 生成PCB数据
    pcb_data = generate_pcb_data(img, x_size_mil, y_size_mil, width, layer, copper_f)
    
    # 保存文件
    try:
        with open(os.path.join(path, lib_filename), 'w', encoding='utf-8') as f:
            f.write(pcb_data)
        
        # 生成PCB边框数据
        pcb_border_data = generate_pcb_border(x_size_mil, y_size_mil)
        with open(os.path.join(path, pcb_filename), 'w', encoding='utf-8') as f:
            f.write(pcb_border_data)
            
        # 生成信息文件
        generate_info_file(path, sourcepath, sourcename, im_raw, im_col,
                         x_size, y_size, x_size_mil, y_size_mil, width, layer,
                         color_invert_f, x_invert_f, y_invert_f, copper_f,
                         threshold, lib_filename, pcb_filename)
        
        return 0
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return -1

def generate_pcb_data(img, x_size_mil, y_size_mil, width, layer, copper_f=0):
    """生成PCB数据"""
    # 获取图像尺寸
    height, width_px = img.shape[:2]
    
    # 计算单位像素代表的实际尺寸
    x_scale = x_size_mil / width_px
    y_scale = y_size_mil / height
    
    # 初始化边框数据
    lines_mil = np.array([
        [0, 0, x_size_mil / 10, 0],  # 0
        [x_size_mil / 10, 0, x_size_mil / 10, y_size_mil / 10],  # 1
        [x_size_mil / 10, y_size_mil / 10, 0, y_size_mil / 10],  # 2
        [0, y_size_mil / 10, 0, 0]  # 3
    ])
    
    # 扫描图像生成线段数据
    lastPix = 255
    line_start = 0
    for i in range(height):
        lastPix = img[i, 0]
        for j in range(width_px):
            if lastPix != img[i, j]:
                if lastPix == 255:
                    line_start = j
                else:
                    # 添加线段
                    lines_mil = np.vstack((
                        lines_mil,
                        np.array([
                            line_start / 10 * width, i * width / 10,
                            j / 10 * width, i * width / 10
                        ])
                    ))
                lastPix = img[i, j]
    
    # 生成文件头
    pcb_data = [
        '{\n    "head": {\n      "docType": "4",\n      "editorVersion": "6.4.2",',
        '\n      "newgId": true,\n      "c_para": {',
        '\n        "package": "PCB_Image",',
        '\n        "pre": "PIC?",',
        '\n        "Contributor": "LCNB",',
        '\n        "link": ""',
        '\n      },',
        '\n      "hasIdFlag": true,',
        f'\n      "x": {(lines_mil[0, 0] + lines_mil[2, 0]) / 2},',
        f'\n      "y": {(lines_mil[0, 1] + lines_mil[2, 1]) / 2}',
        '\n    },',
        '\n    "canvas": "CA~1000~1000~#000000~yes~#FFFFFF~10~1000~1000~line~10~mil',
        f'~1~45~~0.5~{(lines_mil[0, 0] + lines_mil[2, 0]) / 2}~{(lines_mil[0, 1] + lines_mil[2, 1]) / 2}~0~none",',
        '\n    "shape": ['
    ]
    
    # 添加铜皮数据
    if copper_f == 2:
        pcb_data.extend([
            f'\n    "SOLIDREGION~1~~M {lines_mil[0, 0]:.4f} {lines_mil[0, 1]:.4f} '
            f'L {lines_mil[0, 2]:.4f} {lines_mil[0, 3]:.4f} '
            f'L {lines_mil[2, 0]:.4f} {lines_mil[2, 1]:.4f} '
            f'L {lines_mil[2, 2]:.4f},{lines_mil[2, 3]:.4f} Z~solid~ggb0~~~~0",',
            f'\n    "SOLIDREGION~2~~M {lines_mil[0, 0]:.4f} {lines_mil[0, 1]:.4f} '
            f'L {lines_mil[0, 2]:.4f} {lines_mil[0, 3]:.4f} '
            f'L {lines_mil[2, 0]:.4f} {lines_mil[2, 1]:.4f} '
            f'L {lines_mil[2, 2]:.4f},{lines_mil[2, 3]:.4f} Z~solid~ggb1~~~~0",'
        ])
    
    # 添加边框数据
    for i in range(4):
        pcb_data.append(
            f'\n    "TRACK~1~{layer}~~{lines_mil[i, 0]:.4f} {lines_mil[i, 1]:.4f} '
            f'{lines_mil[i, 2]:.4f} {lines_mil[i, 3]:.4f}~ggc{i}~0",'
        )
    
    # 添加图像数据
    for i in range(4, lines_mil.shape[0]):
        pcb_data.append(
            f'\n    "TRACK~{width/10:.1f}~{layer}~~{lines_mil[i, 0]:.4f} {lines_mil[i, 1]:.4f} '
            f'{lines_mil[i, 2]:.4f} {lines_mil[i, 3]:.4f}~gge{i}~0"'
        )
        if i < lines_mil.shape[0] - 1:
            pcb_data.append(',')
    
    # 添加文件尾
    pcb_data.extend([
        '\n    ],',
        '\n    "layers": [',
        '\n      "1~TopLayer~#FF0000~true~true~true~",',
        '\n      "2~BottomLayer~#0000FF~true~false~true~",',
        '\n      "3~TopSilkLayer~#FFCC00~true~false~true~",',
        '\n      "4~BottomSilkLayer~#66CC33~true~false~true~",',
        '\n      "5~TopPasteMaskLayer~#808080~true~false~true~",',
        '\n      "6~BottomPasteMaskLayer~#800000~true~false~true~",',
        '\n      "7~TopSolderMaskLayer~#800080~true~false~true~0.3",',
        '\n      "8~BottomSolderMaskLayer~#AA00FF~true~false~true~0.3",',
        '\n      "9~Ratlines~#6464FF~false~false~true~",',
        '\n      "10~BoardOutLine~#FF00FF~true~false~true~",',
        '\n      "11~Multi-Layer~#C0C0C0~true~false~true~",',
        '\n      "12~Document~#FFFFFF~true~false~true~",',
        '\n      "13~TopAssembly~#33CC99~false~false~false~",',
        '\n      "14~BottomAssembly~#5555FF~false~false~false~",',
        '\n      "15~Mechanical~#F022F0~false~false~false~",',
        '\n      "19~3DModel~#66CCFF~false~false~false~",',
        '\n      "21~Inner1~#999966~false~false~false~~",',
        '\n      "22~Inner2~#008000~false~false~false~~",',
        '\n      "23~Inner3~#00FF00~false~false~false~~",',
        '\n      "24~Inner4~#BC8E00~false~false~false~~",',
        '\n      "25~Inner5~#70DBFA~false~false~false~~",',
        '\n      "26~Inner6~#00CC66~false~false~false~~",',
        '\n      "27~Inner7~#9966FF~false~false~false~~",',
        '\n      "28~Inner8~#800080~false~false~false~~",',
        '\n      "29~Inner9~#008080~false~false~false~~",',
        '\n      "30~Inner10~#15935F~false~false~false~~",',
        '\n      "31~Inner11~#000080~false~false~false~~",',
        '\n      "32~Inner12~#00B400~false~false~false~~",',
        '\n      "33~Inner13~#2E4756~false~false~false~~",',
        '\n      "34~Inner14~#99842F~false~false~false~~",',
        '\n      "35~Inner15~#FFFFAA~false~false~false~~",',
        '\n      "36~Inner16~#99842F~false~false~false~~",',
        '\n      "37~Inner17~#2E4756~false~false~false~~",',
        '\n      "38~Inner18~#3535FF~false~false~false~~",',
        '\n      "39~Inner19~#8000BC~false~false~false~~",',
        '\n      "40~Inner20~#43AE5F~false~false~false~~",',
        '\n      "41~Inner21~#C3ECCE~false~false~false~~",',
        '\n      "42~Inner22~#728978~false~false~false~~",',
        '\n      "43~Inner23~#39503F~false~false~false~~",',
        '\n      "44~Inner24~#0C715D~false~false~false~~",',
        '\n      "45~Inner25~#5A8A80~false~false~false~~",',
        '\n      "46~Inner26~#2B937E~false~false~false~~",',
        '\n      "47~Inner27~#23999D~false~false~false~~",',
        '\n      "48~Inner28~#45B4E3~false~false~false~~",',
        '\n      "49~Inner29~#215DA1~false~false~false~~",',
        '\n      "50~Inner30~#4564D7~false~false~false~~",',
        '\n      "51~Inner31~#6969E9~false~false~false~~",',
        '\n      "52~Inner32~#9069E9~false~false~false~~",',
        '\n      "99~ComponentShapeLayer~#00CCCC~false~false~false~",',
        '\n      "100~LeadShapeLayer~#CC9999~false~false~false~",',
        '\n      "Hole~Hole~#222222~~false~true~",',
        '\n      "DRCError~DRCError~#FAD609~~false~true~"',
        '\n    ],',
        '\n    "objects": [',
        '\n      "All~true~false",',
        '\n      "Component~true~true",',
        '\n      "Prefix~true~true",',
        '\n      "Name~true~false",',
        '\n      "Track~true~true",',
        '\n      "Pad~true~true",',
        '\n      "Via~true~true",',
        '\n      "Hole~true~true",',
        '\n      "Copper_Area~true~true",',
        '\n      "Circle~true~true",',
        '\n      "Arc~true~true",',
        '\n      "Solid_Region~true~true",',
        '\n      "Text~true~true",',
        '\n      "Image~true~true",',
        '\n      "Rect~true~true",',
        '\n      "Dimension~true~true",',
        '\n      "Protractor~true~true"',
        '\n    ],',
        f'\n    "BBox": {{\n      "x": {(lines_mil[0, 0] + lines_mil[2, 0]) / 2},',
        f'\n      "y": {(lines_mil[0, 1] + lines_mil[2, 1]) / 2},',
        f'\n      "width": {x_size_mil},',
        f'\n      "height": {y_size_mil}\n    }},',
        '\n    "netColors": {}\n}'
    ])
    
    return ''.join(pcb_data)

def generate_pcb_border(x_size_mil, y_size_mil):
    """生成PCB边框数据"""
    # 边框的顶点坐标
    vertices = [
        (0, 0),                    # 左上角
        (x_size_mil, 0),          # 右上角
        (x_size_mil, y_size_mil), # 右下角
        (0, y_size_mil),          # 左下角
        (0, 0)                    # 回到起点
    ]
    
    # 生成PCB文件头
    pcb_data = [
        '{\n    "head": {\n      "docType": "3",\n      "editorVersion": "6.4.2",',
        '\n      "newgId": true,\n      "c_para": {},\n      "hasIdFlag": true\n    },',
        '\n    "canvas": "CA~1000~1000~#000000~yes~#FFFFFF~39.370079~1000~1000~line~',
        f'3.937008~mm~1~45~~0.5~{x_size_mil/2}~{y_size_mil/2}~0~yes",\n    "shape": [\n'
    ]
    
    # 添加边框线段
    for i in range(len(vertices)-1):
        x1, y1 = vertices[i]
        x2, y2 = vertices[i+1]
        pcb_data.append(
            f'    "TRACK~1~10~S$998~{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}~ggc{i}~0"'
        )
        if i < len(vertices)-2:
            pcb_data.append(',')
        pcb_data.append('\n')
    
    # 添加文件尾
    pcb_data.extend([
        '    ],\n',
        '    "layers": [\n',
        '      "1~TopLayer~#FF0000~true~false~true~",\n',
        '      "2~BottomLayer~#0000FF~true~false~true~",\n',
        '      "3~TopSilkLayer~#FFCC00~true~false~true~",\n',
        '      "4~BottomSilkLayer~#66CC33~true~true~true~",\n',
        '      "5~TopPasteMaskLayer~#808080~true~false~true~",\n',
        '      "6~BottomPasteMaskLayer~#800000~true~false~true~",\n',
        '      "7~TopSolderMaskLayer~#800080~true~false~true~0.3",\n',
        '      "8~BottomSolderMaskLayer~#AA00FF~true~false~true~0.3",\n',
        '      "9~Ratlines~#6464FF~false~false~true~",\n',
        '      "10~BoardOutLine~#FF00FF~true~false~true~",\n',
        '      "11~Multi-Layer~#C0C0C0~true~false~true~",\n',
        '      "12~Document~#FFFFFF~true~false~true~"\n',
        '    ],\n',
        '    "objects": [\n',
        '      "All~true~false",\n',
        '      "Component~true~true",\n',
        '      "Prefix~true~true",\n',
        '      "Name~true~false",\n',
        '      "Track~true~true",\n',
        '      "Pad~true~true",\n',
        '      "Via~true~true",\n',
        '      "Hole~true~true",\n',
        '      "Copper_Area~true~true",\n',
        '      "Circle~true~true",\n',
        '      "Arc~true~true",\n',
        '      "Solid_Region~true~true",\n',
        '      "Text~true~true",\n',
        '      "Image~true~true",\n',
        '      "Rect~true~true",\n',
        '      "Dimension~true~true",\n',
        '      "Protractor~true~true"\n',
        '    ],\n',
        f'    "BBox": {{\n      "x": {x_size_mil/2},\n      "y": {y_size_mil/2},',
        f'\n      "width": {x_size_mil},\n      "height": {y_size_mil}\n    }},\n',
        '    "netColors": {}\n',
        '}'
    ])
    
    return ''.join(pcb_data)

def generate_info_file(path, sourcepath, sourcename, im_raw, im_col,
                      x_size, y_size, x_size_mil, y_size_mil, width, layer,
                      color_invert_f, x_invert_f, y_invert_f, copper_f,
                      threshold, lib_filename, pcb_filename):
    """生成信息文件"""
    layerstr = ['NULL', '顶层', '底层', '顶层丝印层', '底层丝印层',
                '顶层焊盘层', '底层焊盘层', '顶层阻焊层', '底层阻焊层', '边框层', '文档层']
    
    info_lines = [
        '\n| 参数\t\t| 值 \n',
        '------------------------------------------------------\n',
        f'| 原图X像素\t| {im_col:.4f} pix\n',
        f'| 原图y像素\t| {im_raw:.4f} pix\n',
        f'| 线宽\t\t| {width} mil\n',
        f'| X最大尺寸\t| {x_size:.4f} mm\n',
        f'| y最大尺寸\t| {y_size:.4f} mm\n',
        f'| X实际像素\t| {x_size_mil:.4f} pix\n',
        f'| y实际像素\t| {y_size_mil:.4f} pix\n',
        f'| X实际尺寸\t| {x_size_mil/100*2.54:.4f} mm\n',
        f'| y实际尺寸\t| {y_size_mil/100*2.54:.4f} mm\n',
        f'| 所在层\t\t| {layerstr[layer]}\n',
        f'| 源文件路径\t| {sourcepath}\n',
        f'| 源文件名称\t| {sourcename}\n',
        f'| 生成文件路径\t| {path}\n',
        f'| Lib文件名称\t| {lib_filename}\n',
        f'| PCB文件名称\t| {pcb_filename}\n',
        f'| 图像取反\t| {"true" if color_invert_f != 0 else "false"}\n',
        f'| 水平翻转\t| {"true" if x_invert_f != 0 else "false"}\n',
        f'| 垂直翻转\t| {"true" if y_invert_f != 0 else "false"}\n',
        f'| 创建铜皮\t| {"true" if copper_f != 0 else "false"}\n',
        f'| 阈值\t\t| {threshold}\n'
    ]
    
    with open(os.path.join(path, 'info.txt'), 'w', encoding='utf-8') as f:
        f.writelines(info_lines)
