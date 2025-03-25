import cv2
import re
import numpy as np
from paddleocr import PaddleOCR

def preprocess_image(image_path):
    """
    使用OpenCV对图片进行预处理：
      1. 读取并灰度化
      2. 去噪（中值滤波）
      3. 二值化（OTSU）
      4. (可选) 形态学操作
      5. 转回BGR三通道
    """
    # 1) 读取原图 (BGR格式)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 2) 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) 去噪（中值滤波）
    denoised = cv2.medianBlur(gray, 3)

    # 4) 二值化（OTSU）
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 如果背景不均匀，可试 cv2.adaptiveThreshold()

    # 5) 可选形态学操作 (示例：闭运算去除字间空隙)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6) 转回 BGR 通道 (PaddleOCR 可直接传numpy图)
    processed_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return processed_bgr

def detect_sensitive_words(text, sensitive_words=None):
    """
    在文本中搜索敏感关键词，返回匹配到的列表
    """
    if sensitive_words is None:
        # 可自行扩充关键字
        sensitive_words = ["password", "secret", "credentials", "passwd", "pwd"]

    found_matches = []
    # 构造正则:  \b(password|secret|credentials)\b
    pattern = r"\b({})\b".format("|".join(sensitive_words))
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if matches:
        # 去重并转小写
        found_matches = list(set(m.lower() for m in matches))
    return found_matches

def analyze_image_for_sensitive_data(image_path):
    """
    1. 预处理图片
    2. PaddleOCR 识别中英文
    3. 敏感关键词检测
    """
    # 1) 预处理图片
    processed_bgr = preprocess_image(image_path)

    # 2) 初始化 PaddleOCR
    # use_angle_cls=True: 启用方向分类
    # lang='ch'：中文；同时识别英文默认也支持。如果想更精确，也可 lang='chinese_cht' 等
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')

    # 注意 PaddleOCR 可直接接受 numpy 数组，但需要 RGB 格式
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    # 3) 调用 OCR
    # 返回格式: [ [ [左上,右上,右下,左下], ("文字", 置信度) ], ... ]
    result = ocr.ocr(processed_rgb, cls=True)

    # 合并识别结果成纯文本
    recognized_text_list = []
    for line in result:
        for boxinfo in line:
            text, confidence = boxinfo[1][0], boxinfo[1][1]
            recognized_text_list.append(text)

    if not recognized_text_list:
        print(f"图片 {image_path} 中未识别到任何文字。")
        return

    recognized_text = "\n".join(recognized_text_list)

    # 4) 打印OCR结果
    print(f"=== 图片: {image_path} ===")
    print("【识别结果】:\n", recognized_text)

    # 5) 检测敏感信息
    found = detect_sensitive_words(recognized_text)
    if found:
        print(f"⚠️ 发现敏感关键词: {found}")
    else:
        print("✅ 未发现敏感关键词。")

if __name__ == "__main__":
    # 测试图片路径
    image_path = r"C:\PythonProject\fe3679b199aba747145586210bbdead.png"
    analyze_image_for_sensitive_data(image_path)
