import cv2
import easyocr
import re
import numpy as np

def preprocess_image(image_path):
    """
    使用OpenCV对图片进行预处理：
      1. 读取原图并灰度化
      2. 去噪（中值滤波）
      3. 二值化（大津/自适应阈值）
      4. 可选形态学操作
      5. 转回BGR通道，便于EasyOCR读取
    """
    # 读取原图
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去噪（中值滤波）
    denoised = cv2.medianBlur(gray, 3)

    # 二值化（OTSU）
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # (可选) 形态学操作，示例: 闭运算让文字更连贯
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # EasyOCR要求3通道图像，这里再把单通道阈值图转BGR
    processed_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return processed_bgr

def detect_sensitive_words(text, sensitive_words=None):
    """
    在文本中搜索敏感关键词，返回匹配到的列表
    """
    if sensitive_words is None:
        # 你可以根据需要添加更多可疑关键词
        sensitive_words = ["password", "secret", "credentials", "passwd", "pwd"]

    found_matches = []
    # 构造正则：例如 \b(password|secret|credentials)\b
    pattern = r"\b({})\b".format("|".join(sensitive_words))
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if matches:
        # matches 里包含所有命中的关键词（可能重复）
        found_matches = list(set(m.lower() for m in matches))

    return found_matches

def analyze_image_for_sensitive_data(image_path):
    """
    1. 预处理图片
    2. 用 EasyOCR 识别 (中英文)
    3. 检测是否包含敏感关键词
    """
    # 预处理图片
    processed_img = preprocess_image(image_path)

    # 初始化EasyOCR
    # 这里用 'ch_sim','en' 同时支持中文和英文。如果只需要中文，可改成 ['ch_sim']
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

    # 执行OCR
    results = reader.readtext(processed_img)
    if not results:
        print(f"图片 {image_path} 没有识别到任何文字。")
        return

    # 将识别到的文字拼成一段文本，也可逐条分析
    recognized_text = "\n".join([res[1] for res in results])

    # 打印OCR结果
    print(f"=== 图片: {image_path} ===")
    print(f"【OCR识别结果】:\n{recognized_text}")

    # 敏感信息检测
    found = detect_sensitive_words(recognized_text)
    if found:
        print(f"⚠️ 发现敏感关键词: {found}")
    else:
        print("✅ 未发现敏感关键词。")

if __name__ == "__main__":
    # 你的图片路径 (确保在此处填写本地路径)
    image_path = r"C:\PythonProject\fe3679b199aba747145586210bbdead.png"
    analyze_image_for_sensitive_data(image_path)
