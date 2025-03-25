from PIL import Image
import pytesseract
import re

# 注意此处：配置 Tesseract 的实际安装路径！
pytesseract.pytesseract.tesseract_cmd = r"C:\Software\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    img = Image.open(image_path)
    gray = img.convert('L')  # 灰度化
    return gray

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def detect_sensitive_words(text, sensitive_words=None):
    if sensitive_words is None:
        sensitive_words = ["password", "secret", "credentials", "passwd", "pwd"]
    found_words = []
    for word in sensitive_words:
        if re.search(rf'\b{word}\b', text, re.IGNORECASE):
            found_words.append(word)
    return found_words

def analyze_image(image_path):
    processed_image = preprocess_image(image_path)
    text = extract_text_from_image(processed_image)
    sensitive_words = detect_sensitive_words(text)

    if sensitive_words:
        print(f"⚠️ 图片【{image_path}】中发现敏感信息: {sensitive_words}")
    else:
        print(f"✅ 图片【{image_path}】中未发现敏感信息。")

# 你的测试图片路径
if __name__ == "__main__":
    test_image = r"C:\PythonProject\fe3679b199aba747145586210bbdead.png"

    analyze_image(test_image)

