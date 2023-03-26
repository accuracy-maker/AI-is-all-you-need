import pytesseract
import os

# 设置 Tesseract 环境变量
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'

# 指定要识别的图像文件路径
image_path = '/Users/gaohaitao/Desktop/sys_cv_test/dataset/images/chi_sim_image/Bonnie LIU_Firmenich/page_1.jpg'

# 识别图像并返回文本结果
text = pytesseract.image_to_string(image_path, lang='chi_sim')
print(text)