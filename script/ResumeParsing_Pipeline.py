import os
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import time
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata/'


# 设置输入文件夹路径
pdf_dir = '../dataset/cv_files/'

# 设置输出文件夹路径
image_dir = '../dataset/images'
text_dir = '../dataset/texts'
chatgpt_content_dir = '../dataset/chatGPT_content'

# 创建输出文件夹
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

folder_list = [('eng_pdf','eng_image','eng_text'),('chi_sim_pdf','chi_sim_image','chi_sim_text'),
               ('kor_pdf','kor_image','kor_text')]


def Pdf2Images(input_dir,output_dir):
    for sub_folder in folder_list:
        print("===>processing {} subfolder".format(sub_folder[0]))
        folder_path = input_dir + sub_folder[0]
        # 遍历输入文件夹中的所有PDF文件
        for pdf_file_name in os.listdir(folder_path):
            if not pdf_file_name.startswith('.'):
                print("==>processing the file {}".format(pdf_file_name))
                if pdf_file_name.endswith('.pdf'):
                    pdf_path = os.path.join(input_dir,sub_folder[0],pdf_file_name)
                    # 创建当前PDF文件的输出子文件夹
                    pdf_output_dir = os.path.join(output_dir,sub_folder[1],pdf_file_name[:-4])
                    if not os.path.exists(pdf_output_dir):
                        os.makedirs(pdf_output_dir)
                    print('==>the output dir is: {}'.format(pdf_output_dir))
                    # 将PDF文件转换为图像
                    pages = convert_from_path(pdf_path)
                    # 遍历所有图像页面
                    for i, page in enumerate(pages):
                        # 设置输出图像的文件名
                        output_filename = f'page_{i + 1}.jpg'
                        # 设置输出图像的完整路径
                        output_path = os.path.join(pdf_output_dir, output_filename)
                        # 保存图像文件
                        page.save(output_path, 'JPEG')
                        print("{} covert to image {} and saved.".format(pdf_file_name,output_path))

Pdf2Images(pdf_dir,image_dir)

def Images2Text(image_folder,output_folder):
    for sub_folder in folder_list:
        print("===>processing the subfolder {}".format(sub_folder[1]))
        # 循环遍历图片文件夹中的每个子文件夹
        image_path = image_folder + '/' + sub_folder[1]
        for foldername in os.listdir(image_path):
            print("===>Processing {}".format(foldername))
            if not foldername.startswith('.'):
                folderpath = os.path.join(image_folder,sub_folder[1],foldername)

                # 创建输出文件夹
                output_folderpath = os.path.join(output_folder,sub_folder[2],foldername)
                os.makedirs(output_folderpath, exist_ok=True)

            # 循环遍历子文件夹中的所有图片
            for filename in os.listdir(folderpath):
                print("===>Processing {}".format(filename))
                filepath = os.path.join(folderpath, filename)

                # 检查文件是否为图片文件
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # 读取图片并将其转换为文本
                image = Image.open(filepath)
                lan = sub_folder[2][:3]
                if lan == 'chi':
                    lan = 'chi_sim'
                text = pytesseract.image_to_string(image,lang=(lan + '+eng'))

                # 创建输出文件路径
                output_filename = os.path.splitext(filename)[0] + ".txt"
                output_filepath = os.path.join(output_folderpath, output_filename)

                # 将文本写入输出文件
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                print("{} covert to text and saved".format(filename))
Images2Text(image_dir,text_dir)


def merge_text(text_path,output_dir):
    for sub_folder in folder_list:
        print("===>merge the sub_folder {}".format(sub_folder[2]))
        folder_path = os.path.join(text_path,sub_folder[2])
        out_sub_dir = os.path.join(output_dir,sub_folder[2])
        # 遍历文件夹中的每个子文件夹
        for subdir in os.listdir(folder_path):
            print("==>processing subdir {}".format(subdir))
            # 检查子文件夹是否是文件夹，如果不是，则跳过
            if not os.path.isdir(os.path.join(folder_path, subdir)):
                continue

            # 定义要合并的TXT文件列表
            txt_files = []

            # 遍历子文件夹中的每个TXT文件
            for file in os.listdir(os.path.join(folder_path, subdir)):
                # 检查文件是否是TXT文件，如果不是，则跳过
                if not file.endswith(".txt"):
                    continue

                # 读取TXT文件的内容并添加到txt_files列表中
                with open(os.path.join(folder_path, subdir, file), "r") as f:
                    txt_files.append(f.read())

            # 将所有TXT文件内容合并到一个文件中
            saved_out_dir = os.path.join(out_sub_dir, subdir)
            os.makedirs(saved_out_dir, exist_ok=True)
            with open(os.path.join(saved_out_dir, "_merged.txt"), "w") as f:
                f.write("\n".join(txt_files))
                print("write done!")


merge_text(text_dir,chatgpt_content_dir)


#用python发送post请求
import requests
import json

api_key = "xxxx"
url = "xxxxxx"
headers = {"api-key": api_key, "Authorization": "Bearer " + api_key}

# 遍历文本文件夹
root_path = "../dataset/chatGPT_content"
for sub_folder in folder_list:
    print("===>process sub folder {}".format(sub_folder))
    folder_path = os.path.join(root_path,sub_folder[2])
    for filename in os.listdir(folder_path):
        print("===>process filename {}".format(filename))
        # 读取文本内容
        with open(os.path.join(folder_path, filename,'_merged.txt'), "r") as f:
            text = f.read().strip()
        # print(type(text))
        content = '假如你是一位资深的简历解析者，请提取这份简历的信息：' + text
        # print(content)
        # 发送POST请求
        data = {
            "prompt": content,
            "temperature": 1,
            "top_p": 0.5,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 1500,
            "best_of": 1,
            "stop": None
        }
        start_time = time.time()
        print('post requests, start time is {}'.format(start_time))
        response = requests.post(url, headers=headers, json=data)
        consume_time = time.time() - start_time
        print("recevied response,running time :{}".format(consume_time))
        # print(response.text)
        data_dict = response.json()

        output_root_dir = '../dataset/chatGPT_response_json'
        output_dir = os.path.join(output_root_dir,sub_folder[2],filename)
        os.makedirs(output_dir, exist_ok=True)
        # 保存JSON文件
        with open(output_dir+'/response.json', 'w') as f:
            json.dump(data_dict, f)

        print('JSON文件已保存')



