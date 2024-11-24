import torch
import numpy
from PIL import Image, ImageFile
import os
from collections import defaultdict
import gzip
from TPS import tps_photo
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
# 图片数量少，处理之前需要转换到黑白图片，缩放为 80x80 以免过拟合
# 编码长度 32，不同人物的距离 > 0.2
IMAGE_SIZE = (80, 80)
# 训练集
DATASET_DIR = '/home/fxz/dl_class/dataset/training_set'
USE_GRAYSCALE = True

def calc_resize_parameters(sw, sh):
    sw_new, sh_new = sw, sh
    dw, dh = IMAGE_SIZE
    pad_w, pad_h = 0, 0
    if sw / sh < dw / dh:
        sw_new = int(dw / dh * sh)
        pad_w = (sw_new - sw) // 2 # 填充左右
    else:
        sh_new = int(dh / dw * sw)
        pad_h = (sh_new - sh) // 2 # 填充上下
    return sw_new, sh_new, pad_w, pad_h

def resize_image(img):  # 缩放图片（可能填充）
    sw, sh = img.size
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    img_new = Image.new("RGB", (sw_new, sh_new))
    img_new.paste(img, (pad_w, pad_h))
    img_new = img_new.resize(IMAGE_SIZE)
    return img_new

def image_to_tensor_grayscale(img):  # 以tensor形式保存图片
    img = img.convert("L")  # 黑白
    arr = numpy.asarray(img)
    t = torch.from_numpy(arr)  # 变为张量
    t = t.unsqueeze(0)  # 升维
    t = t / 255.0   # 归一化，t为0到1
    return t

def image_to_tensor_rgb(img):  # 以tensor形式保存图片（彩色）
    img = img.convert("RGB") 
    arr = numpy.asarray(img)
    t = torch.from_numpy(arr)  
    t = t.transpose(0, 2)  # 转换维度 H,W,C 到 C,W,H
    t = t / 255.0 
    return t

def save_tensor(tensor, path):  # .pt存储tensor文件
    torch.save(tensor, gzip.GzipFile(path, "wb"))

if USE_GRAYSCALE:
    image_to_tensor = image_to_tensor_grayscale
else:
    image_to_tensor = image_to_tensor_rgb
# 数据集转换到 tensor 以后会保存在 data 文件夹下
if not os.path.isdir("data"):
    os.makedirs("data")
# 截取后的人脸图片会保存在 debug_faces 文件夹下
if not os.path.isdir("debug_faces"):
    os.makedirs("debug_faces")

images_map = defaultdict(lambda: [])
def add_image(name, path):
    if os.path.splitext(path)[1].lower() not in (".jpg", ".png"):
        return
    name = name.replace(" ", "").replace("-", "").replace(".", "").replace("_", "").lower()
    images_map[name].append(path)
for dirname in os.listdir(DATASET_DIR):
    dirpath = os.path.join(DATASET_DIR, dirname)
    if not os.path.isdir(dirpath):
        continue
    for filename in os.listdir(dirpath):
        add_image(dirname, os.path.join(DATASET_DIR, dirname, filename))
images_count = sum(map(len, images_map.values()))
print(f"found {len(images_map)} peoples and {images_count} images")

# 保存各个人物的图片数据
for index, (name, paths) in enumerate(images_map.items()):
    images = []
    img_index = 1
    # print(paths)  # 同一个人物的所有照片路径构成一个list
    for path in paths:
        print(path)
        img = tps_photo(path)  # 68标准点对齐
        if not os.path.isdir(f"debug_faces/{name}"):
            os.makedirs(f"debug_faces/{name}")

        if(img.all() == numpy.array([]).all()):
            img = Image.open(path)
            # 裁剪图片让各个数据集的人脸占比更接近
            w, h = img.size
            img = img.crop((int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)))
            # img = img.resize((250, 250), Image.Resampling.LANCZOS)
            img = img.resize((250, 250), Image.LANCZOS)
            img.save(f"debug_faces/{name}/{img_index}.jpg")
            # training_set中无法识别的照片：正常裁剪处理
        else:
            cv2.imwrite(f"debug_faces/{name}/{img_index}.jpg", img)

        img = Image.open(f"debug_faces/{name}/{img_index}.jpg")
        images.append(img)
        img_index += 1
    tensors = [ image_to_tensor(resize_image(img)) for img in images ]
    tensor = torch.stack(tensors)  # 维度：图片数量，3，宽，高
    save_tensor(tensor, os.path.join("data", f"{name}.{len(images)}.pt"))
    print(f"saved {index + 1}/{len(images_map)} people")

print("done")