import torch
import numpy
from PIL import Image, ImageFile
import os
from collections import defaultdict
import gzip

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SIZE = (80, 80)
DATASET_DIR = '/home/fxz/dl_class/dataset/training_set'
USE_GRAYSCALE = True

def calc_resize_parameters(sw, sh):
    sw_new, sh_new = sw, sh
    dw, dh = IMAGE_SIZE
    pad_w, pad_h = 0, 0
    if sw / sh < dw / dh:
        sw_new = int(dw / dh * sh)
        pad_w = (sw_new - sw) // 2
    else:
        sh_new = int(dh / dw * sw)
        pad_h = (sh_new - sh) // 2
    return sw_new, sh_new, pad_w, pad_h

def resize_image(img):
    sw, sh = img.size
    sw_new, sh_new, pad_w, pad_h = calc_resize_parameters(sw, sh)
    img_new = Image.new("RGB", (sw_new, sh_new))
    img_new.paste(img, (pad_w, pad_h))
    img_new = img_new.resize(IMAGE_SIZE)
    return img_new

def image_to_tensor_grayscale(img):
    img = img.convert("L")
    arr = numpy.asarray(img)
    t = torch.from_numpy(arr)
    t = t.unsqueeze(0)
    t = t / 255.0
    return t

def image_to_tensor_rgb(img):
    img = img.convert("RGB")
    arr = numpy.asarray(img)
    t = torch.from_numpy(arr)
    t = t.transpose(0, 2)
    t = t / 255.0
    return t

def save_tensor(tensor, path):
    torch.save(tensor, gzip.GzipFile(path, "wb"))

if USE_GRAYSCALE:
    image_to_tensor = image_to_tensor_grayscale
else:
    image_to_tensor = image_to_tensor_rgb

if not os.path.isdir("data"):
    os.makedirs("data")
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

for index, (name, paths) in enumerate(images_map.items()):
    images = []
    img_index = 1
    for path in paths:
        img = Image.open(path)
        img = resize_image(img)
        if not os.path.isdir(f"debug_faces/{name}"):
            os.makedirs(f"debug_faces/{name}")
        img.save(f"debug_faces/{name}/{img_index}.jpg")
        images.append(img)
        img_index += 1

    tensors = [image_to_tensor(img) for img in images]
    tensor = torch.stack(tensors)
    save_tensor(tensor, os.path.join("data", f"{name}.{len(images)}.pt"))
    print(f"saved {index + 1}/{len(images_map)} people")

print("done")

'''
baseline
直接读取图片，然后将其缩放到指定大小（80x80），同时支持灰度或彩色模式。
由于没有使用面部对齐技术，所以可能会导致面部特征的不一致，特别是在进行面部识别或相似度比较时。
'''