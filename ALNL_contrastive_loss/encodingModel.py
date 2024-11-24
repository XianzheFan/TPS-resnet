import os
import sys
import torch
import gzip
import random
import numpy as np
import torchvision
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import lru_cache
from processingPhoto import resize_image, save_tensor, image_to_tensor_grayscale, image_to_tensor_rgb
import csv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import math

RESULT_DIR = '/home/fxz/dl_class/dataset/test_pair'
# 每一轮训练中样本的重复次数
REPEAT_SAMPLES = 3
# 用于对比的不同人物（负样本）数量（只有一张图的人为负样本）
NEGATIVE_SAMPLES = 1
# NEGATIVE_SAMPLES = 10
# 负样本中随机抽取的数量
NEGATIVE_RANDOM_SAMPLES = 1
# NEGATIVE_RANDOM_SAMPLES = 3
# 跳过最接近的人脸数量
NEGATIVE_SKIP_NEAREST = 20
# 识别同一人物最少要求的图片数量
MINIMAL_POSITIVE_SAMPLES = 2
# 是否先转换为黑白图片（防止过拟合）
USE_GRAYSCALE = True

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_plots(training_accuracy_history, validating_accuracy_history, lossDrawjpg, lossDrawjpgvali):
    # 绘制并保存准确率图表
    plt.figure()
    plt.plot(training_accuracy_history, label="training_accuracy")
    plt.plot(validating_accuracy_history, label="validating_accuracy")
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy.jpg')
    plt.clf()

    # 绘制并保存损失值图表
    plt.figure()
    plt.plot(lossDrawjpg, label="training_loss")
    plt.plot(lossDrawjpgvali, label="validating_loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.jpg')
    plt.clf()

def save_metrics_to_csv(training_accuracy_history, validating_accuracy_history, lossDrawjpg, lossDrawjpgvali, filename="metrics.csv"):
    with open(filename, mode='w', newline='') as file:  # 使用'w'模式打开文件
        writer = csv.writer(file)
        
        # 写入标题
        writer.writerow(["Epoch", "Training Accuracy", "Validating Accuracy", "Training Loss", "Validating Loss"])
        
        # 写入数据
        for i in range(len(training_accuracy_history)):
            writer.writerow([
                i + 1,  # Epoch
                training_accuracy_history[i],  # Training Accuracy
                validating_accuracy_history[i] if i < len(validating_accuracy_history) else '',  # Validating Accuracy
                lossDrawjpg[i],  # Training Loss
                lossDrawjpgvali[i] if i < len(lossDrawjpgvali) else ''  # Validating Loss
            ])

class FaceRecognitionModel(nn.Module):  # 计算人脸的编码（基于 ResNet）
    # 编码长度
    EmbeddedSize = 16
    # 要求不同人物编码之间的距离（平方和）
    ExclusiveMargin = 0.2 

    def __init__(self):
        super().__init__()
        # Resnet 的实现（不使用预训练模型）
        self.resnet = torchvision.models.resnet18(num_classes=256)
        # 支持黑白图片
        if USE_GRAYSCALE:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 参数不能变，是resnet共有的特性

        # 添加 Batch Normalization 层
        self.batch_norm = nn.BatchNorm1d(256)

        # 最终输出编码的线性模型
        self.encode_model = nn.Sequential(
            nn.ReLU(inplace=True),  # 原地赋值，节省内存
            nn.Linear(256, 128),  # 全连接层（二维张量）
            nn.ReLU(inplace=True),
            nn.Linear(128, FaceRecognitionModel.EmbeddedSize))

    def forward(self, x):
        tmp = self.resnet(x)
        tmp = self.batch_norm(tmp)  # 应用 Batch Normalization
        y = self.encode_model(tmp)
        return y
    
    @staticmethod
    def contrastive_loss(a, b, c, margin):
        """
        Contrastive loss function.
        - a and b: embeddings of a positive pair (similar images)
        - c: embedding of a negative example (a dissimilar image)
        - margin: contrastive margin
        """
        positive_pair_distance = (a - b).pow(2).sum()
        # Assuming c contains multiple negative samples, compute the distance from a to all negative samples
        negative_pair_distances = (a.unsqueeze(0) - c).pow(2).sum(dim=1)
        # Take the minimum distance for the hardest negative
        hardest_negative_distance = negative_pair_distances.min()
        # Calculate the loss such that positive pairs have a distance less than margin from the negatives
        loss = torch.relu(positive_pair_distance - hardest_negative_distance + margin)
        return loss
    
    @staticmethod
    def angular_loss(xa, xp, xn, alpha):
        # print("Shapes:", xa.shape, xp.shape, xn.shape)

        xc = (xa + xp) / 2
        dist_ap = torch.norm(xa - xp, p=2, dim=0) ** 2
        tan_alpha = math.tan(alpha)
        dist_nc = 4 * tan_alpha ** 2 * torch.norm(xn - xc, p=2, dim=1) ** 2
        loss = torch.relu(dist_ap - dist_nc)
        return loss.mean()

    # @staticmethod
    # def angular_loss(xa, xp, xn, alpha):
    #     # Calculate the circumcircle's center and radius
    #     xc = (xa + xp) / 2
    #     radius = torch.norm(xa - xp, p=2, dim=0) / 2

    #     # Calculate the squared distance from the negative samples to the center
    #     dist_nc_squared = torch.norm(xn - xc.unsqueeze(0), p=2, dim=1) ** 2

    #     # Calculate the angular loss as per the thesis
    #     tan_alpha = math.tan(alpha)
    #     angular_loss_value = torch.relu(radius ** 2 - 4 * tan_alpha ** 2 * dist_nc_squared)
    #     return angular_loss_value.mean()

    @staticmethod
    def n_pair_loss(anchor, positive, negatives, l2_reg=0.02):
        """
        Compute the N-pair loss.
        - anchor and positive: embeddings of the positive pair (similar images)
        - negatives: embeddings of negative examples (dissimilar images)
        - l2_reg: a regularization term for L2 norm
        """
        # normalize to avoid numerical overflow
        anchor = F.normalize(anchor, p=2, dim=0)
        positive = F.normalize(positive, p=2, dim=0)
        negatives = F.normalize(negatives, p=2, dim=1)
        
        # Cal. scores for anchor-positive and anchor-negative pairs
        positive_score = torch.matmul(anchor, positive.t())  # Should be high
        negative_scores = torch.matmul(anchor, negatives.t())  # Should be low

        # Calculate the log-sum-exp of the negative scores
        # This acts as a smooth maximum; we use log-sum-exp for numerical stability
        lse_negative_scores = torch.logsumexp(negative_scores, dim=0)

        # The loss is the difference between the positive score and the log-sum-exp of negative scores
        loss = F.relu(lse_negative_scores - positive_score + l2_reg)

        # batch's mean
        return loss.mean()

    @staticmethod
    def loss_function(predicted):
        losses = []
        for index in range(0, predicted.shape[0], 2 + NEGATIVE_SAMPLES):
            a = predicted[index]   # 基础人物的编码
            b = predicted[index + 1]  #（另一张图片）
            c = predicted[index + 2 : index + 2 + NEGATIVE_SAMPLES]  # 对比人物的编码（10个）
            
            # degree_alpha = 45
            # radian_alpha = math.radians(degree_alpha)
            # # implement angular loss function
            # angular_loss = FaceRecognitionModel.angular_loss(a, b, c, alpha=radian_alpha)
            # losses.append(angular_loss)

            # degree_alpha = 45
            # radian_alpha = math.radians(degree_alpha)
            # # Calculate the N-pair loss
            # n_pair_loss_value = FaceRecognitionModel.n_pair_loss(a, b, c)

            # # Calculate the Angular Loss
            # angular_loss_value = FaceRecognitionModel.angular_loss(a, b, c, alpha=radian_alpha)

            # # Combine N-pair and Angular Loss
            # total_loss = n_pair_loss_value + angular_loss_value
            # print(total_loss)
            # losses.append(total_loss)

            # # Here, we will use a contrastive loss function instead of angular loss
            margin = 1.0  # could be a hyperparameter
            contrastive_loss_value = FaceRecognitionModel.contrastive_loss(a, b, c, margin)
            losses.append(contrastive_loss_value)

        loss_total = torch.stack(losses).mean()
        print(f"loss:{loss_total.item()}")
        return loss_total, loss_total.item()

    @staticmethod
    def calc_accuracy(predicted):  # 正确率
        total_count = 0
        correct_count = 0
        for index in range(0, predicted.shape[0], 2 + NEGATIVE_SAMPLES):
            a = predicted[index]
            b = predicted[index + 1] 
            c = predicted[index + 2 : index + 2 + NEGATIVE_SAMPLES] 
            diff_positive = (a - b).pow(2).sum()
            diff_negative = (a - c).pow(2).sum(dim=1)
            if (diff_positive < diff_negative).sum() == diff_negative.shape[0]:
                correct_count += 1
            total_count += 1
        return correct_count / total_count

# 减少读取时间，缓存读取的 tensor
# 如果内存不够应该适当减少 maxsize
@lru_cache(maxsize=10000)
def load_tensor(path):
    return torch.load(gzip.GzipFile(path, "rb"))

if USE_GRAYSCALE:  # 是否黑白
    image_to_tensor = image_to_tensor_grayscale
else:
    image_to_tensor = image_to_tensor_rgb

def train():
    # 创建模型实例
    model = FaceRecognitionModel().to(device)

    loss_function = model.loss_function

    # 优化器，lr为学习率。torch.optim集成的优化器只有L2正则化
    # weight_decay相当于L2正则化中的lambda
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.003)

    # 训练集和验证集的正确率变化
    training_accuracy_history = []
    validating_accuracy_history = []

    # 最高的验证集正确率
    validating_accuracy_highest = -1
    validating_accuracy_highest_epoch = 0

    calc_accuracy = model.calc_accuracy

    # 读取人物列表，区分图片数量足够的人物和图片数量不足的人物（负样本）
    filenames = os.listdir("data")
    multiple_samples = []
    single_samples = []
    for filename in filenames:
        if int(filename.split('.')[-2]) >= MINIMAL_POSITIVE_SAMPLES:
            multiple_samples.append(filename)
        else:
            single_samples.append(filename)
    random.shuffle(multiple_samples)
    random.shuffle(single_samples)
    total_multiple_samples = len(multiple_samples)
    total_single_samples = len(single_samples)

    # 训练集：验证集：测试集 = 8:1:1
    training_set = multiple_samples[:int(total_multiple_samples*0.8)]
    training_set_single = single_samples[:int(total_single_samples*0.8)]
    validating_set = multiple_samples[int(total_multiple_samples*0.8):int(total_multiple_samples*0.9)]
    validating_set_single = single_samples[int(total_single_samples*0.8):int(total_single_samples*0.9)]
    testing_set = multiple_samples[int(total_multiple_samples*0.9):]
    testing_set_single = single_samples[int(total_single_samples*0.9):]

    # 训练集编码
    training_image_to_vector_index = {}
    training_vector_index_to_image = {}
    for filename in training_set + training_set_single:
        for image_index in range(int(filename.split('.')[1])):
            vector_index = len(training_image_to_vector_index)
            training_image_to_vector_index[(filename, image_index)] = vector_index
            training_vector_index_to_image[vector_index] = (filename, image_index)
    training_vectors = torch.zeros(len(training_image_to_vector_index), FaceRecognitionModel.EmbeddedSize)
    training_vectors_calculated_indices = set()

    # 训练集生成
    # [A, P, N]
    def generate_inputs(dataset_multiple, dataset_single, batch_size):
        is_training = dataset_multiple == training_set
        if is_training:
            calculated_index_list = list(training_vectors_calculated_indices)
            calculated_index_set = set(calculated_index_list)
            calculated_index_to_image = {
                ci: training_vector_index_to_image[vi]
                for ci, vi in enumerate(calculated_index_list)
            }
            training_vectors_calculated = training_vectors[calculated_index_list]
        # 枚举数据集，会重复 REPEAT_SAMPLES 次以减少随机选择导致的正确率浮动
        image_tensors = []
        vector_indices = []
        for base_filename in dataset_multiple * REPEAT_SAMPLES:
            # 读取基础人物的图片
            base_tensor = load_tensor(os.path.join("data", base_filename))
            base_tensors = list(enumerate(base_tensor))
            # 打乱顺序，然后两张两张图片的选取基础图片和正样本
            random.shuffle(base_tensors)
            for index in range(0, len(base_tensors)-1, 2):
                # 添加基础图片和正样本到列表
                anchor_image_index, anchor_tensor = base_tensors[index]
                positive_image_index, positive_tensor = base_tensors[index+1]
                image_tensors.append(anchor_tensor)
                image_tensors.append(positive_tensor)
                if is_training:
                    vector_indices.append(training_image_to_vector_index[(base_filename, anchor_image_index)])
                    vector_indices.append(training_image_to_vector_index[(base_filename, positive_image_index)])
                # train：计算基础图片的编码与其他编码的距离
                nearest_indices = []
                if is_training:
                    vector_index = training_image_to_vector_index[(base_filename, anchor_image_index)]
                    if vector_index in calculated_index_set:
                        nearest_indices = ((training_vectors_calculated -
                            training_vectors[vector_index]).abs().sum(dim=1).sort().indices).tolist()
                # train：编码最接近的样本 + 随机样本（负样本）
                # validate + test：随机选取样本
                if is_training and nearest_indices:
                    negative_samples = NEGATIVE_SAMPLES - NEGATIVE_RANDOM_SAMPLES
                    negative_random_samples = NEGATIVE_RANDOM_SAMPLES
                else:
                    negative_samples = 0
                    negative_random_samples = NEGATIVE_SAMPLES
                negative_skip_nearest = NEGATIVE_SKIP_NEAREST
                for calculated_index in nearest_indices:
                    if negative_samples <= 0:
                        break
                    filename, image_index = calculated_index_to_image[calculated_index]
                    if filename == base_filename:
                        continue  # 跳过同一人物
                    if negative_skip_nearest > 0:
                        negative_skip_nearest -= 1
                        continue  # 跳过非常相似的人物
                    target_tensor = load_tensor(os.path.join("data", filename))
                    # 负样本
                    image_tensors.append(target_tensor[image_index])
                    if is_training:
                        vector_indices.append(training_image_to_vector_index[(filename, image_index)])
                    negative_samples -= 1
                while negative_random_samples > 0:
                    file_index = random.randint(0, len(dataset_multiple) + len(dataset_single) - 1)
                    if file_index < len(dataset_multiple):
                        filename = dataset_multiple[file_index]
                    else:
                        filename = dataset_single[file_index - len(dataset_multiple)]
                    if filename == base_filename:
                        continue  # 跳过同一人物
                    target_tensor = load_tensor(os.path.join("data", filename))
                    image_index = random.randint(0, target_tensor.shape[0] - 1)
                    # 负样本
                    image_tensors.append(target_tensor[image_index])
                    if is_training:
                        vector_indices.append(training_image_to_vector_index[(filename, image_index)])
                    negative_random_samples -= 1
                assert negative_samples == 0
                assert negative_random_samples == 0
                # 如果图片数量大于batch，则返回
                if len(image_tensors) >= batch_size:
                    yield torch.stack(image_tensors).to(device), vector_indices
                    image_tensors.clear()
                    vector_indices.clear()
        if image_tensors:
            yield torch.stack(image_tensors).to(device), vector_indices

    # 开始训练过程
    lossDrawjpg = []
    lossDrawjpgvali = []

    # 从之前的参数开始训练
    model_weights_path = "model.recognition.pt"
    if os.path.exists(model_weights_path):
        model.load_state_dict(load_tensor(model_weights_path))
    else:
        print("No pre-trained weights found, starting training from scratch.")

    writer = SummaryWriter()
    try:
        for epoch in range(0, 200):
            print(f"epoch: {epoch}")

            # 训练模式
            model.train()
            training_accuracy_list = []
            for index, (batch_x, vector_indices) in enumerate(
                generate_inputs(training_set, training_set_single, 400)):
                # 计算预测值
                predicted = model(batch_x)
                loss, lossDraw = loss_function(predicted)
                lossDrawjpg.append(lossDraw)
                # 反向传播
                loss.backward()
                # 改参数
                optimizer.step()
                # 清空梯度
                optimizer.zero_grad()
                # 各个人物的编码
                for vector_index, vector in zip(vector_indices, predicted):
                    training_vectors[vector_index] = vector.to("cpu").detach()
                    training_vectors_calculated_indices.add(vector_index)
                # 不自动求导
                with torch.no_grad():
                    training_batch_accuracy = calc_accuracy(predicted)
                training_accuracy_list.append(training_batch_accuracy)
                print(f"epoch: {epoch}, batch: {index}, accuracy: {training_batch_accuracy}")
            training_accuracy = sum(training_accuracy_list) / len(training_accuracy_list)
            training_accuracy_history.append(training_accuracy)
            print(f"training accuracy: {training_accuracy}")

            # 验证集
            model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
            validating_accuracy_list = []
            for batch_x, _ in generate_inputs(validating_set, validating_set_single, 100):
                predicted = model(batch_x)
                lossvali, lossDrawvali = loss_function(predicted)
                lossDrawjpgvali.append(lossDrawvali)
                validating_batch_accuracy = calc_accuracy(predicted)
                validating_accuracy_list.append(validating_batch_accuracy)
                # 释放 predicted 占用的显存避免显存不足的错误
                predicted = None
            validating_accuracy = sum(validating_accuracy_list) / len(validating_accuracy_list)
            validating_accuracy_history.append(validating_accuracy)
            print(f"validating accuracy: {validating_accuracy}")


            # 在每三个epoch保存图像和指标
            if (epoch + 1) % 3 == 0:
                save_plots(training_accuracy_history, validating_accuracy_history, lossDrawjpg, lossDrawjpgvali)
                save_metrics_to_csv(training_accuracy_history, validating_accuracy_history, lossDrawjpg, lossDrawjpgvali)


            # 使用 writer 记录训练和验证的损失和准确率
            writer.add_scalar("Loss/train", lossDraw, epoch)
            writer.add_scalar("Accuracy/train", training_accuracy, epoch)
            writer.add_scalar("Loss/validation", lossDrawvali, epoch)
            writer.add_scalar("Accuracy/validation", validating_accuracy, epoch)


            # 最高验证集正确率
            # 允许 1% 的波动（可增加训练次数）
            if (validating_accuracy + 0.01) > validating_accuracy_highest:
                if validating_accuracy > validating_accuracy_highest:
                    validating_accuracy_highest = validating_accuracy
                    print("highest validating accuracy updated")
                else:
                    print("highest validating accuracy not dropped")
                validating_accuracy_highest_epoch = epoch
                save_tensor(model.state_dict(), "model.recognition.pt")
            elif epoch - validating_accuracy_highest_epoch > 12:
                # 12次训练后没有刷新，结束
                print("stop training because validating accuracy dropped from highest in 12 epoches")
                break

        # 最高正确率模型状态
        print(f"highest validating accuracy: {validating_accuracy_highest}",
            f"from epoch {validating_accuracy_highest_epoch}")
        model.load_state_dict(load_tensor("model.recognition.pt"))

        # 测试集
        testing_accuracy_list = []
        for batch_x, _ in generate_inputs(testing_set, testing_set_single, 100):
            predicted = model(batch_x)
            testing_batch_accuracy = calc_accuracy(predicted)
            testing_accuracy_list.append(testing_batch_accuracy)
        testing_accuracy = sum(testing_accuracy_list) / len(testing_accuracy_list)
        print(f"testing accuracy: {testing_accuracy}")

    except Exception as e:
        print(f"训练过程中出现异常: {e}")
    finally:
        # 不管程序是正常结束还是因异常中断，都会执行这里的代码：保存准确率和损失值的图表以及数据
        writer.close()
        save_plots(training_accuracy_history, validating_accuracy_history, lossDrawjpg, lossDrawjpgvali)
        save_metrics_to_csv(training_accuracy_history, validating_accuracy_history, lossDrawjpg, lossDrawjpgvali)

def resize_and_tensorize_image(image_path):
    img = Image.open(image_path)
    img = resize_image(img)  # 假设 resize_image 能够正确处理图片大小
    tensor = image_to_tensor(img)  # 假设 image_to_tensor 能够正确转换图片为张量
    return tensor

def verify(verify_loss):
    # read ground truth data:
    with open('ground_truth.txt', 'r') as f:
        ground_truths = [int(line.strip()) for line in f.readlines()]

    # 数据集转换到 tensor 以后会保存在 testdata 文件夹下
    if not os.path.isdir("testdata"):
        os.makedirs("testdata")

    # 截取后的人脸图片会保存在 test_debug_faces 文件夹下
    if not os.path.isdir("test_debug_faces"):
        os.makedirs("test_debug_faces")

    # images_map = defaultdict(lambda: [])
    # def add_image(name, path):
    #     if os.path.splitext(path)[1].lower() not in (".jpg", ".png"):
    #         return
    #     name = name.replace(" ", "").replace("-", "").replace(".", "").replace("_", "").lower()
    #     images_map[name].append(path)

    # for dirname in os.listdir(RESULT_DIR):
    #     dirpath = os.path.join(RESULT_DIR, dirname)
    #     if not os.path.isdir(dirpath):
    #         continue
    #     for filename in os.listdir(dirpath):
    #         add_image(dirname, os.path.join(RESULT_DIR, dirname, filename))
    # images_count = sum(map(len, images_map.values()))
    # print(f"found {len(images_map)} pairs and {images_count} images")

    # for index, (name, paths) in enumerate(images_map.items()):
    #     print(f"name: {name}  paths: {paths}")
    #     for i, path in enumerate(paths):
    #         output_path = f"test_debug_faces/{name}.{i}.jpg"
    #         if os.path.exists(output_path):
    #             print(f"Skipped {name}.{i}, jpg already exists.")
    #             continue

    #         img = Image.open(path)
    #         w, h = img.size
    #         img = img.crop((int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)))
    #         img = img.resize((250, 250), Image.LANCZOS)  # pillow==8.4.0
    #         # img = img.resize((250, 250), Image.Resampling.LANCZOS)
    #         img.save(f"test_debug_faces/{name}.{i}.jpg")
    #         tensor = image_to_tensor(resize_image(img))
    #         save_tensor(tensor, os.path.join("testdata", f"{name}.{i}.pt"))
    #     print(f"saved {index+1}/{len(images_map)} peoples")
    # print("done")
        


    recognize_model = FaceRecognitionModel().to(device)
    recognize_model.load_state_dict(load_tensor("model.recognition.pt"))
    recognize_model.eval()

    test_pairs_dir = '/home/fxz/dl_class/dataset/test_pair'
    test = "testdata"
    filename = 'test_result.txt'
    results = []
    diffs = []
    similarities = []
    for i in range(600):
        pair_dir = os.path.join(test_pairs_dir, str(i))
        images = [resize_and_tensorize_image(os.path.join(pair_dir, file)) for file in sorted(os.listdir(pair_dir))]
        
        if len(images) != 2:
            print(f"Folder {pair_dir} does not contain exactly two images.")
            continue        

        image_tensors = torch.stack(images).to(device)

        predicted = recognize_model(image_tensors)
        
        if verify_loss == 'c':
            # Euclidian Dist. => For contrastive loss
            a, b = predicted[0], predicted[1]
            # print(f"a.shape: {a.size()}")  # 16
            # print(f"b.shape: {b.size()}")  # 16
            diff = (a - b).pow(2).sum()
            diffs.append(diff.item())
        elif verify_loss == 'a':
            # # cosine similarity
            a, b = nn.functional.normalize(predicted[0], p=2, dim=0), nn.functional.normalize(predicted[1], p=2, dim=0)
            cosine_similarity = nn.functional.cosine_similarity(a, b, dim=0)
            diffs.append(cosine_similarity.item())
        else:
            raise ValueError("You should type c(contrastive loss) or a(angular+np loss)")
    
    if verify_loss == 'c':
        threshold_value = np.linspace(0, 10, num=100)
        
    elif verify_loss == 'a':
        threshold_value = np.linspace(-1, 1, num=100)

    best_acc = 0
    best_threshold = 0
    for threshold in threshold_value:
        predictions = [0 if diff >= threshold else 1 for diff in diffs]

        assert len(ground_truths) == len(predictions)
        correct_predictions = sum(gt == pred for gt, pred in zip(ground_truths, predictions))
        acc = correct_predictions / len(ground_truths)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_acc}")

    # with open('test_result.txt', 'w') as f:
    #     header = "{:<15} {:<15} {:<15}".format("Same Person", "Difference", "Correctness")
    #     f.write(header + '\n')
    #     for gt, diff in zip(ground_truths, diffs):
    #         result = 0 if diff >= best_threshold else 1
    #         correctness = "Correct" if result == gt else "Wrong"
    #         row = "{:<15} {:<15} {:<15}".format(result, round(diff, 5), correctness)
    #         f.write(row + '\n')

    with open('test_result.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["Same Person", "Difference", "Correctness"]
        csvwriter.writerow(header)

        for gt, diff in zip(ground_truths, diffs):
            result = 0 if diff >= best_threshold else 1
            correctness = "Correct" if result == gt else "Wrong"
            row = [result, round(diff, 5), correctness]
            csvwriter.writerow(row)



def main():
    # 设置随机数种子
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if len(sys.argv) < 2:
        print(f"Please run: {sys.argv[0]} train|test")
        exit()

    operation = sys.argv[1]
    # verify_loss = sys.argv[2]
    if operation == "train":
        train()
    elif operation == "test":
        if len(sys.argv) < 3:
            print("For testing, please specify loss type: 'c' for contrastive or 'a' for angular+np")
            sys.exit(1)
        verify_loss = sys.argv[2]
        verify(verify_loss)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

if __name__ == "__main__":
    main()