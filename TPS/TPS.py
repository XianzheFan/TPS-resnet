import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import dlib
import cv2
from PIL import Image, ImageFile

DEVICE = torch.device("cpu")

def norm(points_int, width, height):  # 像素坐标归一化 -1-1
	points_int_clone = torch.from_numpy(points_int).detach().float().to(DEVICE)
	x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
	y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
	return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)

class TPS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h, device):
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]  # 老版本torch==1.8.0
        Q = torch.linalg.solve(L, Z)
        W, A = Q[:, :k], Q[:, k:]

        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        n, k = grid.shape[:2]
        device = grid.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid

        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)

# 设定两张图片的68标准点
predictor_path = "model/shape_predictor_68_face_landmarks.dat"  # 官方训练的模型
detector = dlib.get_frontal_face_detector()  # 检测框
predictor = dlib.shape_predictor(predictor_path)  # 关键点检测

f = 'model/merge_clip.jpg'
img = cv2.imread(f, cv2.IMREAD_COLOR)
# 默认参数，读入彩色图片，忽略alpha通道
dets = detector(img, 0)  # 返回人脸集合

points = []
for index, face in enumerate(dets):
    shape = predictor(img, face)
    for index, pt in enumerate(shape.parts()):
        pt_pos = (pt.x, pt.y)
        pt_pos = list(pt_pos)
        points.append(pt_pos)

def tps_photo(path):
    img1 = cv2.imread(path, cv2.IMREAD_COLOR)
    dets1 = detector(img1, 0) 

    points1 = []
    for index1, face1 in enumerate(dets1):
        shape1 = predictor(img1, face1)
        for index1, pt1 in enumerate(shape1.parts()):
            pt_pos1 = (pt1.x, pt1.y)
            pt_pos1 = list(pt_pos1)
            points1.append(pt_pos1)
    
    if(len(points1) > 68):
        points1 = points1[:68]  # 多人情形
    elif(len(points1) == 0):
        return np.array([])

    ten_img = ToTensor()(img1).to(DEVICE)
    # 把PIL.Image或ndarray从(H x W x C)形状转换为(C x H x W)的tensor
    h, w = ten_img.shape[1], ten_img.shape[2]
    ten_source = norm(np.array(points1), w, h) 
    ten_target = norm(np.array(points), w, h)

    tps = TPS()
    warped_grid = tps(ten_target[None, ...], ten_source[None, ...], w, h, DEVICE)
    ten_wrp = torch.grid_sampler_2d(ten_img[None, ...], warped_grid, 0, 0, align_corners=True)
    new_img_torch = np.array(ToPILImage()(ten_wrp[0].cpu()))
    return new_img_torch