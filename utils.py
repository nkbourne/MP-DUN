import numpy as np
import math
import torch
from PIL import Image

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def fill_image(x, block_size):
    c = x.size()[0]
    ori_h = x.size()[1]
    h_lack = 0
    ori_w = x.size()[2]
    w_lack = 0
    if ori_h % block_size != 0:
        h_lack = block_size - ori_h % block_size
        temp_h = torch.zeros(c, h_lack, ori_w)
        h = ori_h + h_lack
        x = torch.cat((x, temp_h), 1)

    if ori_w % block_size != 0:
        w_lack = block_size - ori_w % block_size
        temp_w = torch.zeros(c, h, w_lack)
        x = torch.cat((x, temp_w), 2)
    return x, ori_h ,ori_w

def read_img(path, mode = 'G'):
    image = np.array(Image.open(path))
    if len(image.shape) == 3: #rgb转y
        image = rgb2ycbcr(image)
    image = (image/127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image)
    if mode == 'RGB':
        image = np.transpose(image,(2,0,1))
    else:
        image = torch.unsqueeze(image, 0)
    return image

def make_batch(image, block_size, device, channels = 3):
    x ,h ,w = fill_image(image, block_size)
    batchs = torch.unsqueeze(x, 0)
    batchs = batchs.to(memory_format=torch.contiguous_format).float()
    batchs = batchs.to(device)
    return batchs, h, w