import math
import string
import os
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import random
import cv2
from PIL import Image
import glob
import matplotlib.pyplot as plt


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def add_text_noise(noise, occupancy=50):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = noise.shape
    img_for_cnt = np.zeros((h, w), np.uint8)
    occupancy = np.random.uniform(0, occupancy)
    while True:
        n = random.randint(5, 10)
        random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
        font_scale = np.random.uniform(0.5, 1)
        thickness = random.randint(1, 3)
        (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
        x = random.randint(0, max(0, w - 1 - fw))
        y = random.randint(fh, h - 1 - baseline)
        color = (random.random(), random.random(), random.random())
        # img_plot = imgn_train[0]
        # img_plot = np.ascontiguousarray(np.transpose(img_plot, (1, 2, 0)))
        # img_u = (img_plot*255).astype(np.uint8)
        cv2.putText(noise, random_str, (x, y), font, font_scale, color, thickness)
        cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)
        # plt.imshow(noise[i])
        # plt.show()
        if (img_for_cnt > 0).sum() > h * w * occupancy / 100:
            # noise = torch.FloatTensor(noise)
            break
    return noise


def add_watermark_noise(img_train, scale_lists=None, idx_lists=None, is_test=False, threshold=50):
    watermarks = list()
    for ii in range(12):
        watermarks.append(Image.open(r"./logos/%02d.png" % (ii+1)))

    img_train = img_train.numpy()
    imgn_train = img_train

    _, _, img_h, img_w = img_train.shape
    img_train = np.ascontiguousarray(np.transpose(img_train, (0, 2, 3, 1)))
    imgn_train = np.ascontiguousarray(np.transpose(imgn_train, (0, 2, 3, 1)))
    if scale_lists is None:
        ans_scale_lists = list()
        ans_idx_lists = list()
    else:
        ans_scale_lists = scale_lists
        ans_idx_lists = idx_lists
    for i in range(len(img_train)):
        tmp = Image.fromarray((img_train[i] * 255).astype(np.uint8))
        img_for_cnt = np.zeros((img_h, img_w, 3), np.uint8)
        img_for_cnt = Image.fromarray(img_for_cnt)

        if scale_lists is None:
            scale_list = list()
            idx = random.randint(0, 11)
            ans_idx_lists.append(idx)
            watermark = watermarks[idx]
            w, h = watermark.size
            mark_size = np.array(watermark).size
            if is_test:
                occupancy = threshold
            else:
                occupancy = np.random.uniform(0, 10)
            cnt, ratio = 0, img_w * img_h * 3 * occupancy / 100
            finish = False
            while True:
                if (ratio - cnt ) < mark_size * 0.3:
                    img_train[i] = np.array(tmp).astype(np.float64) / 255.
                    break
                elif (ratio - cnt) < mark_size:
                    scale = (ratio - cnt) * 1.0 / mark_size
                    finish = True
                else:
                    scale = np.random.uniform(0.5, 1)
                scale_list.append(scale)                   

                water = watermark.resize((int(w * scale), int(h * scale)))
                x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
                y = random.randint(0, img_h - int(w * scale))  # int(-h * scale)

                tmp.paste(water, (x, y), water)

                img_for_cnt.paste(water, (x, y), water)
                img_cnt = np.array(img_for_cnt)
                cnt = (img_cnt > 0).sum()
                if finish:
                    img_train[i] = np.array(tmp).astype(np.float64) / 255.
                    break
            ans_scale_lists.append(scale_list)
        else:
            scale_list = scale_lists[i]
            idx = idx_lists[i]
            watermark = watermarks[idx]
            w, h = watermark.size
            for ii in range(len(scale_list)):
                scale = scale_list[ii]
                water = watermark.resize((int(w * scale), int(h * scale)))
                x = random.randint(0, img_w - int(w * scale))
                y = random.randint(0, img_h - int(w * scale))
                tmp.paste(water, (x, y), water)
            img_train[i] = np.array(tmp).astype(np.float64) / 255.

    img_train = np.transpose(img_train, (0, 3, 1, 2))
    imgn_train = np.transpose(imgn_train, (0, 3, 1, 2))
    return img_train, img_train - imgn_train, ans_scale_lists, ans_idx_lists
    

def add_watermark_noise_test(img, num_wm=1):

    watermarks = list()
    for ii in range(12):
        watermarks.append(Image.open(r"./logos/%02d.png" % (ii+1)))

    img = img.numpy()
    imgn = img

    _, _, img_h, img_w = img.shape
    img = np.ascontiguousarray(np.transpose(img, (0, 2, 3, 1)))
    imgn = np.ascontiguousarray(np.transpose(imgn, (0, 2, 3, 1)))
    for i in range(len(img)):
        tmp = Image.fromarray((img[i] * 255).astype(np.uint8))
        
        idx = random.randint(0, 11)
        watermark = watermarks[idx]
        w, h = watermark.size
        mark_size = np.array(watermark).size
            
        for ii in range(num_wm):
            scale = np.random.uniform(0.5, 1)               

            water = watermark.resize((int(w * scale), int(h * scale)))
            x = random.randint(0, img_w - int(w * scale))  # int(-w * scale)
            y = random.randint(0, img_h - int(w * scale))  # int(-h * scale)

            tmp.paste(water, (x, y), water)
        img[i] = np.array(tmp).astype(np.float64) / 255.

    img = np.transpose(img, (0, 3, 1, 2))
    imgn = np.transpose(imgn, (0, 3, 1, 2))
    return img, img - imgn


def findLastCheckpoint(save_path):
    import os
    files = glob.glob(save_path + '/*')
    last_epoch = -1
    for fi in files:
        if not fi.endswith(".pth"):
            continue
        epoch = int(os.path.basename(fi)[-7:-4])
        if last_epoch < epoch:
            last_epoch = epoch
    return last_epoch


if __name__ == '__main__':
    watermark = Image.open("npu.png")
    mark_array = np.array(watermark)
    print(np.array(watermark).size)