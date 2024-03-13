import cv2
import os
import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import UNet_Atten_3
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--ckpt_dir", type=str, default="ckpts", help='path of log files')
parser.add_argument("--test_data", type=str, default='Kodak24',
                    help='one testset of CBSD68, Kodak24, McMaster, LVWDs, CLWDs')
parser.add_argument("--test_dir", type=str, default=r'/data/xiaojingyu/', help='path of test sets')
parser.add_argument("--test_noiseL", type=list, default=list([15, 25, 50]), help='the noise level used on')
parser.add_argument("--wm_threshold", type=list, default=list([10, 20, 30]), help='the level of watermark')
parser.add_argument("--wm_num", type=list, default=list([1, 2, 4]), help='the number of watermarks')
opt = parser.parse_args()

#  CUDA_VISIBLE_DEVICES=1 python test.py --ckpt_dir ckpts --test_data Kodak24 --test_dir /data/xiaojingyu/


def normalize(data):
    return data / 255.


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        cv2.imwrite(path, np.clip(result, 0, 255))


def test():
    # Build model
    net = UNet_Atten_3()
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    seed = 6  # 8  85
    adjust = 32

    model.load_state_dict(torch.load(os.path.join(opt.ckpt_dir, 'MyModel.pth')))
    model.eval()

    files_source = glob.glob(os.path.join(opt.test_dir, opt.test_data, '*'))
    files_source.sort()
    # process data
    for idx_n in range(len(opt.test_noiseL)):
        for idx_t in range(len(opt.wm_num)):
            psnr_test, ssim_test = 0, 0
            valid_num = 0
            for f in files_source:
                # image
                if not (f.endswith(".jpg") or f.endswith(".bmp") or f.endswith(".png")):
                    continue
                valid_num += 1
                Img = cv2.imread(f)
                Img = normalize(np.float32(Img[:, :, :]))
                Img = np.expand_dims(Img, 0)
                # Img = np.expand_dims(Img, 1)
                Img = np.transpose(Img, (0, 3, 1, 2))
                _, _, w, h = Img.shape
                w_new = int(int(w / adjust) * adjust)
                h_new = int(int(h / adjust) * adjust)
                # Img = Img[:, :, 0:w_new, 0:h_new]
                ISource = torch.Tensor(Img)
                # noise
                # seed_torch(seed) # for result reproducity
                noise_level, wm_num = opt.test_noiseL[idx_n] / 255., opt.wm_num[idx_t]
                noise_gs = torch.FloatTensor(ISource.size()).normal_(mean=0, std=noise_level)
                INoisy, noise = add_watermark_noise_test(ISource, num_wm=wm_num)

                INoisy = torch.Tensor(INoisy) + noise_gs
                ISource, INoisy = Variable(ISource).cuda(), Variable(INoisy).cuda()
                Result = INoisy.clone()
                with torch.no_grad():  # this can save much memory
                    Out = model(INoisy[:, :, :w_new, :h_new])[0]
                    torch.cuda.synchronize()
                    Result[:, :, 0:w_new, 0:h_new] = Out
                    if adjust != 1:
                        peices = list([INoisy[:, :, w - w_new:, :h_new], INoisy[:, :, :w_new, h - h_new:],
                                       INoisy[:, :, w - w_new:, h - h_new:]])
                        out_peices = list()
                        for ii in range(len(peices)):
                            out_peices.append(torch.clamp(model(peices[ii])[0], 0., 1.))
                        Result[:, :, w_new:, :h_new] = out_peices[0][:, :, 2 * w_new - w:, :]
                        Result[:, :, :w_new, h_new:] = out_peices[1][:, :, :, 2 * h_new - h:]
                        Result[:, :, w_new:, h_new:] = out_peices[2][:, :, 2 * w_new - w:, 2 * h_new - h:]

                    Result = torch.clamp(Result, 0., 1.)
                    INoisy = torch.clamp(INoisy, 0., 1.)

                    if not os.path.exists("data/output_" + opt.test_data + "/"):
                        os.makedirs("data/output_" + opt.test_data + "/")

                    result = np.transpose(Result[0, :, :, :].cpu().numpy(), (1, 2, 0))
                    result2 = np.transpose(ISource[0, :, :, :].cpu().numpy(), (1, 2, 0))
                    result = np.ascontiguousarray(result)
                    result2 = np.ascontiguousarray(result2)
                    psnr_x = peak_signal_noise_ratio(result, result2)
                    ssim_x = structural_similarity(result, result2, channel_axis=2, data_range=1)
                    psnr_test += psnr_x
                    ssim_test += ssim_x
            psnr_test /= valid_num
            ssim_test /= valid_num
            print(
                "wm_num {} on {}: PSNR {} SSIM {}".format(wm_num, opt.test_data, psnr_test, ssim_test))


if __name__ == "__main__":
    test()


