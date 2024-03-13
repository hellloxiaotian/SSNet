import argparse
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models import UNet_Atten_3, sum_squared_error
from dataset_my import prepare_data, Dataset
from utils import *
import time

# python train.py
parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--model_name", type=str, default='MyModel', help='name of saved checkpoints')
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--data_path", type=str, default='/data/xiaojingyu', help='path of training data')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--epochs", type=int, default=90, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="saved_models", help='path of log files')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=6, 
                              batch_size=opt.batchSize, pin_memory=True, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    model = UNet_Atten_3()

    criterion = sum_squared_error()
    cuda = torch.cuda.is_available()
    if cuda:
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        criterion = criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = StepLR(optimizer, milestones=[30, 60], gamma=0.2)  # learning rates

    save_path = opt.outf + '/' + opt.model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    initial_epoch = findLastCheckpoint(save_path)  # load the last model in matconvnet style
    if initial_epoch != -1:
        model.load_state_dict(torch.load(os.path.join(save_path, opt.model_name + '%03d.pth' % initial_epoch)))
    initial_epoch += 1
    # training
    noiseL_B = [0, 55]

    for epoch in range(initial_epoch, opt.epochs):
        start_time = time.time()

        epoch_loss = 0
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            optimizer.zero_grad()
            img_train = data
            noise_gauss = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise_gauss.size()[0])
            for n in range(noise_gauss.size()[0]):
                sizeN = noise_gauss[0, :, :, :].size()
                noise_gauss[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)

            noise_gauss_mid = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise_gauss_mid.size()[0])
            for n in range(noise_gauss_mid.size()[0]):
                sizeN = noise_gauss_mid[0, :, :, :].size()
                noise_gauss_mid[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)

            noise_final_gauss = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise_final_gauss.size()[0])
            for n in range(noise_final_gauss.size()[0]):
                sizeN = noise_final_gauss[0, :, :, :].size()
                noise_final_gauss[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)

            imgn_train, wm, scale_lists, idx_lists = add_watermark_noise(img_train)
            imgn_train_final_, wm_final, _, _ = add_watermark_noise(img_train, scale_lists, idx_lists)

            imgn_train_mid = torch.Tensor(imgn_train) + noise_gauss_mid
            imgn_train = torch.Tensor(imgn_train) + noise_gauss

            imgn_train_final_ = torch.Tensor(imgn_train_final_)
            imgn_train_final = imgn_train_final_ + noise_final_gauss

            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            imgn_train_final = Variable(imgn_train_final.cuda())
            imgn_train_mid = Variable(imgn_train_mid.cuda())
            imgn_train_final_ = Variable(imgn_train_final_.cuda())
            main_out_final, out_denoise, out_wm = model(imgn_train)

            loss = (1.0 * criterion(main_out_final, imgn_train_final) + 1.0 * criterion(out_denoise, imgn_train_mid)
                    + 1.0 * criterion(out_wm, imgn_train_final_)) / (imgn_train.size()[0] * 2)

            loss.backward()
            optimizer.step()

        scheduler.step(epoch)
        model.eval()

        save_path = opt.outf + '/' + opt.model_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, opt.model_name + '%03d.pth' % epoch))


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path=opt.data_path, patch_size=256, stride=128, aug_times=1, mode='color')

    main()
