import glob
import os
from pickletools import int4
from sys import path_hooks
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import itertools
from skimage.metrics import peak_signal_noise_ratio as psnr
from easydict import EasyDict as edict
from scipy import io
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hyperparameters
opt = edict()
opt.resume = False
opt.ckpt_dir = "./checkpoint"
opt.ckpt_reload = '0'
opt.train_dir = "./data/Low_High_CT_mat_slice"
opt.test_dir = "./data/test"
opt.result_save_dir = "./result"
opt.test_save_dir = "./test"
opt.best_save_dir = "./best"
opt.lr = 0.0002
opt.epoch = 200
opt.batch_size = 4
opt.img_size = 256
# img_size 256 -> 9 res_blocks in U-net
opt.loss_lambda = 10

# data load & preprocess
train_path = glob.glob(os.path.join(opt.train_dir, '**/*.mat'), recursive=True)
test_path = glob.glob(os.path.join(opt.test_dir, '**/*.mat'), recursive=True)
low = []
high = []
for path in train_path:
    img_mat = io.loadmat(path)
    img_low_high = img_mat.get('imdb')
    low.append(img_low_high[0][0][0])
    high.append(img_low_high[0][0][1])
test_low = []
test_high = []
for path in test_path:
    test_mat = io.loadmat(path)
    test_low_high = test_mat.get('imdb')
    test_low.append(test_low_high[0][0][0])
    test_high.append(test_low_high[0][0][1])

class AAPM(Dataset):
    def __init__(self, train, transform = None):
        self.train_len = len(low)
        self.low = low
        self.high = high
        self.test_len = len(test_low)
        self.test_low = test_low
        self.test_high = test_high
        self.transform = transform
        self.train = train

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len
    def __getitem__(self, idx):
        if self.transform is not None:
            if self.train:
                low = self.transform(self.low[idx])
                high = self.transform(self.high[idx])
                low_min, low_max = low.min(), low.max()
                high_min, high_max = high.min(), high.max()
                # min-max scale
                low = (low-low_min)/(low_max-low_min)
                high = (high-high_min)/(high_max-high_min)
                return low, high
            else:
                low = self.transform(self.test_low[idx])
                high = self.transform(self.test_high[idx])
                low_min, low_max = low.min(), low.max()
                high_min, high_max = high.min(), high.max()
                # min-max scale
                low = (low-low_min)/(low_max-low_min)
                high = (high-high_min)/(high_max-high_min)
                return low, high
        else:
            return self.low, self.high

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(opt.img_size),
    transforms.ToTensor(),
])

train_dataset = AAPM(train = True, transform = transform)
train_data = DataLoader(dataset=train_dataset, batch_size = opt.batch_size, shuffle=True, num_workers = 1, drop_last = True)
test_dataset = AAPM(train = False, transform = transform)
test_data = DataLoader(dataset=test_dataset, batch_size = opt.batch_size, drop_last = True, shuffle = True, num_workers = 1)
# low, high = next(iter(train_data))
# print(low.shape)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(low.squeeze(), cmap="gray")
# ax[0].set_title('low_dose')
# ax[1].imshow(high.squeeze(), cmap="gray")
# ax[1].set_title('high_dose')
# plt.show()

# 256*256 so 9 residual blocks 
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        res_block = [nn.ReflectionPad2d(1),
                    nn.Conv2d(256, 256, 3),
                    nn.InstanceNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(256, 256, 3),
                    nn.InstanceNorm2d(256)  ]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture : c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,
        # R256,R256,R256,u128, u64,c7s1-3
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(1, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)]
        model += [nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)]
        model += [nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)]
        for _ in range(9):
            model += [ResidualBlock()]
        model += [nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True) ]
        model += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True) ]
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 1, 7),
                    nn.Sigmoid() ]
        self.model = nn.Sequential(*model)
        self.apply(self.init_params)
    def forward(self, z):
        return self.model(z)
    def init_params(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, 1, 0.02)
            nn.init.constant(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Receptive field : (output_size-1)*stride + kernel_size
        # 70*70 patchgan for discriminator
        model = [nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, True)]
                # Receptive field : 70
        model += [nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, True)]
                # Receptive field : 34
        model += [nn.Conv2d(128, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True)]
                # Receptive field : 16
        model += [nn.Conv2d(256, 512, 4, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, True)]
                # Receptive field : 7
        model += [nn.Conv2d(512, 1, 4, padding=1),
                nn.AvgPool2d(30)]
                # Receptive field : 4
        self.model = nn.Sequential(*model)
        self.apply(self.init_params)
    def forward(self, x):
        output = self.model(x)
        return output.view(output.size()[0], 1)
    def init_params(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif type(m) == nn.BatchNorm2d:
            nn.init.normal_(m.weight.data, 1, 0.02)
            nn.init.constant(m.bias.data, 0)

def lr_decay(epoch):
    lr = 1 - max(0, epoch - 100)/200
    return lr
def psnr_accuracy(x, y):
    batch_size = x.shape[0]
    accuracy = 0
    for i in range(batch_size):
        accuracy += psnr(x[i], y[i], data_range=1)
    return accuracy/batch_size


G_low_high = Generator().to(device)
G_high_low = Generator().to(device)
D_low = Discriminator().to(device)
D_high = Discriminator().to(device)
gen_optimizer = torch.optim.Adam(itertools.chain(G_low_high.parameters(),G_high_low.parameters()), lr=opt.lr)
dis_low_optimizer = torch.optim.Adam(D_low.parameters(), lr=opt.lr)
dis_high_optimizer = torch.optim.Adam(D_high.parameters(), lr=opt.lr)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=lr_decay)
lr_scheduler_D_low = torch.optim.lr_scheduler.LambdaLR(dis_low_optimizer, lr_lambda=lr_decay)
lr_scheduler_D_high = torch.optim.lr_scheduler.LambdaLR(dis_high_optimizer, lr_lambda=lr_decay)
summary = SummaryWriter('logs/')


if opt.resume:
    ckpt = opt.ckpt / ('%s.pt' % opt.ckpt_reload)
    try:
        checkpoint = torch.load(ckpt)
        G_low_high.load_state_dict(checkpoint['G_low_high'])
        G_high_low.load_state_dict(checkpoint['G_high_low'])
        D_low.load_state_dict(checkpoint['D_low'])
        D_high.load_state_dict(checkpoint['D_high'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_low_optimizer.load_state_dict(checkpoint['dis_low_optimizer'])
        dis_high_optimizer.load_state_dict(checkpoint['dis_high_optimizer'])
        epoch = checkpoint['epoch'] + 1
    except Exception as e:
        print(e)
if not os.path.exists(opt.result_save_dir):
    os.makedirs(opt.result_save_dir)
if not os.path.exists(opt.ckpt_dir):
    os.makedirs(opt.ckpt_dir)
if not os.path.exists(opt.test_save_dir):
    os.makedirs(opt.test_save_dir)
if not os.path.exists(opt.best_save_dir):
    os.makedirs(opt.best_save_dir)

ones = torch.ones((opt.batch_size, 1)).to(device)
zeros = torch.zeros((opt.batch_size, 1)).to(device)
best_accuracy = 0
criterion_identity = nn.L1Loss()
criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()
epoch = 0
best_epoch = 0
for epoch in range(epoch, opt.epoch):
    start_time = time.time()
    G_low_high.train()
    G_high_low.train()
    D_low.train()
    D_high.train()
    for i, (low, high) in enumerate(train_data):
        low = low.to(device)
        high = high.to(device)
        # train generator
        gen_optimizer.zero_grad()
        # identity loss : G_low_high(high), high (L1)
        loss_identity_low = criterion_identity(G_high_low(low),low)*5
        loss_identity_high = criterion_identity(G_low_high(high), high)*5
        # GAN loss : D_high(G_low_high(low)), 1 (MSE)
        fake_high = G_low_high(low)
        fake_low = G_high_low(high)
        loss_gan_low = criterion_gan(D_high(fake_high), ones)
        loss_gan_high = criterion_gan(D_low(fake_low), ones)
        # cycle loss : G_low_high(low), low (L1)
        loss_cycle_low = criterion_cycle(G_high_low(fake_high), low)*10
        loss_cycle_high = criterion_cycle(G_low_high(fake_low), high)*10
        loss_G = loss_identity_high + loss_identity_low + loss_gan_high + loss_gan_low + loss_cycle_high + loss_cycle_low
        loss_G.backward()
        gen_optimizer.step()

        # train discriminator_low
        dis_low_optimizer.zero_grad()
        fake_low_output = D_low(fake_low.detach())
        low_fake_loss = criterion_gan(fake_low_output, zeros)

        real_low_output = D_low(low)
        low_real_loss = criterion_gan(real_low_output, ones)

        loss_D_low = (low_fake_loss + low_real_loss)/2
        loss_D_low.backward()
        dis_low_optimizer.step()
        # train discriminator_high
        dis_high_optimizer.zero_grad()
        fake_high_output = D_high(fake_high.detach())
        high_fake_loss = criterion_gan(fake_high_output, zeros)

        real_high_output = D_high(high)
        high_real_loss = criterion_gan(real_high_output, ones)

        loss_D_high = (high_fake_loss + high_real_loss)/2
        loss_D_high.backward()
        dis_high_optimizer.step()
        accuracy = psnr_accuracy(high.cpu().detach().numpy(), fake_high.cpu().detach().numpy())
        
    if epoch == 0:
        fake_high = fake_high.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        fake_low = fake_low.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        real_high = high.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        real_low = low.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        save_image(fake_high, "./fake_high.png", nrow=4)
        save_image(fake_low, "./fake_low.png", nrow=4)
        save_image(real_high, "./real_high.png", nrow=4)
        save_image(real_low, "./real_low.png", nrow=4)
    if (epoch+1) % 10 == 0:
        fake_high = fake_high.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        fake_low = fake_low.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        save_image(fake_high, os.path.join(opt.result_save_dir, f"{epoch}_high.png"), nrow=4)
        save_image(fake_low, os.path.join(opt.result_save_dir, f"{epoch}_low.png"), nrow=4)
    t = time.time()-start_time
    lr_scheduler_G.step()
    lr_scheduler_D_low.step()
    lr_scheduler_D_high.step()
    print(f'Train: Epoch {epoch} || discriminator low loss={loss_D_low:.4f} || discriminator high loss={loss_D_high:.4f} || generator loss={loss_G:.4f} || accuracy = {accuracy:.3f} || time {t:.3f}')
    summary.add_scalar("dis_low", loss_D_low, epoch)
    summary.add_scalar("dis_high", loss_D_high, epoch)
    summary.add_scalar("gen", loss_G, epoch)

    G_low_high.eval()
    G_high_low.eval()
    D_low.eval()
    D_high.eval()
    start_time = time.time()
    with torch.no_grad():
        test_accuracy = 0
        test_num = 0
        for i, (test_low, test_high) in enumerate(test_data):
            low = test_low.to(device)
            high = test_high.to(device)
            # identity loss
            loss_identity_low = criterion_identity(G_high_low(low),low)*5
            loss_identity_high = criterion_identity(G_low_high(high), high)*5
            # GAN loss 
            fake_high = G_low_high(low)
            fake_low = G_high_low(high)
            loss_gan_low = criterion_gan(D_high(fake_high), ones)
            loss_gan_high = criterion_gan(D_low(fake_low), ones)
            # cycle loss
            loss_cycle_low = criterion_cycle(G_high_low(fake_high), low)*10
            loss_cycle_high = criterion_cycle(G_low_high(fake_low), high)*10
            loss_G = loss_identity_high + loss_identity_low + loss_gan_high + loss_gan_low + loss_cycle_high + loss_cycle_low
            # discriminator_low
            fake_low_output = D_low(fake_high)
            low_fake_loss = criterion_gan(fake_low_output, zeros)

            real_low_output = D_low(low)
            low_real_loss = criterion_gan(real_low_output, ones)

            loss_D_low = (low_fake_loss + low_real_loss)/2
            # discriminator_high
            fake_high_output = D_high(fake_low)
            high_fake_loss = criterion_gan(fake_high_output, zeros)

            real_high_output = D_high(high)
            high_real_loss = criterion_gan(real_high_output, ones)

            loss_D_high = (high_fake_loss + high_real_loss)/2  
            test_accuracy += psnr_accuracy(low.cpu().detach().numpy(), fake_high.cpu().detach().numpy())  
            test_num += low.shape[0]
        
        test_accuracy /= test_num
        if best_accuracy < test_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            fake_high = fake_high.view(opt.batch_size, 1, opt.img_size,opt.img_size)
            low = low.view(opt.batch_size, 1, opt.img_size,opt.img_size)
            psnr_residual = psnr(low.cpu().detach().numpy(), fake_high.cpu().detach().numpy(), data_range=1)
            residual = (low - fake_high).view(opt.batch_size, 1, opt.img_size, opt.img_size)
            save_image(low, os.path.join(opt.best_save_dir, "best_low.png"), nrow=4)
            save_image(fake_low, os.path.join(opt.best_save_dir, "best_fake_low.png"), nrow=4)
            save_image(residual, os.path.join(opt.best_save_dir, "residual.png"), nrow = 4)
    # if (epoch+1) % 10 == 0:
    #     fake_high = fake_high.view(opt.batch_size, 1, opt.img_size,opt.img_size)
    #     high = high.view(opt.batch_size, 1, opt.img_size,opt.img_size)
    #     save_image(high, os.path.join(opt.test_save_dir, f"{epoch}_high.png"), nrow=4)
    #     save_image(fake_high, os.path.join(opt.test_save_dir, f"{epoch}_fake_high.png"), nrow=4)
    t = time.time() - start_time
    print(f'Test : Epoch {epoch} || discriminator low loss={loss_D_low:.4f} || discriminator high loss={loss_D_high:.4f} || generator loss={loss_G:.4f} || accuracy = {accuracy:.3f} || best_accuracy = {best_accuracy:.3f} || best_residual = {psnr_residual:.3f} || time {t:.3f}')        
    torch.save(dict(epoch = epoch, G_low_high = G_low_high.state_dict(), G_high_low = G_high_low.state_dict(), D_low = D_low.state_dict(), D_high = D_high.state_dict(), 
                gen_optimizer = gen_optimizer.state_dict(), dis_low_optimizer = dis_low_optimizer.state_dict(), dis_high_optimizer = dis_high_optimizer.state_dict()), str(opt.ckpt_dir)+'/'+str(epoch)+'.pt')
summary.close()

# evaluation metric
# see whether low - G_low_high(low) result in noise
# see whether psnr(low, G_low_high(low)) ~= 4dB

# question
# how to get fully denoised image with cropped generator?
# how to average psnr of cropped image while parameter is updated every epoch?


        
