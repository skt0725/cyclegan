import glob
import os
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import itertools
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
opt.resume = False
opt.ckpt_reload = 0
opt.data_dir = "./data/Low_High_CT_mat_slice"
opt.result_save_dir = "./result"
opt.lr = 0.0002
opt.epoch = 150
opt.batch_size = 16
opt.img_size = 128
opt.loss_lambda = 10

# data load & preprocess
img_path = glob.glob(os.path.join(opt.data_dir, '**/*.mat'), recursive=True)
low = []
high = []
for path in img_path:
    img_mat = io.loadmat(path)
    img_low_high = img_mat.get('imdb')
    low.append(img_low_high[0][0][0])
    high.append(img_low_high[0][0][1])
class AAPM(Dataset):
    def __init__(self, transform = None):
        self.len = len(low)
        self.low = low
        self.high = high
        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.transform is not None:
            low = self.transform(self.low[idx])
            high = self.transform(self.high[idx])
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
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
])

dataset = AAPM(transform = transform)
dataloader = DataLoader(dataset=dataset, batch_size = opt.batch_size, shuffle=True, num_workers = 1, drop_last = True)

# low, high = next(iter(dataloader))
# print(low.shape)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(low.squeeze(), cmap="gray")
# ax[0].set_title('low_dose')
# ax[1].imshow(high.squeeze(), cmap="gray")
# ax[1].set_title('high_dose')
# plt.show()

# 128*128 so 6 residual blocks 
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
        return x + self.resblock(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture : c7s1-64,d128,d256,R256,R256,R256,
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
        for _ in range(6):
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
            nn.init.normal(m.weight.data, 0, 0.02)
        elif type(m) == nn.BatchNorm2d:
            nn.init.normal(m.weight.data, 1, 0.02)
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
                nn.AvgPool2d(14)]
                # Receptive field : 4
        self.model = nn.Sequential(*model)
        self.apply(self.init_params)
    def forward(self, x):
        output = self.model(x)
        return output.view(output.size()[0], 1)
    def init_params(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal(m.weight.data, 0, 0.02)
        elif type(m) == nn.BatchNorm2d:
            nn.init.normal(m.weight.data, 1, 0.02)
            nn.init.constant(m.bias.data, 0)


G_low_high = Generator().to(device)
G_high_low = Generator().to(device)
D_low = Discriminator().to(device)
D_high = Discriminator().to(device)
gen_optimizer = torch.optim.Adam(itertools.chain(G_low_high.parameters(),G_high_low.parameters()), lr=opt.lr)
dis_low_optimizer = torch.optim.Adam(D_low.parameters(), lr=opt.lr)
dis_high_optimizer = torch.optim.Adam(D_high.parameters(), lr=opt.lr)
summary = SummaryWriter('logs/')
epoch = 0

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
ones = torch.ones((opt.batch_size, 1)).to(device)
zeros = torch.zeros((opt.batch_size, 1)).to(device)
criterion_identity = nn.L1Loss()
criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()
for epoch in range(epoch, opt.n_epochs):
    start_time = time()
    for i, (low, high) in enumerate(dataloader):
        low = low.to(device)
        high = high.to(device)
        # train generator
        gen_optimizer.zero_grad()
        # identity loss : G_low_high(high), high (L1)
        loss_identity_low = criterion_identity(G_high_low(low),low)
        loss_identity_high = criterion_identity(G_low_high(high), high)
        # GAN loss : D_high(G_low_high(low)), 1 (MSE)
        fake_high = G_low_high(low)
        fake_low = G_high_low(high)
        loss_gan_low = criterion_gan(D_high(fake_high), ones)
        loss_gan_high = criterion_gan(D_low(fake_low), ones)
        # cycle loss : G_low_high(low), low (L1)
        loss_cycle_low = criterion_cycle(G_high_low(fake_high), low)
        loss_cycle_high = criterion_cycle(G_low_high(fake_low), high)
        loss_G = loss_identity_high + loss_identity_low + loss_gan_high + loss_gan_low + loss_cycle_high + loss_cycle_low
        loss_G.backward()
        gen_optimizer.step()

        # train discriminator_low
        dis_low_optimizer.zero_grad()
        fake_low_output = D_low(fake_high.detach())
        low_fake_loss = criterion_gan(fake_low_output, zeros)

        real_low_output = D_low(low)
        low_real_loss = criterion_gan(real_low_output, ones)

        loss_D_low = (low_fake_loss + low_real_loss)/2
        loss_D_low.backward()
        dis_low_optimizer.step()
        # train discriminator_high
        dis_high_optimizer.zero_grad()
        fake_high_output = D_high(fake_low.detach())
        high_fake_loss = criterion_gan(fake_high_output, zeros)

        real_high_output = D_high(high)
        high_real_loss = criterion_gan(real_high_output, ones)

        loss_D_high = (high_fake_loss + high_real_loss)/2
        loss_D_high.backward()
        dis_high_optimizer.step()

    if not os.path.exists(opt.result_save_dir):
        os.makedirs(opt.result_save_dir)
    if epoch == 0:
        fake_high = fake_high.view(opt.batch_size, 1, 256,256)
        fake_low = fake_low.view(opt.batch_size, 1, 256,256)
        real_high = high.view(opt.batch_size, 1, 256,256)
        real_low = low.view(opt.batch_size, 1, 256,256)
        save_image(fake_high, "./fake_high.png", nrow=4)
        save_image(fake_low, "./fake_low.png", nrow=4)
        save_image(real_high, "./real_high.png", nrow=4)
        save_image(real_low, "./real_low.png", nrow=4)
    if (epoch+1) % 5 == 0:
        fake_high = fake_high.view(opt.batch_size, 1, 256,256)
        fake_low = fake_low.view(opt.batch_size, 1, 256,256)
        save_image(fake_high, os.path.join(opt.result_save_dir, f"{epoch}_high.png"), nrow=4)
        save_image(fake_low, os.path.join(opt.result_save_dir, f"{epoch}_low.png"), nrow=4)

    torch.save(dict(epoch = epoch, G_low_high = G_low_high.state_dict(), G_high_low = G_high_low.state_dict(), D_low = D_low.state_dict(), D_low = D_low.state_dict(), D_high = D_high.state_dict(), 
                gen_optimizer = gen_optimizer.state_dict(), dis_low_optimizer = dis_low_optimizer.state_dict(), dis_high_optimizer = dis_high_optimizer.state_dict()), str(opt.ckpt_dir)+'/'+str(epoch)+'.pt')
    t = time()-start_time
    print(f'Epoch {epoch}/{opt.n_epochs} || discriminator low loss={loss_D_low:.4f} || discriminator high loss={loss_D_high:.4f} || generator loss={loss_G:.4f} || time {t:.3f}')
    summary.add_scalar("dis_low", loss_D_low, epoch)
    summary.add_scalar("dis_high", loss_D_high, epoch)
    summary.add_scalar("gen", loss_G, epoch)
summary.close()




        
