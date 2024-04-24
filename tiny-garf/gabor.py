import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import PIL.Image
import matplotlib.pyplot as plt
import tqdm

# for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Data

class Image(Dataset):
    def __init__(self, image_path, H=100, W=100):
        super().__init__()
        self.load_image(H=H, W=W)

    def load_image(self, H=100, W=100):
        image_raw = PIL.Image.open('swan.jpg')
        tmp =  PIL.Image.new("RGB", image_raw.size, (255, 255, 255))
        if image_raw.mode == 'RGBA':
            r, g, b, a = image_raw.split()
            tmp.paste(image_raw, (0, 0), mask=a)
        else:
            tmp.paste(image_raw, (0, 0))

        # tmp.paste(image_raw, (0, 0), image_raw)
        image_raw = tmp
        # augment the image
        transform =  Compose([Resize([H, W]), ToTensor()])
        self.image_raw = transform(image_raw).to(device)
        self.H = self.image_raw.shape[1]
        self.W = self.image_raw.shape[2]
        self.channel = 3
        self.coords = self.get_coords(self.H, self.W)
        self.labels = self.image_raw.view(1, 3, self.H*self.W).permute(0,2,1)

    def get_coords(self, H, W):
        y_range = ((torch.arange(H,dtype=torch.float32,device=device))/H*2-1)*(H/max(H,W))
        x_range = ((torch.arange(W,dtype=torch.float32,device=device))/W*2-1)*(W/max(H,W))
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
        xy_grid = xy_grid.repeat(1,1,1) # [B,HW,2]
        return xy_grid

    def __len__(self):
        return self.coords.shape[1]

    def __getitem__(self,idx):
        return self.coords[0, idx], self.labels[0, idx]

H = 100
W = 100
y_range = ((torch.arange(H,dtype=torch.float32,device=device))/H*2-1)*(H/max(H,W))
x_range = ((torch.arange(W,dtype=torch.float32,device=device))/W*2-1)*(W/max(H,W))
Y,X = torch.meshgrid(y_range,x_range) # [H,W]
xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
xy_grid = xy_grid.repeat(1,1,1) # [B,HW,2]
print(xy_grid.shape)

# Models
class ComplexGaborLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = torch.nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = torch.nn.Parameter(self.scale_0*torch.ones(1), trainable)

        self.linear = torch.nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j*omega - scale.abs().square())

# Gabor Model

class ComplexImageFunction(torch.nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        # self.hidden_features = int(hidden_features/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'
        self.define_network()

    def define_network(self):
        self.mlp = []

        self.mlp.append(ComplexGaborLayer(self.in_features, self.hidden_features,omega0=30,sigma0=10,is_first=True,))
        for i in range(self.hidden_layers-1):
            self.mlp.append(ComplexGaborLayer(self.hidden_features, self.hidden_features,omega0=30,sigma0=10))

        self.mlp.append(torch.nn.Linear(self.hidden_features, self.out_features,dtype=torch.cfloat))
        self.mlp = torch.nn.Sequential(*self.mlp)

    def forward(self, input):
        rgb = self.mlp(input)
        if self.wavelet == 'gabor':
            return rgb.real
        return rgb

model = ComplexImageFunction(in_features=2, out_features=3, hidden_features=256, hidden_layers=4)
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)

# Define MSELoss
criterion = torch.nn.MSELoss()


# Visualise initial state
model.eval()
data = Image("swan.jpg")
pred_rgb = model(data.coords)
plt.imshow(pred_rgb[0].reshape(H, W, 3).detach().cpu().numpy())
plt.show()

num_epoch = 200
val_freq = 20
model.train()

train_psnrs_gabor = []
test_psnrs_gabor = []

trainloader = DataLoader(data, batch_size=512, shuffle=True)
progress_loader = tqdm.trange(num_epoch, desc="training", leave=False)


for i in progress_loader:
    for j, (input, gt) in enumerate(trainloader):

        optimizer.zero_grad()
        pred_rgb = model(input)
        loss = criterion(pred_rgb, gt)

        train_psnr = -10 * loss.log10()
        loss.backward()
        optimizer.step()
        progress_loader.set_postfix(it=i,psnr="{:.4f}".format(train_psnr))

    train_psnrs_gabor.append(train_psnr)

    if i % val_freq == 0 and i > 0:
        with torch.no_grad():
            val_rgb = model(data.coords)
            loss = criterion(val_rgb, data.labels)
            psnr = -10 * loss.log10()
            test_psnrs_gabor.append(psnr)
            print("Epoch {}.....Test PSNR {}".format(i, psnr))
            plt.imshow(val_rgb.view(H, W, 3).detach().cpu().numpy())
            plt.show()