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
class GaussianLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma=0.05):
        super().__init__()
        self.sigma = sigma
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        return self.gaussian(self.linear(input))

    def gaussian(self, input):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        k1 = (-0.5*(input)**2/self.sigma**2).exp()
        return k1

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

class ReLULayer(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma=0.05):
        super().__init__()
        self.sigma = sigma
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(self.linear(input))

class GeLULayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # self.sigma = sigma
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.GELU()

    def forward(self, input):
        return self.relu(self.linear(input))

class Sin(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)

class Cos(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super(Cos, self).__init__()

    def forward(self, input):
        return torch.cos(input)

class consandsinLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.sin = Sin()
        self.cos = Cos()
        self.a = torch.nn.Parameter(torch.ones(1))
        self.b = torch.nn.Parameter(torch.ones(1))


    def forward(self, input):
      x = self.linear(input)
      # print("a："+str(self.a))
      # print("b："+str(self.b))
      return self.a*self.sin(x)+self.b*self.cos(x)

# Gaussian Model
# Define Gaussian Model
class NeuralGaussianImageFunction(torch.nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=4, sigma=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.sigma = sigma
        self.hidden_layers = hidden_layers
        self.define_network()


    def define_network(self):
        self.mlp = []

        self.mlp.append(GaussianLayer(self.in_features, self.hidden_features, sigma=self.sigma))
        for i in range(self.hidden_layers-1):
            self.mlp.append(GaussianLayer(self.hidden_features, self.hidden_features, sigma=self.sigma))

        self.mlp.append(torch.nn.Linear(self.hidden_features, self.out_features))
        self.mlp = torch.nn.Sequential(*self.mlp)

    def forward(self, input):
        rgb = self.mlp(input)
        return rgb

# Init model
model = NeuralGaussianImageFunction(in_features=2, out_features=3, hidden_features=256, hidden_layers=4, sigma=0.05)
model
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)

# Define MSELoss
criterion = torch.nn.MSELoss()


# Visualise initial state
model.eval()
data = Image("swan.jpg")
pred_rgb = model(data.get_coords(H, W))
plt.imshow(pred_rgb[0].reshape(H, W, 3).detach().cpu().numpy())
plt.show()

# Setup dataloader
num_epoch = 200
val_freq = 20
model.train()

train_psnrs_Gaussian = []
test_psnrs_Gaussian = []

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

    train_psnrs_Gaussian.append(train_psnr)

    if i % val_freq == 0 and i > 0:
        with torch.no_grad():
            val_rgb = model(data.coords)
            loss = criterion(val_rgb, data.labels)
            psnr = -10 * loss.log10()
            test_psnrs_Gaussian.append(psnr)
            print("Epoch {}.....Test PSNR {}".format(i, psnr))
            plt.imshow(val_rgb.view(H,W,3).detach().cpu().numpy())
            plt.show()

# sin Model
# Define sin Model
class NeuralSinAndCosImageFunction(torch.nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.define_network()


    def define_network(self):
        self.mlp = []

        self.mlp.append(consandsinLayer(self.in_features, self.hidden_features))
        for i in range(self.hidden_layers-1):
            self.mlp.append(consandsinLayer(self.hidden_features, self.hidden_features))

        self.mlp.append(torch.nn.Linear(self.hidden_features, self.out_features))
        self.mlp = torch.nn.Sequential(*self.mlp)

    def forward(self, input):
        rgb = self.mlp(input)
        return rgb

model = NeuralSinAndCosImageFunction(in_features=2, out_features=3, hidden_features=256, hidden_layers=4)
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

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

train_psnrs_sin = []
test_psnrs_sin = []

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

    train_psnrs_sin.append(train_psnr)

    if i % val_freq == 0 and i > 0:
        with torch.no_grad():
            val_rgb = model(data.coords)
            loss = criterion(val_rgb, data.labels)
            psnr = -10 * loss.log10()
            test_psnrs_sin.append(psnr)
            print("Epoch {}.....Test PSNR {}".format(i, psnr))
            plt.imshow(val_rgb.view(H, W, 3).detach().cpu().numpy())
            plt.show()

# ReLU Model
# Define ReLU Model
class NeuralReLUImageFunction(torch.nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=4, sigma=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.sigma = sigma
        self.hidden_layers = hidden_layers
        self.define_network()


    def define_network(self):
        self.mlp = []

        self.mlp.append(ReLULayer(self.in_features, self.hidden_features, sigma=self.sigma))
        for i in range(self.hidden_layers-1):
            self.mlp.append(ReLULayer(self.hidden_features, self.hidden_features, sigma=self.sigma))

        self.mlp.append(torch.nn.Linear(self.hidden_features, self.out_features))
        self.mlp = torch.nn.Sequential(*self.mlp)

    def forward(self, input):
        rgb = self.mlp(input)
        return rgb

# Init model
model = NeuralReLUImageFunction(in_features=2, out_features=3, hidden_features=256, hidden_layers=4, sigma=0.05)
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

# Define MSELoss
criterion = torch.nn.MSELoss()


# Visualise initial state
model.eval()
data = Image("swan.jpg")
pred_rgb = model(data.coords)
plt.imshow(pred_rgb[0].reshape(H, W, 3).detach().cpu().numpy())
plt.show()

# Setup dataloader
num_epoch = 200
val_freq = 20
model.train()

train_psnrs_ReLU = []
test_psnrs_ReLU = []

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

    train_psnrs_ReLU.append(train_psnr)

    if i % val_freq == 0 and i > 0:
        with torch.no_grad():
            val_rgb = model(data.coords)
            loss = criterion(val_rgb, data.labels)
            psnr = -10 * loss.log10()
            test_psnrs_ReLU.append(psnr)
            print("Epoch {}.....Test PSNR {}".format(i, psnr))
            plt.imshow(val_rgb.view(H, W, 3).detach().cpu().numpy())
            plt.show()

# GELU Model
# Define GeLU Model
class GeluImageFunction(torch.nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features=256, hidden_layers=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.define_network()


    def define_network(self):
        self.mlp = []

        self.mlp.append(GeLULayer(self.in_features, self.hidden_features))
        for i in range(self.hidden_layers-1):
            self.mlp.append(GeLULayer(self.hidden_features, self.hidden_features))

        self.mlp.append(torch.nn.Linear(self.hidden_features, self.out_features))
        self.mlp = torch.nn.Sequential(*self.mlp)

    def forward(self, input):
        rgb = self.mlp(input)
        return rgb

model = GeluImageFunction(in_features=2, out_features=3, hidden_features=256, hidden_layers=4)
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

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

train_psnrs_gelu = []
test_psnrs_gelu = []

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

    train_psnrs_gelu.append(train_psnr)

    if i % val_freq == 0 and i > 0:
        with torch.no_grad():
            val_rgb = model(data.coords)
            loss = criterion(val_rgb, data.labels)
            psnr = -10 * loss.log10()
            test_psnrs_gelu.append(psnr)
            print("Epoch {}.....Test PSNR {}".format(i, psnr))
            plt.imshow(val_rgb.view(H, W, 3).detach().cpu().numpy())
            plt.show()

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