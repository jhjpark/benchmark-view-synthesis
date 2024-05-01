import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import PIL.Image
import nvtx

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
        image_raw = PIL.Image.open('../images/swan.jpg')
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

    @nvtx.annotate("iteration")
    def forward(self, input):
        rgb = self.mlp(input)
        return rgb

# Init model
model = NeuralGaussianImageFunction(in_features=2, out_features=3, hidden_features=256, hidden_layers=4, sigma=0.05)
model = model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)

# Define MSELoss
criterion = torch.nn.MSELoss()


# Visualise initial state
model.eval()
data = Image("../images/swan.jpg")
pred_rgb = model(data.get_coords(H, W))

# Setup dataloader
num_epoch = 4
val_freq = 20
model.train()

trainloader = DataLoader(data, batch_size=512, shuffle=True)

with torch.autograd.profiler.emit_nvtx():
    for j, (input, gt) in enumerate(trainloader):
        pred_rgb = model(input)
