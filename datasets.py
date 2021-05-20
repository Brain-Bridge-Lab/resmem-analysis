
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
from csv import reader
from torch import nn
from torchvision.transforms.transforms import CenterCrop
import glob

memnet_transform = transforms.Compose((
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    transforms.ToTensor()
    )
)


class FacesDataset(Dataset):
    def __init__(self, loc='./Sources/faces/'):
        self.loc = loc
        self.faces_frame = np.array(np.loadtxt(f'{loc}face_scores.csv', delimiter=',', dtype=str, skiprows=1))

    def __len__(self):
        return len(self.faces_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.faces_frame[idx, 0]
        image = Image.open(f'{self.loc}images/{img_name}')
        y = self.faces_frame[idx, 1]
        y = torch.Tensor([float(y)])
        image_x = memnet_transform(image)
        return [image_x, y, img_name]


class ThingsDataset(Dataset):
    def __init__(self, loc='./Sources/things/'):
        self.loc = loc
        self.things_frame = np.array(np.loadtxt(f'{loc}object-memorabilities.csv', delimiter=',', dtype=str))

    def __len__(self):
        return len(self.things_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.things_frame[idx, 1]
        image = Image.open(f'{self.loc}{img_name}').convert('RGB')
        y = self.things_frame[idx, 2]
        y = torch.Tensor([float(y)])
        image_x = memnet_transform(image)
        return [image_x, y]


class MemCatDataset(Dataset):
    def __init__(self, loc='./Sources/memcat/', transform=memnet_transform):
        self.loc = loc
        self.transform = transform
        with open(f'{loc}data/memcat_image_data.csv', 'r') as file:
            r = reader(file)
            next(r)
            data = [d for d in r]
            self.memcat_frame = np.array(data)

    def __len__(self):
        return len(self.memcat_frame)

    def ys(self):
        for y in self.memcat_frame[:, 12]:
            yield torch.Tensor([float(y)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.memcat_frame[idx, 1]
        cat = self.memcat_frame[idx, 2]
        scat = self.memcat_frame[idx, 3]
        img = Image.open(f'{self.loc}images/{cat}/{scat}/{img_name}').convert('RGB')
        y = self.memcat_frame[idx, 12]
        y = torch.Tensor([float(y)])
        image_x = self.transform(img)
        return [image_x, y, f'{self.loc}images/{cat}/{scat}/{img_name}']


class LamemDataset(Dataset):
    def __init__(self, loc='./Sources/lamem/', transform=memnet_transform):
        self.lamem_frame = np.array(np.loadtxt(f'{loc}splits/full.txt', delimiter=' ', dtype=str))
        self.loc = loc
        self.transform = transform

    def __len__(self):
        return self.lamem_frame.shape[0]

    def ys(self):
        for y in self.lamem_frame[:, 1]:
            yield torch.Tensor([float(y)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.lamem_frame[idx, 0]
        image = Image.open(f'{self.loc}/images/{img_name}')
        image = image.convert('RGB')
        y = self.lamem_frame[idx, 1]
        y = torch.Tensor([float(y)])
        image_x = self.transform(image)
        return [image_x, y, f'{self.loc}/images/{img_name}']

class EmoMem(Dataset):
    def __init__(self, loc='./Sources/emomem'):
        self.emomem_frame = np.array(glob.glob(f'{loc}/images/*'))
        self.loc = loc

    def __len__(self):
        return self.emomem_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.emomem_frame[idx]
        image = Image.open(img_name)
        image = image.convert('RGB')
        image_x = memnet_transform(image)
        return [image_x, [], img_name]