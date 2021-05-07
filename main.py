
from resmem import ResMem, transformer
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor, Normalize
from PIL import Image, ImageFilter
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import math


def get_gaussian_kernel(kernel_size=3, sigma=1, channels=3):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float().cuda()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) /
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_kernel.cuda()

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)
    gaussian_filter.cuda()
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = object()

    def hook_fn(self, module, i, output):
        self.features = output.clone().requires_grad_(True).cuda()

    def close(self):
        self.hook.remove()


class Viz:
    def __init__(self, size=67, upscaling_steps=16, upscaling_factor=1.2,
                 branch='alex', rn_address=''):
        self.branch = branch
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = ResMem(pretrained=True).cuda().eval()
        self.target = self.model
        if self.branch == 'resnet':
            assert rn_address
            self.rn_address = rn_address
            # ResNet 151 is defined fractally (wouldn't you? it's huge), so we have to make a little
            # addressing system for now.
            address = rn_address.split('-')
            top = int(address[0])
            second = int(address[1])
            self.target = list(list(self.model.features.children())[top].children())[second]

        self.output = None
        self.normer = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def visualize(self, layer, filt, lr=1e-2, opt_steps=36):
        sz = self.size
        img = Image.fromarray(np.uint8(np.random.uniform(150, 180, (sz, sz, 3))))
        activations = SaveFeatures(list(self.target.children())[layer])
        gaussian_filter = get_gaussian_kernel()
        self.model.zero_grad()
        for outer in tqdm(range(self.upscaling_steps), leave=False):
            img_var = torch.unsqueeze(ToTensor()(img), 0).cuda().requires_grad_(True)
            img_var.requires_grad_(True).cuda()
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

            pbar = tqdm(range(opt_steps), leave=False)
            for n in pbar:
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filt].mean() + 0.00*torch.norm(img_var)
                loss.backward()
                pbar.set_description(f'Loss: {loss.item()}')
                optimizer.step()

            sz = int(sz * self.upscaling_factor)
            img = ToPILImage()(img_var.squeeze(0))
            if outer != self.upscaling_steps:
                img = img.resize((sz, sz))
                img = img.filter(ImageFilter.BoxBlur(2))
            self.output = img.copy()
        self.save(layer, filt)
        activations.close()

    def save(self, layer, filt):
        if self.branch == 'alex':
            self.output.save(f'alex/layer_{layer}_filter_{filt}.jpg', 'JPEG')
        else:
            self.output.save(f'resnet/layer_{self.rn_address}-{layer}_filter_{filt}.jpg', 'JPEG')


if __name__ == '__main__':
    vis = Viz(branch='resnet', rn_address='4-1')
    i = 0
    while True:
        try:
            tqdm.write(f'Layer: {0}. Filter: {i} \r')
            vis.visualize(0, i)
            i += 1
        except IndexError:
            break
