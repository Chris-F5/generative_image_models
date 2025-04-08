import torch
import torchvision
import math
import PIL.Image
import numpy as np
import os


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = [
            layer for c1, c2 in zip(channels[:-2], channels[1:-1]) for layer in [
                torch.nn.Conv2d(c1, c2, 3, padding=1),
                torch.nn.BatchNorm2d(c2),
                torch.nn.LeakyReLU(0.2)
            ]
        ] + [
            torch.nn.Conv2d(channels[-2], channels[-1], 3, padding=1),
            torch.nn.Tanh()
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3:
            return self.forward(x[None, :, :, :])[0]
        return self.net(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = [
            layer for c1, c2 in zip(channels[:-2], channels[1:-1]) for layer in [
                torch.nn.Conv2d(c1, c2, 3, padding=1),
                torch.nn.BatchNorm2d(c2),
                torch.nn.LeakyReLU(0.2)
            ]
        ] + [
            torch.nn.Conv2d(channels[-2], channels[-1], 3, padding=1),
            torch.nn.Sigmoid()
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3:
            return self.forward(x[None, :, :, :])[0]
        return self.net(x)


class GeneratorStack(torch.nn.Module):
    def __init__(self, device, sizes):
        super().__init__()
        num_kernels = [32 * 2**(i//4) for i in range(len(sizes))]
        self.generators = torch.nn.ModuleList([
            Generator([3] + [kernels]*4 + [3])
            for kernels in num_kernels
        ])
        self.sizes = sizes
        self.device = device
        self.reconstruction_z = torch.randn((3, sizes[0][0], sizes[0][1]), device=device)
        self.reconstruction_zs = [self.reconstruction_z] + [
            torch.zeros((3, size[0], size[1]), device=device) for size in sizes[1:]
        ]
        #self.noise_amp = [1.0]*len(sizes)
        self.register_buffer('noise_amp', torch.tensor([1.0]*len(sizes)))
    def forward(self, zs, depth):
        x = self.generators[0](zs[0]*self.noise_amp[0])
        for i in range(1, depth+1):
            x = torchvision.transforms.Resize(self.sizes[i])(x)
            x = x.detach()
            x = self.generators[i](zs[i]*self.noise_amp[i]+x)+x
        return x

    def reconstruction(self, depth):
        return self.forward(self.reconstruction_zs, depth)

    def random(self, depth, variations=1):
        zs = [
            torch.randn((variations, 3, size[0], size[1]), device=self.device) for size in self.sizes
        ]
        out = self.forward(zs, depth)
        if variations == 1:
            return out[0]
        else:
            return out


def train_level(device, stack, reals, depth, iterations):
    num_kernels = 32 * 2**(depth//4)
    discriminator = Discriminator([3] + [num_kernels]*4 + [1]).to(device)
    discriminator.apply(init_weights)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9999))
    g_optimizer = torch.optim.Adam(stack.parameters(), lr=1e-4, betas=(0.5, 0.9999))
    #d_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=d_optimizer, step_size=400)
    #g_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=g_optimizer, step_size=400)
    discriminator.train()
    stack.train()

    if depth > 0:
        reconstruction = stack.reconstruction(depth - 1)
        reconstruction = torchvision.transforms.Resize(stack.sizes[depth])(reconstruction)
        stack.noise_amp[depth] = torch.sqrt(torch.nn.MSELoss()(reconstruction, reals[depth])).detach()

    for i in range(iterations):

        # Step discriminator.
        discriminator.zero_grad()
        d_optimizer.zero_grad()
        fakes = stack.random(depth, 16)
        x_pred = discriminator(reals[depth])
        z_pred = discriminator(fakes)
        d_loss = z_pred.mean() - x_pred.mean()
        d_loss.backward()
        d_optimizer.step()

        # Step Generator
        stack.zero_grad()
        g_optimizer.zero_grad()
        fakes = stack.random(depth, 16)
        assert fakes.shape[0] == 16
        assert fakes.shape[1:] == reals[depth].shape
        z_pred = discriminator(fakes)
        reconstruction = stack.reconstruction(depth)
        r_loss = torch.nn.MSELoss()(reconstruction, reals[depth]) * 10
        g_loss = -z_pred.mean()
        (g_loss + r_loss).backward()
        g_optimizer.step()

        #d_scheduler.step()
        #g_scheduler.step()

        print(f"{g_loss.item():>4f} {r_loss.item():>4f} {d_loss.item():>4f} {depth+1}/{len(sizes)} {stack.noise_amp[depth]:>4f} {i+1}")


def init_weights(layer):
    if hasattr(layer, 'weight'):
        torch.nn.init.normal_(layer.weight.data, std=0.02)
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'data'):
        torch.nn.init.constant_(layer.bias.data, 0.0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device = {device}")

real_img = PIL.Image.open("birds.png")
real_np = np.array(real_img)
real_np = np.transpose(real_np, (2, 0, 1))
real_np = (real_np / 255)*2.0 - 1.0
real_ = torch.tensor(real_np).float()[:3, :, :].to(device)
min_size = 15
sizes = [(real_.shape[1], real_.shape[2])]
while sizes[0][0] > min_size and sizes[0][1] > min_size:
    i = len(sizes)
    s = (int(sizes[-1][0] * 0.75**i), int(sizes[-1][1] * 0.75**i))
    sizes.insert(0, s)

#sizes = sizes[:2]

reals = [
    torchvision.transforms.Resize(size)(real_) for size in sizes
]
with torch.no_grad():
    for i, real in enumerate(reals):
        _real = real.cpu().numpy()
        _real = (_real + 1.0) / 2.0
        _real = np.transpose(_real, (1, 2, 0))
        _real_img = PIL.Image.fromarray((_real*255).astype(np.uint8))
        _real_img.save(f"real{i}.png")

stack = GeneratorStack(device, sizes).to(device)
if os.path.isfile('stack.pth'):
    stack.load_state_dict(torch.load('stack.pth', map_location=device))
else:
    stack.apply(init_weights)
    for d in range(len(sizes)):
        train_level(device, stack, reals, d, 5000)
    torch.save(stack.state_dict(), 'stack.pth')
stack.eval()

print(stack.noise_amp)

with torch.no_grad():
    for d in range(len(sizes)):
        reconstruction = stack.reconstruction(d)
        reconstruction_loss = torch.nn.MSELoss(reduction='sum')(reconstruction, reals[d])
        reconstruction = reconstruction.cpu().numpy()
        reconstruction = (reconstruction + 1.0) / 2.0
        reconstruction = np.transpose(reconstruction, (1, 2, 0))
        reconstruction = np.clip(reconstruction, 0.0, 1.0)
        reconstruction_img = PIL.Image.fromarray((reconstruction*255).astype(np.uint8))
        fname = f"reconstruction{d}.png"
        print(fname, np.min(reconstruction), np.max(reconstruction), reconstruction_loss.item())
        reconstruction_img.save(fname)

        for i in range(10):
            fake = stack.random(d).cpu().numpy()
            fake = (fake + 1.0) / 2.0
            fake = np.transpose(fake, (1, 2, 0))
            fake = np.clip(fake, 0.0, 1.0)
            fake_img = PIL.Image.fromarray((fake*255).astype(np.uint8))
            fname = f"fake{d}-{i}.png"
            print(fname, np.min(fake), np.max(fake))
            fake_img.save(fname)

