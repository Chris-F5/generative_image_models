import torch
import torchvision
import math
import PIL.Image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--real', type=str)
parser.add_argument('--d_grad_clamp', type=float, default=0.1)
parser.add_argument('--res_loss_mult', type=float, default=0.2)
parser.add_argument('-i', '--iterations', type=int, default=4000)
parser.add_argument('-o', '--output_dir', type=str, default='.')
args = parser.parse_args()


# TODO:
# Dont require grads on fixed generators.
# Eval mode for fixed generators.
# Performance improvelemnt.
# Reconstruction image is differing due to differing batchnorm behaviour in eval mode.
# Remove BatchNorm?
# WGAN
#   gradient penalty


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
            # torch.nn.Sigmoid() # TODO: THIS ISNT NEEDED IN WGAN
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

    def disable_train(self):
        for g in self.generators:
            g.eval()
            for param in g.parameters():
                param.requires_grad = False

    def train_at_depth(self, depth):
        self.disable_train()
        self.generators[depth].train()
        for param in self.generators[depth].parameters():
            param.requires_grad = True
        return self.generators[depth].parameters()


# # Ripped from https://github.com/FriedRonaldo/SinGAN/blob/master/code/ops.py
# # TODO: read https://arxiv.org/abs/1701.07875
# def compute_grad_gp_wgan(device, D, x_real, x_fake):
#     alpha = torch.rand(x_real.size(0), 1, 1, 1, device=device)
# 
#     x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
#     x_interpolate.requires_grad = True
#     d_inter_logit = D(x_interpolate)
#     grad = torch.autograd.grad(d_inter_logit, x_interpolate,
#                                grad_outputs=torch.ones_like(d_inter_logit, device=device),
#                                create_graph=True)[0]
# 
#     norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
# 
#     d_gp = ((norm - 1) ** 2).mean()
#     return d_gp

def train_level(device, stack, reals, depth, iterations):
    num_kernels = 32 * 2**(depth//4)
    discriminator = Discriminator([3] + [num_kernels]*4 + [1]).to(device)
    discriminator.apply(init_weights)

    g_params = stack.train_at_depth(depth)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9999))
    g_optimizer = torch.optim.Adam(g_params, lr=1e-4, betas=(0.5, 0.9999))
    #d_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=d_optimizer, step_size=400)
    #g_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=g_optimizer, step_size=400)
    discriminator.train()
    stack.train()

    if depth > 0:
        reconstruction = stack.reconstruction(depth - 1)
        reconstruction = torchvision.transforms.Resize(stack.sizes[depth])(reconstruction)
        stack.noise_amp[depth] = torch.sqrt(torch.nn.MSELoss()(reconstruction, reals[depth])).detach()

    for i in range(iterations):

        # TODO: WGAN LOSS

        # Step discriminator.
        discriminator.zero_grad()
        d_optimizer.zero_grad()
        fakes = stack.random(depth, 16)
        x_pred = discriminator(reals[depth])
        z_pred = discriminator(fakes.detach())
        # gp_loss = compute_grad_gp_wgan(device, discriminator, reals[depth], fakes[0]) * 0.01
        d_loss = z_pred.mean() - x_pred.mean()
        #(d_loss + gp_loss).backward()
        d_loss.backward()
        d_optimizer.step()
        with torch.no_grad():
            for param in discriminator.parameters():
                param.clamp_(-0.01, 0.01)
        # Step Generator
        stack.zero_grad()
        g_optimizer.zero_grad()
        #fakes = stack.random(depth, 16)
        assert fakes.shape[0] == 16
        assert fakes.shape[1:] == reals[depth].shape
        z_pred = discriminator(fakes)
        reconstruction = stack.reconstruction(depth)
        r_loss = torch.nn.MSELoss()(reconstruction, reals[depth]) * args.res_loss_mult
        g_loss = -z_pred.mean()
        (g_loss + r_loss).backward()
        g_optimizer.step()

        #d_scheduler.step()
        #g_scheduler.step()

        print(f"{g_loss.item():10.6f} {r_loss.item():10.6f} {d_loss.item():10.6f} {depth+1}/{len(sizes)} {stack.noise_amp[depth]:8.4f} {i+1}")


def init_weights(layer):
    if hasattr(layer, 'weight'):
        torch.nn.init.normal_(layer.weight.data, std=0.02)
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'data'):
        torch.nn.init.constant_(layer.bias.data, 0.0)


def save_image(fname, img_tensor):
    assert len(img_tensor.shape) == 3
    assert img_tensor.shape[0] == 3
    img_np = img_tensor.cpu().numpy()
    img_np = (img_np + 1.0) / 2.0
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np, 0.0, 1.0)
    img = PIL.Image.fromarray((img_np*255).astype(np.uint8))
    img.save(args.output_dir + '/' + fname)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device = {device}")

real_img = PIL.Image.open(args.real)
real_np = np.array(real_img)
real_np = np.transpose(real_np, (2, 0, 1))
real_np = (real_np / 255)*2.0 - 1.0
real_ = torch.tensor(real_np).float()[:3, :, :].to(device)
min_size = 22
sizes = [(real_.shape[1], real_.shape[2])]
while sizes[0][0] > min_size and sizes[0][1] > min_size:
    i = len(sizes)
    s = (int(sizes[-1][0] * 0.75**i), int(sizes[-1][1] * 0.75**i))
    sizes.insert(0, s)

#sizes = sizes[:5]

reals = [
    torchvision.transforms.Resize(size)(real_) for size in sizes
]

for i, real in enumerate(reals):
    save_image(f"real{i}.png", real)

stack = GeneratorStack(device, sizes).to(device)
if os.path.isfile('stack.pth'):
    stack.load_state_dict(torch.load('stack.pth', map_location=device))
else:
    stack.apply(init_weights)
    for d in range(len(sizes)):
        train_level(device, stack, reals, d, args.iterations)
    torch.save(stack.state_dict(), args.output_dir + '/' + 'stack.pth')
stack.disable_train()

print(stack.noise_amp)

with torch.no_grad():
    for d in range(len(sizes)):
        reconstruction = stack.reconstruction(d)
        reconstruction_loss = torch.nn.MSELoss(reduction='sum')(reconstruction, reals[d])
        fname = f"reconstruction{d}.png"
        print(fname, torch.min(reconstruction).item(), torch.max(reconstruction).item(), reconstruction_loss.item())
        save_image(fname, reconstruction)

        for i in range(10):
            fake = stack.random(d)
            fname = f"fake{d}-{i}.png"
            print(fname, torch.min(fake).item(), torch.max(fake).item())
            save_image(fname, fake)

