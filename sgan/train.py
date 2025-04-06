import os
import torch
from sgan_models import SGAN4_Generator, SGAN4_Discriminator
from patch_dataset import PatchDataset


def init_weights(layer):
    if hasattr(layer, 'weight'):
        torch.nn.init.normal_(layer.weight.data, std=0.02)
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'data'):
        torch.nn.init.constant_(layer.bias.data, 0.0)


latent_image_dims = 4
batch_size = 64
train_iterations = 6000
save_frequency = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device = {device}")

generator = SGAN4_Generator().to(device)
discriminator = SGAN4_Discriminator().to(device)
assert generator.SCALE_FACTOR == generator.SCALE_FACTOR

patch_size = latent_image_dims * generator.SCALE_FACTOR

dataset = PatchDataset("./examples/crumpled-train.png", patch_size, mean_patch=False)
patch_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)

generator.apply(init_weights)
discriminator.apply(init_weights)

discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
loss_fn = torch.nn.BCELoss()

generator.train()
discriminator.train()
for i in range(train_iterations):

    # Step discriminator
    discriminator.zero_grad()
    x = next(iter(patch_loader))
    x = x.to(device)
    z = generator.random_latent_tensor(batch_size, patch_size, patch_size, device)
    x_pred = discriminator(x)
    z_pred = discriminator(generator(z))
    discriminator_optimizer.zero_grad()
    discriminator_loss = (
      loss_fn(x_pred, torch.ones_like(x_pred))
      + loss_fn(z_pred, torch.zeros_like(z_pred))
    ) / 2.0
    discriminator_loss.backward()
    discriminator_optimizer.step()

    # Step generator
    generator.zero_grad()
    z = generator.random_latent_tensor(batch_size, patch_size, patch_size, device)
    z_pred = discriminator(generator(z))
    generator_optimizer.zero_grad()
    generator_loss = loss_fn(z_pred, torch.ones_like(z_pred))
    generator_loss.backward()
    generator_optimizer.step()

    print(f"G: {generator_loss:>4f}, D: {discriminator_loss:>4f}    ", end='')
    print(f"[{i+1:>6d}/{train_iterations:>6d}][{100*(i+1)/train_iterations:04.1f}%]")

    if (i+1) % save_frequency == 0:
        generator_script = torch.jit.script(generator)
        discriminator_script = torch.jit.script(discriminator)
        torch.jit.save(generator_script, f"generator-{i+1:>06d}_iterations.pth")
        torch.jit.save(discriminator_script, f"discriminator-{i+1:>06d}_iterations.pth")
