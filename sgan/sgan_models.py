import torch

class SGAN4_Generator(torch.nn.Module):
    SCALE_FACTOR=16

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(20, 256, 5, stride=2, padding=2, output_padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

    @torch.jit.export
    def random_latent_tensor(
            self, batch_size: int, width: int, height: int,
            device: torch.device = torch.device("cpu")):
        latent_w = (width+15) // 16
        latent_h = (height+15) // 16
        return 2.0 * torch.rand((batch_size, 20, latent_h, latent_w), device=device) - 1.0


class SGAN4_Discriminator(torch.nn.Module):
    SCALE_FACTOR=16

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 5, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 5, stride=2, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 5, stride=2, padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 1, 5, stride=2, padding=2),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
