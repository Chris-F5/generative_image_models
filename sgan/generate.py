import torch
import numpy as np
import PIL.Image


def generate_texture(model_fname, width, height):
    with torch.no_grad():
        device = torch.device('cpu')
        generator = torch.jit.load(model_fname, map_location=device)
        z = generator.random_latent_tensor(1, width, height)
        x = generator(z)[0, :, :height, :width]
        im_np = 255.0 * ((x[0, :, :].numpy() + 1.0) / 2.0)
        im = PIL.Image.fromarray(im_np.astype(np.uint8))
        return im


if __name__ == "__main__":
    im = generate_texture("./models/paper_foundation_online-generator-006000_iterations.pth", 1920, 1080)
    im.save("output.png")
