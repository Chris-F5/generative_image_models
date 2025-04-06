import torch
import numpy as np
import PIL.Image


class PatchDataset(torch.utils.data.Dataset):

    # TODO: Add option to flip and rotate image for more data.

    def __init__(self, image_path, patch_size, mean_patch=False):
        self.patch_size = patch_size
        self.mean_patch = mean_patch

        # Load image as greyscale.
        im = PIL.Image.open(image_path).convert('L')
        im.save('tmp.png')

        assert im.width >= patch_size
        assert im.height >= patch_size

        # Rescale to [-1,1].
        self.im_tensor = torch.from_numpy((2.0*np.array(im)/255.0)-1.0).float()

        self.patches_width = (self.im_tensor.shape[0]-self.patch_size)
        self.patches_height = (self.im_tensor.shape[1]-self.patch_size)

    def __len__(self):
        return self.patches_width * self.patches_height

    def __getitem__(self, idx):
        x = idx % self.patches_width
        y = idx // self.patches_width
        patch = self.im_tensor[x:x+self.patch_size, y:y+self.patch_size].clone()
        if self.mean_patch:
            patch -= torch.mean(patch) - torch.mean(self.im_tensor)
            patch = torch.clamp(patch, min=-1.0, max=1.0)
        return patch.view(1, self.patch_size, self.patch_size)


if __name__ == "__main__":
    import PIL.Image
    count = 10
    dataset = PatchDataset("rough_100x_2d_scaled.png", 16*8, mean_patch=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=count,
        shuffle=True,
    )
    x = next(iter(loader))
    for i in range(count):
        im_np = 255.0 * ((x[i, 0, :, :].numpy() + 1.0) / 2.0)
        im = PIL.Image.fromarray(im_np.astype(np.uint8))
        im.save(f"patch_{i:02d}.png")
