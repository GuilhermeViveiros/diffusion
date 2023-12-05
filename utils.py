import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = "tight"


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    # return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0
    return torch.clamp((tensors + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
    # return (tensors + 1.0) * 127.5


def show(imgs, path: str):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # save img
    plt.savefig(path)
