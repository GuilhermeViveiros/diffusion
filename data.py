import torch
from torchvision import datasets, transforms
import os
from constants import IMG_FOLDER, DEVICE
from torch.utils.data import DataLoader
from utils import inverse_transform


# load mnist dataset
def load_mnist(img_size: int = 64, batch_size: int = 64, is_train=True, flatten=True):
    dataset = datasets.MNIST(
        "../data",
        train=is_train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # transforms.RandomHorizontalFlip(),
                #             torchvision.transforms.Normalize(MEAN, STD),
                transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
                # transforms.Lambda(lambda t: t.to(DEVICE)),
            ]
        ),
    )

    # get num workers
    num_workers = os.cpu_count()
    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=int(num_workers / 2),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # for batch in dataloader: x = batch; break
    return dataloader, None


# load cifar10 dataset
def load_cifar10(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.CIFAR10(
        "../data",
        train=is_train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (128, 128),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # transforms.RandomHorizontalFlip(),
                #             torchvision.transforms.Normalize(MEAN, STD),
                transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
            ]
        ),
    )
    # check if is numpy or tensor
    if not isinstance(dataset.data, torch.Tensor):
        dataset.data = torch.tensor(dataset.data)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True)
    # for batch in dataloader: x = batch; break
    # return dataloader
    return dataloader, None


composition = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (128, 128),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        #             torchvision.transforms.Normalize(MEAN, STD),
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
)


def preprocess(sample, img_size=(32, 32)):
    # get images - TO RGB, RESIZE, Transform
    sample["image"] = [composition(img.convert("RGB")) for img in sample["image"]]
    # return sample
    return sample


# load one piece dataset
def load_one_piece(batch_size: int = 6, visualize=False):
    from datasets import load_dataset, load_from_disk

    # x = datasets.list_datasets()

    # Load dataset and apply preprocessing -> save to disk
    if not os.path.exists("OnePiece"):
        dataset = load_dataset("polytechXhf/onepiece-dataset", split="train")
        # map x_train images (PIL images) and resize them to 128x128
        dataset = dataset.map(preprocess, batched=True, batch_size=16, num_proc=1)
        # save dataset to disk
        dataset.save_to_disk("OnePiece")

    # get dataset from disk
    dataset = load_from_disk("OnePiece").with_format("torch", device=DEVICE)
    # split dataset
    # dataset = dataset.train_test_split(test_size=0.1)
    # get train and test datasets
    # train, test = dataset["train"], dataset["test"]
    # visualize data
    # create dataloader
    train_dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    # test_dataloader = DataLoader(test, batch_size=batch_size, drop_last=True)
    # return train and test dataloaders
    return train_dataloader, None


# visualize data
def visualize_data(dataloader: DataLoader):
    import matplotlib.pyplot as plt

    # get data
    batch = next(iter(dataloader))
    # get images and labels
    # x, label, text = batch["image"], batch["char_name"], batch["text"]
    x, label = batch
    # print information statistics about the batch
    print(f"Batch Shape: {x.shape}")
    print(f"Label Shape: {label.shape}")
    print("X Mean: {}, X Std: {}".format(x.mean(), x.std()))
    print("X Min: {}, X Max: {}".format(x.min(), x.max()))

    # inverse transform the image from [-1, 1] to [0, 1]
    x = inverse_transform(x) / 255.0
    # if x in cuda, move to cpu
    if x.is_cuda:
        x = x.detach().cpu()
    # if x (C, H, W), transpose to (H, W, C)
    if x.shape[1] <= 3:
        x = x.permute(0, 2, 3, 1)
    # get number of samples
    n_samples = len(x)
    # if n_samples > 10, set n_samples to 10
    n_samples = 9 if n_samples > 9 else n_samples
    # create figure
    fig = plt.figure(figsize=(15, 15))
    # plot samples
    for i in range(n_samples):
        # get image & character name
        img, name = x[i], label[i]
        # add subplot
        ax = fig.add_subplot(3, 3, i + 1)
        # plot image
        ax.imshow(img)
        # set title with character name label at low with text
        ax.set_title(f"{name}", fontsize=15)
        # remove axis
        ax.axis("off")
    # save figure
    plt.savefig(IMG_FOLDER + "/data.png")


if __name__ == "__main__":
    # load mnist dataset
    train_dataloader, test_dataloader = load_one_piece()
    # sample 10 images from the dataset
    visualize_data(train_dataloader, img_path="assets/one_piece.png")
