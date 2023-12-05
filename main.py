import torch
from torch import nn
import argparse
import tqdm
import os
import matplotlib.pyplot as plt
from data import *
from constants import IMG_FOLDER, DEVICE
import numpy as np
from diffusion import DiffusionProcess
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from utils import inverse_transform, show
import torchvision
import gc


def load_data(data, img_size: int, batch_size: int, visualize=False):
    if data == "mnist":
        train_dataloader, test_dataloader = load_mnist(
            img_size=img_size, batch_size=batch_size, flatten=True
        )
    elif data == "cifar10":
        train_dataloader, test_dataloader = load_cifar10(flatten=False)
    elif data == "one_piece":
        train_dataloader, test_dataloader = load_one_piece(batch_size=batch_size)
    else:
        raise Exception("invalid data name: {}".format(data))
    # ensure the data comes in data loader format
    assert isinstance(train_dataloader, DataLoader)
    # assert isinstance(test_dataloader, DataLoader)

    # print statistics about the data distribution (mean & std, number of samples, etc.)
    print("train dataset statistics:")
    print("\tmean: {}".format(train_dataloader.dataset.data.float().mean()))
    print("\tstd: {}".format(train_dataloader.dataset.data.float().std()))
    print("\tmin: {}".format(train_dataloader.dataset.data.float().min()))
    print("\tmax: {}".format(train_dataloader.dataset.data.float().max()))
    print("\tshape: {}".format(train_dataloader.dataset.data.shape))
    print("\tlabels: {}".format(train_dataloader.dataset.targets))

    # visualize data
    if visualize:
        visualize_data(train_dataloader)
    # return train and test dataloaders
    return train_dataloader, test_dataloader


def diffusion_illustration(
    x: torch.Tensor, diffusion_process: DiffusionProcess, sample: int = 10
):
    # this function illustrates the diffusion process at several steps
    # get number of total steps
    n_steps = diffusion_process.linear_scheduler.n_steps
    # get ts
    ts = torch.linspace(0, n_steps - 1, steps=sample, dtype=torch.long)
    # repeat x for sample times
    xs = x.repeat(sample, 1, 1, 1)
    # get the diffused images
    diffused_imgs, noises = diffusion_process.vectorized_diffuse(xs, ts)
    # illustrate the diffusion process for 10 evenly spaced steps and save all the results in the same figure
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(sample):
        # get timestep
        t = ts[i]
        # get the current image
        xt = diffused_imgs[i]
        noise = noises[i]
        # if current_img on cuda, move to cpu
        if xt.is_cuda:
            xt = xt.detach().cpu()
            noise = noise.detach().cpu()
        # if (C, H, W), transpose to (H, W, C)
        if xt.shape[0] == 3:
            xt = xt.permute(1, 2, 0)
            noise = noise.permute(1, 2, 0)
        # transform the image from [-1, 1]
        xt = inverse_transform(xt) / 255.0
        # plot the current image & noise
        ax = axes[i // 5, i % 5]
        ax.imshow(np.squeeze(xt), cmap="gray")
        ax.set_title("time step: {}".format(t))
    # save image in IMAGE_FOLDER
    plt.savefig(IMG_FOLDER + "/diffusion_process.png")


# Algorithm 1: Training
def train_one_epoch(model, dataloader, diffusion_process, loss_fn, optimizer, epoch):
    loss_record = MeanMetric()
    model.train()

    with tqdm.tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{1000}")
        for imgs, _ in dataloader:
            # for batch in dataloader:
            # imgs, label, text = batch["image"], batch["char_name"], batch["text"]
            # get image and label
            # imgs, labels = x["image"], x["char_name"]
            # sample statics about the distribution
            # print(
            #    "Sample Mean: {}, Sample Std: {}".format(
            #        torch.mean(imgs), torch.std(imgs)
            #    )
            # )
            # print(
            #    "Sample Min: {}, Sample Max: {}".format(
            #        torch.min(imgs), torch.max(imgs)
            #    )
            # )
            imgs = imgs.to(DEVICE)
            # labels.to(DEVICE)
            # sample dataloader.batch_size steps from the diffusion process
            t_steps = torch.randint(
                0, diffusion_process.n_steps, (dataloader.batch_size,)
            ).to(DEVICE)

            # get the sampled diffused images
            diffused_imgs, sampled_noise = diffusion_process.vectorized_diffuse(
                imgs, t_steps
            )

            # clear the gradients
            optimizer.zero_grad()
            # predict the noise
            pred_noise = model(diffused_imgs, t_steps)
            # compute the loss
            loss = loss_fn(sampled_noise, pred_noise)
            # if loss as nans or infs, raise an exception
            if torch.isnan(loss) or torch.isinf(loss):
                print(sampled_noise, pred_noise)
                raise Exception("loss is nan or inf")

            # update the loss record
            loss_record.update(loss.item(), dataloader.batch_size)
            # backpropagation
            loss.backward()
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # update the parameters
            optimizer.step()
            # update tqdm
            tq.set_postfix(loss=loss_record.compute())
            # print("loss: {}".format(loss_record.compute()))
            tq.update(1)
    return loss_record.compute()


# Algorithm 2: Sampling
@torch.no_grad()
def reverse_diffusion(
    model,
    diffusion_process,
    timesteps=1000,
    img_shape=(3, 128, 128),
    num_images=1,
    nrow=8,
    visualize=True,
    **kwargs,
):
    x = torch.randn((num_images, *img_shape), device=DEVICE)
    model.eval()

    def get(xs, ts):
        # iterate over all time steps and get x_t
        return torch.tensor([xs[t] for t in ts]).view(num_images, 1, 1, 1)

    for time_step in tqdm.tqdm(
        iterable=reversed(range(1, timesteps)),
        total=timesteps - 1,
        dynamic_ncols=False,
        desc="Sampling :: ",
        position=0,
    ):
        ts = torch.ones(num_images, dtype=torch.int, device=DEVICE) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)
        # get the predicted noise
        predicted_noise = model(x, ts)
        # if predicted_noise contains nans raise Exception
        if torch.isnan(predicted_noise).any():
            raise Exception("predicted noise contains nans")
        # get the current beta and alpha values
        beta_t = get(diffusion_process._beta, ts).to(DEVICE)
        alpha_t = get(diffusion_process._alpha, ts).to(DEVICE)
        # alpha_cum_t = get(diffusion_process._alpha_cumulative, ts).to(DEVICE)
        one_by_sqrt_alpha_t = get(diffusion_process._one_by_sqrt_alpha, ts).to(DEVICE)
        sqrt_one_minus_alpha_cumulative_t = get(
            diffusion_process._sqrt_one_minus_alpha_cumulative, ts
        ).to(DEVICE)

        x = (
            one_by_sqrt_alpha_t
            * (x - ((beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise))
        ) + torch.sqrt(beta_t) * z

        # Display and save the image at the final timestep of the reverse process.
        if visualize and time_step % 1 == 0:
            # print("time step: {}".format(time_step))
            # clone torch x into _x
            _x = x.clone()
            _x = inverse_transform(_x).type(torch.uint8)
            grid = torchvision.utils.make_grid(_x, nrow=nrow, pad_value=255.0).to("cpu")
            # grid = torchvision.utils.make_grid(_x, nrow=nrow)
            show(grid, kwargs["save_path"])
        # pil_image = TF.functional.to_pil_image(grid)
        # pil_image.save(IMG_FOLDER + "/reverse_diffusion.png")


def train(epochs, model, diffusion, dataloader, optimizer, loss_fn, checkpt_dir):
    for epoch in range(1, epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, dataloader, diffusion, loss_fn, optimizer, epoch=epoch)
        # sample every 20 epochs
        if epoch % 1 == 0:
            # Algorithm 2: Sampling
            reverse_diffusion(
                model,
                diffusion,
                timesteps=diffusion.n_steps,
                num_images=6,
                save_path=IMG_FOLDER + f"/reverse_diffusion_{epoch}.png",
                # save_path=IMG_FOLDER + f"/reverse_diffusion_4.png",
                img_shape=(1, IMG_SIZE, IMG_SIZE),
                nrow=8,
                visualize=True,
            )

        # save checkpoint per 100 epochs
        if epoch % 10 == 0:
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "model": model.state_dict(),
            }
            torch.save(checkpoint_dict, os.path.join(checkpt_dir, f"ckpt_{epoch}.pt"))
            del checkpoint_dict


if __name__ == "__main__":
    # read arguments from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to config file"
    )
    parser.add_argument("--mode", type=str, default="train", help="mode: train or test")
    parser.add_argument("--data", type=str, default="mnist", help="data")
    parser.add_argument("--batch_size", type=int, default=224, help="batch size")
    parser.add_argument("--visualize", type=bool, default=False, help="visualize data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--img_size", type=int, default=32)
    # parse arguments
    args = parser.parse_args()
    # print arguments in a beautiful way
    print("Arguments:")
    for arg in vars(args):
        print("\t" + arg + ":", getattr(args, arg))
    IMG_SIZE = args.img_size
    # load device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load data
    train_dataloader, y_dataloader = load_data(
        args.data, args.img_size, args.batch_size, visualize=args.visualize
    )
    # initialize diffusion process
    diffusion_process = DiffusionProcess(args.steps)
    # illustrate diffusion process
    if args.visualize:
        # get first batch
        (x, y) = next(iter(train_dataloader))
        # x, label, text = batch["image"], batch["char_name"], batch["text"]
        print("sample image shape: {}".format(x[0].shape))
        diffusion_illustration(x[0], diffusion_process)

    # load the model
    # from models import UNET
    # model = UNET(in_channels=3, out_channels=512).to(device)
    from models import UNet

    model = UNet(image_channels=1, n_channels=64)
    model.to(device)

    # load model parameters
    # if os.path.exists(os.path.join(args.checkpoint_dir, "ckpt.pt")):
    #    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "ckpt.pt"))
    #    model.load_state_dict(checkpoint["model"])
    #    print("model loaded")
    # define model configurations
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    # one training epoch
    # train_one_epoch(model, train_dataloader, diffusion_process, loss_fn, optimizer, 1)
    # reverse diffusion
    """
    reverse_diffusion(
        model,
        diffusion_process,
        timesteps=1000,
        img_shape=(1, 128, 128),
        num_images=5,
        nrow=8,
        save_path=IMG_FOLDER + "/reverse_diffusion.png",
        visualize=True,
    )
    """

    # train model
    train(
        epochs=1000,
        model=model,
        diffusion=diffusion_process,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpt_dir=args.checkpoint_dir,
    )

    # load model
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "ckpt.pt"))
    model.load_state_dict(checkpoint["model"])
    print("model loaded")
    reverse_diffusion(
        model,
        diffusion_process,
        timesteps=args.steps,
        img_shape=(1, IMG_SIZE, IMG_SIZE),
        num_images=5,
        nrow=8,
        save_path=IMG_FOLDER + "/reverse_diffusion_test.png",
        visualize=True,
    )
