from typing import List
import torch
import torch.nn as nn
import math
from models.attention import attn_block


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        max_len: int = 1000,
        apply_dropout: bool = True,
    ):
        """Section 3.5 of attention is all you need paper.

        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.

        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.

        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, embedding_dim, 2).float()
            / embedding_dim
        )

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name="pos_encoding", tensor=pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """Get precalculated positional embedding at timestep t. Outputs same as video implementation
        code but embeddings are in [sin, cos, sin, cos] format instead of [sin, sin, cos, cos] in that code.
        Also batch dimension is added to final output.
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        """Double convolutions as applied in the unet paper architecture."""
        super(DoubleConv, self).__init__()
        self.residual = residual

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return torch.nn.functional.relu(x + self.double_conv(x))

        return self.double_conv(x)


# the following represents a residual block between resoultion levels
class Down(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        attention: bool = False,
        device=torch.device("cuda:0"),
    ):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(
                in_channels=in_channels, out_channels=in_channels, residual=True
            ),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )
        self.attention = attention
        # positional embedding layer
        self.emb_layer = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, t_embedding: torch.LongTensor) -> torch.Tensor:
        # apply convolutions
        x = self.maxpool_conv(x)
        # apply attention
        if self.attention:
            out, self.att_w = attn_block(x)
            x = x + out
        # apply positional embedding
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


# the following class represents the Encoder part of the UNET
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, ch_size: int = 64, emb_dim: int = 256):
        super(Encoder, self).__init__()
        # each encoder stage comprises two residual blocks with convolutional downsampling except the last level.
        self.input_conv = DoubleConv(in_channels=in_channels, out_channels=ch_size)
        self.emb_layer = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb_dim, out_features=ch_size),
        )
        self.down1 = Down(in_channels=ch_size, out_channels=ch_size * 2)
        self.down2 = Down(in_channels=ch_size * 2, out_channels=ch_size * 4)
        self.down3 = Down(
            in_channels=ch_size * 4, out_channels=ch_size * 8, attention=True
        )
        self.down4 = Down(
            in_channels=ch_size * 8, out_channels=ch_size * 8, attention=True
        )

    def forward(self, x, t_embedding: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_conv(x)
        # apply positional embedding
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        x = x + emb
        x1 = self.down1(x, t_embedding)
        x2 = self.down2(x1, t_embedding)
        x3 = self.down3(x2, t_embedding)
        x4 = self.down4(x3, t_embedding)
        return (x, x1, x2, x3, x4)


# the following represents a residual block between resoultion levels
class Up(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256,
        attention: bool = False,
        device=torch.device("cuda:0"),
    ):
        super(Up, self).__init__()
        self.conv_transpose = nn.Sequential(
            DoubleConv(
                in_channels=in_channels, out_channels=in_channels, residual=True
            ),
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.attention = attention
        # positional embedding layer
        self.emb_layer = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, t_embedding: torch.LongTensor) -> torch.Tensor:
        # apply convolutions
        x = self.conv_transpose(x)
        # apply attention
        if self.attention:
            out, self.att_w = attn_block(x)
            x = x + out
        # apply positional embedding
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


# the following class represents the Decoder part of the UNET
class Decoder(torch.nn.Module):
    def __init__(self, in_channels: int = 512):
        super(Decoder, self).__init__()
        # upsampling is done by transposed convolution
        # each decoder stage comprises two residual blocks with convolutional upsampling except the last level.
        self.up1 = Up(in_channels=in_channels, out_channels=in_channels, attention=True)
        self.up2 = Up(
            in_channels=int(in_channels),
            out_channels=int(in_channels / 2),
            attention=True,
        )
        self.up3 = Up(
            in_channels=int(in_channels / 2), out_channels=int(in_channels / 4)
        )
        self.up4 = Up(
            in_channels=int(in_channels / 4), out_channels=int(in_channels / 8)
        )
        self.final_conv = DoubleConv(
            in_channels=int(in_channels / 8), out_channels=3, residual=False
        )

    def forward(self, x, t_embedding: torch.Tensor) -> torch.Tensor:
        x1 = x[1] + self.up1(x[0], t_embedding)
        x2 = x[2] + self.up2(x1, t_embedding)
        x3 = x[3] + self.up3(x2, t_embedding)
        x4 = x[4] + self.up4(x3, t_embedding)
        x5 = self.final_conv(x4)
        return x5


# the following class implements UNET
class UNET(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
    ):
        super(UNET, self).__init__()
        self.pos_encoding = PositionalEncoding(
            embedding_dim=256, max_len=1000, apply_dropout=False
        )
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(in_channels=out_channels)
        self.bottleneck1 = DoubleConv(
            in_channels=out_channels, out_channels=out_channels, residual=True
        )
        self.bottleneck2 = DoubleConv(
            in_channels=out_channels, out_channels=out_channels, residual=True
        )

    def forward(self, x, t: torch.LongTensor) -> torch.Tensor:
        """Forward pass with image tensor and timestep reduce noise.

        Args:
            x: Image tensor of shape, [batch_size, channels, height, width].
            t: Time step defined as long integer. If batch size is 4, noise step 500, then random timesteps t = [10, 26, 460, 231].
        """
        t = self.pos_encoding(t)
        # encode
        [x1, x2, x3, x4, x5] = self.encoder(x, t)
        # bottleneck
        x5 = x5 + self.bottleneck1(x5)
        x5 = x5 + self.bottleneck2(x5)
        # decode
        decoded = self.decoder([x5, x4, x3, x2, x1], t)
        return decoded


if __name__ == "__main__":
    from constants import IMG_FOLDER
    import matplotlib.pyplot as plt
    from torchsummary import summary

    # implement UNET has in this image:
    # https://learnopencv.com/wp-content/uploads/2023/02/denoising-diffusion-probabilistic-models_UNet_model_architecture.png
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    B, C, H, W = 2, 3, 64, 64
    dummy_input = torch.randn(B, C, H, W).to(device)

    # test attention block
    """
    out, att_w = attn_block(dummy_input, device=device)
    plt.imshow(att_w.view(B, H * W, H * W, -1)[0, :, :, 0].detach().cpu().numpy())
    plt.colorbar()
    plt.savefig(IMG_FOLDER + '/attention_weights.png')
    """

    # test positional embedding
    """
    encoding = PositionalEncoding(embedding_dim=256, max_len=1000, apply_dropout=False)
    pos_encoding = encoding([2])
    """

    # test encoder
    """
    encoder = Encoder(in_channels=C)
    encoded_out = encoder(dummy_input)
    # visualize encoder architecture summary
    summary(encoder, (C, H, W))
    """

    # test decoder
    """
    decoder = Decoder(in_=512)
    # invert encoder output
    decoder_out = decoder(encoded_out[::-1])
    """

    # test Unet

    unet = UNET(in_channels=C).to(device)
    # unet_out = unet(dummy_input)
    unet_out = unet(dummy_input, t=[100])
    # print summary of UNET in a beautiful way
    # summary(unet, (C, H, W))
    print("UNET summary:")
    print("--------------")
    # summary(unet, [(C, H, W), torch.LongTensor(1)], batch_size=1)
    print("Input shape: ", dummy_input.shape)
    print("Output shape: ", unet_out.shape)
    print("--------------")
