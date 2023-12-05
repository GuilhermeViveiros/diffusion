import torch


# the following function implements a self-attention block
def attn_block(x, device=torch.device("cuda:0")):
    """
    Self-attention layer as by Ian Goodfellow as described in:
    Self-Attention Generative Adversarial Networks [https://arxiv.org/pdf/1805.08318.pdf]
    Attention is all you need [https://arxiv.org/pdf/1706.03762.pdf]
    @MainIdea
        Convolution processes the information in a local neighborhood,
        thus using convolutional layers alone is computationally inefficient
        for modeling long-range dependencies in images
    """
    B, C, H, W = x.shape
    # define convolutions
    q = torch.nn.Conv2d(
        C, C, kernel_size=1, stride=1, padding=0, device=device
    )  # query (B, C, H, W)
    k = torch.nn.Conv2d(
        C, C, kernel_size=1, stride=1, padding=0, device=device
    )  # key (B, C, H, W)
    v = torch.nn.Conv2d(
        C, C, kernel_size=1, stride=1, padding=0, device=device
    )  # value (B, C, H, W)
    # apply convolutions
    q = q(x)  # (B, C, H, W)
    k = k(x)  # (B, C, H, W)
    v = v(x)  # (B, C, H, W)
    # view tensors to make them compatible for matrix multiplication
    q = q.view(B, C, -1)  # Reshape to (B, C, H * W)
    k = k.view(B, C, -1)  # Reshape to (B, C, H * W)

    # apply scaled dot product attention (https://arxiv.org/pdf/1706.03762.pdf)
    w = (
        q.transpose(-2, -1) @ k * (int(C) ** (-0.5))
    )  # (B, H * W, C) @ (B, C, H * W) = (B, H * W, H * W)
    # view w to make it compatible for matrix multiplication
    w = w.view(B, H, W, -1)  # (B, H * W, H * W) -> (B, H, W, H * W)
    # apply softmax to get attention weights
    import pdb

    pdb.set_trace()
    att_w = torch.nn.functional.softmax(w, dim=-1)  # (B, H, W, H * W)
    # Reshape attention weights to make them compatible for matrix multiplication
    att_w = att_w.view(B, H * W, H * W)  # (B, H, W, H * W) -> (B, H * W, H * W)
    v = v.view(B, -1, C)  # Reshape to (B, H * W, C)
    # apply attention weights to v
    out = att_w @ v  # (B, H * W, H * W) @ (B, H * W, C) = (B, H * W, C)
    # reshape attention_output
    out = out.view(B, C, H, W)  # (B, H * W, C) -> (B, C, H, W)
    # return attention_output and attention weights
    return out, att_w


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        is_inplace: bool = True,
    ):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head**-0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(
            d_model, d_attn, bias=False
        )  # query (B, E, H, W) -> (B, D, H, W)
        self.to_k = nn.Linear(
            d_cond, d_attn, bias=False
        )  # key (B, E, H, W) -> (B, D, H, W)
        self.to_v = nn.Linear(
            d_cond, d_attn, bias=False
        )  # value (B, E, H, W) -> (B, D, H, W)

        # Final linear layer
        self.to_out = nn.Sequential(
            nn.Linear(d_attn, d_model)
        )  # (B, D, H, W) -> (B, E, H, W)

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attention import FlashAttention

            self.flash = FlashAttention()
            # Set the scale for scaled dot-product attention.
            self.flash.softmax_scale = self.scale
        # Set to `None` if it's not installed
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """

        # If `cond` is `None` we perform self attention
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if (
            CrossAttention.use_flash_attention
            and self.flash is not None
            and not has_cond
            and self.d_head <= 128
        ):
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f"Head size ${self.d_head} too large for Flash Attention")

        # Pad the heads
        if pad:
            qkv = torch.cat(
                (qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1
            )

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out, _ = self.flash(qkv)
        # Truncate the extra head size
        out = out[:, :, :, : self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum("bhij,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, height * width, d_model]` with a linear layer
        return self.to_out(out)
