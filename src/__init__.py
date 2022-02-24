import torch


def unfold1d(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """
    Utility function to spread/slice/unfold? 1D signal into its separate windows
    Only works for stride == 1, batch_size == 1, n_channels == 1.
    """
    _, _, l = x.size()
    return x.as_strided((1, 1, l - (ksize - 1), ksize), (1, 1, 1, 1))


def scs(signal: list[float], kernel: list[float], p: int, q: float) -> dict:
    """
    Calculate the result of a normal convolution and a sharpened cosine similarity
    on the input signal (1D), given the kernel/filter and hyperparameters of SCS.
    """

    """
    1,2,3,4,5

    1,2 * [k1 k2]
    2,3
    4,5

    """
    ksize = len(kernel)

    # prepare signal tensor
    signal = torch.tensor(signal, dtype=torch.float32)[None, None]

    # prepare kernel tensort
    kernel = torch.tensor(kernel, dtype=torch.float32)[None, None]

    # create sliding windows
    signal_windows = unfold1d(signal, ksize)

    # calculate dot product/convolution
    dp = (signal_windows * kernel).sum(-1, keepdim=True)

    # sign function on dot product
    sign = dp.sign()
    # sign = 1

    # calculate then norms of the windows and kernel
    s_norm = signal_windows.pow(2).sum(-1, keepdim=True).sqrt()
    k_norm = kernel.pow(2).sum(-1, keepdim=True).sqrt()

    # put it al together
    scs = sign * (dp / ((k_norm + q) * (s_norm + q))).pow(p)

    # prepare dot product for response
    dp = dp[0, 0, :, 0].tolist()

    # prepare SCS for response
    scs = scs[0, 0, :, 0].tolist()

    # adding padding to match input length
    return {"convolution": [0] + dp + [0], "scs": [0]+ scs + [0]}
