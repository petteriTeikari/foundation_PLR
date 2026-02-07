import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    """Inception block with multiple parallel convolutional kernels of varying sizes.

    This block applies multiple 2D convolutions with different kernel sizes
    (1x1, 3x3, 5x5, etc.) in parallel and averages the results. This allows
    the network to capture patterns at multiple scales simultaneously.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for each convolutional kernel.
    num_kernels : int, optional
        Number of parallel convolutional kernels with different sizes.
        Kernel sizes will be 1, 3, 5, ..., (2*num_kernels-1). Default is 6.
    init_weight : bool, optional
        Whether to initialize weights using Kaiming initialization.
        Default is True.
    """

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolutional weights using Kaiming normal initialization.

        Applies Kaiming (He) initialization to all Conv2d layers in the module,
        which is suitable for layers followed by ReLU activations. Biases are
        initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Apply all parallel convolutions and average the results.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
            The output is the element-wise mean of all parallel convolution outputs.
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    """Inception block with separable horizontal and vertical convolutions.

    This block uses asymmetric kernels (1xN and Nx1) to capture horizontal
    and vertical patterns separately, plus a 1x1 convolution. This reduces
    computational cost compared to full NxN kernels while maintaining
    expressiveness.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for each convolutional kernel.
    num_kernels : int, optional
        Controls the number of parallel convolutional kernels.
        Creates (num_kernels // 2) pairs of horizontal/vertical kernels
        plus one 1x1 kernel. Default is 6.
    init_weight : bool, optional
        Whether to initialize weights using Kaiming initialization.
        Default is True.
    """

    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolutional weights using Kaiming normal initialization.

        Applies Kaiming (He) initialization to all Conv2d layers in the module,
        which is suitable for layers followed by ReLU activations. Biases are
        initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Apply all parallel convolutions and average the results.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
            The output is the element-wise mean of all parallel convolution outputs,
            including horizontal kernels, vertical kernels, and the 1x1 kernel.
        """
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
