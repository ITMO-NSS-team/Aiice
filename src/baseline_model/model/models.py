from torchcnnbuilder.models import ForecasterBase
from torch import nn
from typing import Union, List
import torch
import torch.nn.functional as F

class CustomEncoderDecoder(nn.Module):
    def __init__(self, num_layers:int, channels_in: int, channels_out: int, image_size:Union[List[int], int]) -> None:
        """
        A custom Encoder-Decoder model for image forecasting, built upon `ForecasterBase`.

        This class wraps the `ForecasterBase` model from `torchcnnbuilder` to provide
        a convenient interface for constructing a Encoder - Decoder architecture tailored
        for forecasting tasks. It handles the initial setup of the image size
        and delegates the core model logic to `ForecasterBase`.

        Args:
            num_layers (int): The number of layers (depth) for the Encoder-Decoder architecture.
                            This parameter is passed directly to `ForecasterBase`.
            channels_in (int): The number of input channels (channels are the smallest uni). This corresponds
                            to `in_time_points` in `ForecasterBase`.
            channels_out (int): The number of output channels (time duration that model will predict). 
                            This corresponds to `out_time_points` in `ForecasterBase`.
            image_size (Union[List[int], int]): The spatial dimensions of the input image.
                                                 If an integer is provided, it will be
                                                 interpreted as a square image (e.g., 256
                                                 becomes [256, 256]). This is used as
                                                 `input_size` for `ForecasterBase`.
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        self.model = ForecasterBase(input_size = image_size,
                                    in_time_points = channels_in,
                                    out_time_points = channels_out,
                                    n_layers = num_layers)
    
    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor to the model. Expected shape depends on
                              the `ForecasterBase` requirements, typically
                              `(batch_size, channels_in, height, width)`.

        Returns:
            torch.Tensor: The output tensor from the model. Expected shape typically
                          `(batch_size, channels_out, height, width)`.
        """
        return self.model(x)
    
    
class SimpleUNet(nn.Module):
    """A simplified U-Net implementation in a single monolithic class.

    This class defines the entire U-Net architecture without using separate
    sub-modules. All layers are defined directly in the initializer, and the
    forward pass logic is explicitly laid out in the `forward` method. This
    structure makes the data flow and layer sequence clear at a glance.

    The architecture consists of an encoding path (contracting) with max-pooling
    and a decoding path (expansive) that uses trainable transposed convolutions
    to restore resolution. Skip connections concatenate feature maps from the
    encoder to the decoder to preserve spatial information.

    Args:
        n_channels (int): Number of input channels for the image (e.g., 3 for
            RGB, 1 for grayscale).
        n_classes (int): Number of output classes for the segmentation task.
            This determines the number of output channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.inc_bn1 = nn.BatchNorm2d(64)
        self.inc_relu1 = nn.ReLU(inplace=True)
        self.inc_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.inc_bn2 = nn.BatchNorm2d(64)
        self.inc_relu2 = nn.ReLU(inplace=True)

        self.down1_pool = nn.MaxPool2d(2)
        self.down1_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.down1_bn1 = nn.BatchNorm2d(128)
        self.down1_relu1 = nn.ReLU(inplace=True)
        self.down1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.down1_bn2 = nn.BatchNorm2d(128)
        self.down1_relu2 = nn.ReLU(inplace=True)

        self.down2_pool = nn.MaxPool2d(2)
        self.down2_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.down2_bn1 = nn.BatchNorm2d(256)
        self.down2_relu1 = nn.ReLU(inplace=True)
        self.down2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.down2_bn2 = nn.BatchNorm2d(256)
        self.down2_relu2 = nn.ReLU(inplace=True)

        self.down3_pool = nn.MaxPool2d(2)
        self.down3_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.down3_bn1 = nn.BatchNorm2d(512)
        self.down3_relu1 = nn.ReLU(inplace=True)
        self.down3_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.down3_bn2 = nn.BatchNorm2d(512)
        self.down3_relu2 = nn.ReLU(inplace=True)

        self.down4_pool = nn.MaxPool2d(2)
        self.down4_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.down4_bn1 = nn.BatchNorm2d(1024)
        self.down4_relu1 = nn.ReLU(inplace=True)
        self.down4_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False)
        self.down4_bn2 = nn.BatchNorm2d(1024)
        self.down4_relu2 = nn.ReLU(inplace=True)

        self.up1_upsample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.up1_bn1 = nn.BatchNorm2d(512)
        self.up1_relu1 = nn.ReLU(inplace=True)
        self.up1_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.up1_bn2 = nn.BatchNorm2d(512)
        self.up1_relu2 = nn.ReLU(inplace=True)

        self.up2_upsample = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.up2_bn1 = nn.BatchNorm2d(256)
        self.up2_relu1 = nn.ReLU(inplace=True)
        self.up2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(256)
        self.up2_relu2 = nn.ReLU(inplace=True)

        self.up3_upsample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.up3_bn1 = nn.BatchNorm2d(128)
        self.up3_relu1 = nn.ReLU(inplace=True)
        self.up3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.up3_bn2 = nn.BatchNorm2d(128)
        self.up3_relu2 = nn.ReLU(inplace=True)

        self.up4_upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.up4_bn1 = nn.BatchNorm2d(64)
        self.up4_relu1 = nn.ReLU(inplace=True)
        self.up4_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.up4_bn2 = nn.BatchNorm2d(64)
        self.up4_relu2 = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the U-Net.

        Args:
            x (torch.Tensor): The input tensor of shape
                (N, n_channels, H, W).

        Returns:
            torch.Tensor: The output segmentation map of shape
                (N, n_classes, H, W).
        """
        x1 = self.inc_relu2(self.inc_bn2(self.inc_conv2(self.inc_relu1(self.inc_bn1(self.inc_conv1(x))))))

        x2_in = self.down1_pool(x1)
        x2 = self.down1_relu2(self.down1_bn2(self.down1_conv2(self.down1_relu1(self.down1_bn1(self.down1_conv1(x2_in))))))

        x3_in = self.down2_pool(x2)
        x3 = self.down2_relu2(self.down2_bn2(self.down2_conv2(self.down2_relu1(self.down2_bn1(self.down2_conv1(x3_in))))))

        x4_in = self.down3_pool(x3)
        x4 = self.down3_relu2(self.down3_bn2(self.down3_conv2(self.down3_relu1(self.down3_bn1(self.down3_conv1(x4_in))))))

        x5_in = self.down4_pool(x4)
        x5 = self.down4_relu2(self.down4_bn2(self.down4_conv2(self.down4_relu1(self.down4_bn1(self.down4_conv1(x5_in))))))

        up1_out = self.up1_upsample(x5)
        diffY = x4.size()[2] - up1_out.size()[2]
        diffX = x4.size()[3] - up1_out.size()[3]
        up1_out = F.pad(up1_out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, up1_out], dim=1)
        x = self.up1_relu2(self.up1_bn2(self.up1_conv2(self.up1_relu1(self.up1_bn1(self.up1_conv1(x))))))

        up2_out = self.up2_upsample(x)
        diffY = x3.size()[2] - up2_out.size()[2]
        diffX = x3.size()[3] - up2_out.size()[3]
        up2_out = F.pad(up2_out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, up2_out], dim=1)
        x = self.up2_relu2(self.up2_bn2(self.up2_conv2(self.up2_relu1(self.up2_bn1(self.up2_conv1(x))))))

        up3_out = self.up3_upsample(x)
        diffY = x2.size()[2] - up3_out.size()[2]
        diffX = x2.size()[3] - up3_out.size()[3]
        up3_out = F.pad(up3_out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, up3_out], dim=1)
        x = self.up3_relu2(self.up3_bn2(self.up3_conv2(self.up3_relu1(self.up3_bn1(self.up3_conv1(x))))))

        up4_out = self.up4_upsample(x)
        diffY = x1.size()[2] - up4_out.size()[2]
        diffX = x1.size()[3] - up4_out.size()[3]
        up4_out = F.pad(up4_out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, up4_out], dim=1)
        x = self.up4_relu2(self.up4_bn2(self.up4_conv2(self.up4_relu1(self.up4_bn1(self.up4_conv1(x))))))

        logits = self.out_conv(x)
        return logits