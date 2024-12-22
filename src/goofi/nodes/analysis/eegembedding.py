import torch
from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam
from transformers import AutoConfig
import os
from safetensors.torch import load_file
import json
from torch import nn
import torch.nn as nn
import math


class EEGEmbedding(Node):
    def config_input_slots():
        return {"eeg": DataType.ARRAY}

    def config_output_slots():
        return {"embeddings": DataType.ARRAY}

    def config_params():
        return {
            "model": {
                "model_path": StringParam(
                    "path/to/pretrained/eeg_model",
                    doc="Path to the pre-trained EEG model (eeg_encoder_55-95_40_classes)",
                )
            }
        }

    def setup(self):
        model_path = self.params.model.model_path.value

        # Load the model configuration
        config = EEGModelConfig.from_pretrained(model_path)

        # Path to the safetensors file
        model_weights_path = os.path.join(model_path, "eeg_encoder", "model.safetensors")  # Update file name if necessary

        # Load model weights from safetensors
        self.model = ChannelNetModel(config)
        self.model.load_state_dict(load_file(model_weights_path))

    def process(self, eeg: Data):
        if eeg is None:
            return None

        eeg_data = eeg.data  # Shape: (time_points, channels)

        # Convert EEG data to PyTorch tensor and move to device
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to("cpu")

        # Extract embeddings from the EEG encoder
        with torch.no_grad():
            # assert False, str(eeg_tensor.shape)
            emb_out, _ = self.model(eeg_tensor)

        # Return the embeddings as output
        return {"embeddings": (emb_out.cpu().numpy(), {})}


from transformers import PretrainedConfig


class EEGModelConfig(PretrainedConfig):
    model_type = "eeg_channelnet"

    def __init__(
        self,
        in_channels=1,
        temp_channels=10,
        out_channels=50,
        num_classes=40,
        embedding_size=512,
        input_width=440,
        input_height=128,
        temporal_dilation_list=None,
        temporal_kernel=(1, 33),
        temporal_stride=(1, 2),
        num_temp_layers=4,
        num_spatial_layers=4,
        spatial_stride=(2, 1),
        num_residual_blocks=4,
        down_kernel=3,
        down_stride=2,
        **kwargs,
    ):
        if temporal_dilation_list is None:
            temporal_dilation_list = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16)]

        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.temp_channels = temp_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_width = input_width
        self.input_height = input_height
        self.temporal_dilation_list = temporal_dilation_list
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride
        self.num_temp_layers = num_temp_layers
        self.num_spatial_layers = num_spatial_layers
        self.spatial_stride = spatial_stride
        self.num_residual_blocks = num_residual_blocks
        self.down_kernel = down_kernel
        self.down_stride = down_stride

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Creates a config from a dictionary."""
        return cls(**config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Loads the configuration from a file."""
        config_file = f"{model_path}/config.json"
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, **kwargs)


# This is the model presented in the work: S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, M. Shah, Decoding Brain Representations by
# Multimodal Learning of Neural Activity and Visual Features,  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2020, doi: 10.1109/TPAMI.2020.2995909
import torch
import torch.nn as nn
from transformers import PreTrainedModel


class FeaturesExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.temporal_block = TemporalBlock(
            config.in_channels,
            config.temp_channels,
            config.num_temp_layers,
            config.temporal_kernel,
            config.temporal_stride,
            config.temporal_dilation_list,
            config.input_width,
        )

        self.spatial_block = SpatialBlock(
            config.temp_channels * config.num_temp_layers,
            config.out_channels,
            config.num_spatial_layers,
            config.spatial_stride,
            config.input_height,
        )

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(
                        config.out_channels * config.num_spatial_layers,
                        config.out_channels * config.num_spatial_layers,
                    ),
                    ConvLayer2D(
                        config.out_channels * config.num_spatial_layers,
                        config.out_channels * config.num_spatial_layers,
                        config.down_kernel,
                        config.down_stride,
                        0,
                        1,
                    ),
                )
                for i in range(config.num_residual_blocks)
            ]
        )

        self.final_conv = ConvLayer2D(
            config.out_channels * config.num_spatial_layers,
            config.out_channels,
            config.down_kernel,
            1,
            0,
            1,
        )

    def forward(self, x):
        out = self.temporal_block(x)

        out = self.spatial_block(out)

        if len(self.res_blocks) > 0:
            for res_block in self.res_blocks:
                out = res_block(out)

        out = self.final_conv(out)

        return out


class ChannelNetModel(PreTrainedModel):
    """The model for EEG classification.
    The imput is a tensor where each row is a channel the recorded signal and each colums is a time sample.
    The model performs different 2D to extract temporal e spatial information.
    The output is a vector of classes where the maximum value is the predicted class.
    Args:
        in_channels: number of input channels
        temp_channels: number of features of temporal block
        out_channels: number of features before classification
        num_classes: number possible classes
        embedding_size: size of the embedding vector
        input_width: width of the input tensor (necessary to compute classifier input size)
        input_height: height of the input tensor (necessary to compute classifier input size)
        temporal_dilation_list: list of dilations for temporal convolutions, second term must be even
        temporal_kernel: size of the temporal kernel, second term must be even (default: (1, 32))
        temporal_stride: size of the temporal stride, control temporal output size (default: (1, 2))
        num_temp_layers: number of temporal block layers
        num_spatial_layers: number of spatial layers
        spatial_stride: size of the spatial stride
        num_residual_blocks: the number of residual blocks
        down_kernel: size of the bottleneck kernel
        down_stride: size of the bottleneck stride
    """

    def __init__(self, config: EEGModelConfig):
        super().__init__(config=config)

        self.encoder = FeaturesExtractor(config=config)

        encoding_size = (
            self.encoder(torch.zeros(1, config.in_channels, config.input_height, config.input_width))
            .contiguous()
            .view(-1)
            .size()[0]
        )
        self.projector = nn.Linear(encoding_size, config.embedding_size)
        self.classifier = nn.Linear(config.embedding_size, config.num_classes)

    def forward(self, x):
        out = self.encoder(x)

        out = out.view(x.size(0), -1)
        emb = self.projector(out)

        cls = self.classifier(emb)

        return emb, cls


class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(True))
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, bias=True
            ),
        )
        self.add_module("drop", nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list, in_size):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        # Compute padding for each temporal layer to have a fixed size output
        # Output size is controlled by striding to be 1 / 'striding' of the original size
        for dilation in dilation_list:
            filter_size = kernel_size[1] * dilation[1] - 1
            temp_pad = math.floor((filter_size - 1) / 2) - 1 * (dilation[1] // 2 - 1)
            padding.append((0, temp_pad))

        self.layers = nn.ModuleList(
            [ConvLayer2D(in_channels, out_channels, kernel_size, stride, padding[i], dilation_list[i]) for i in range(n_layers)]
        )

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)
        return out


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()

        kernel_list = []
        for i in range(num_spatial_layers):
            kernel_list.append(((input_height // (i + 1)), 1))

        padding = []
        for kernel in kernel_list:
            temp_pad = math.floor((kernel[0] - 1) / 2)  # - 1 * (kernel[1] // 2 - 1)
            padding.append((temp_pad, 0))

        feature_height = input_height // stride[0]

        self.layers = nn.ModuleList(
            [ConvLayer2D(in_channels, out_channels, kernel_list[i], stride, padding[i], 1) for i in range(num_spatial_layers)]
        )

    def forward(self, x):
        features = []

        for layer in self.layers:
            out = layer(x)
            features.append(out)

        out = torch.cat(features, 1)

        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
