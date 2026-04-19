# Neural Network building blocks
import torch
# Brevitas quantized equivalents of PyTorch layers
from brevitas.nn import QuantIdentity

# Brevitas quantizer configuration generator
from quant import act_quantizer, weight_quantizer
# Selection of supported activation functions shared by different models
from activations import ACTIVATIONS
# Lazy initialization versions of Brevitas layers
from lazy import LazyQuantLinear, LazyQuantConv2d
# Quantized custom implementation of multihead attention
from attention import QuantMultiheadAttention
# Tensor packing/unpacking operations in convenient Einstein notation
from einops import pack, unpack
# Einops layers for rearranging data with convenient Einstein notation
from einops.layers.torch import Rearrange


# Pointwise MLP block according to Vaswani et al. 2017, expanding the embedding
# dimension before projecting it back. Comprises a normalization layer which can
# be placed in pre- or post-norm configuration.
class MLP(torch.nn.Module):
    def __init__(
            self,
            # Embedding (output) dimension of the second linear layer
            emb_dim,
            # Expanded embedding dimension of the first linear layer
            expansion_dim,
            # Enable a bias added to the projections
            bias=True,
            # Normalization layer preceding or following the block
            norm="batch-norm",
            # Activation functions to use after the first linear layer
            activation="relu",
            # Placement of the normalization layer: pre-norm or post-norm
            norm_placement="post-norm",
            # Number of quantization bits for weights and activations (for all
            # intermediate layers)
            bits=None,
            # Dropout: probability of an element to be zeroed during training
            dropout=0.0,
            # Catches all remaining, unused configuration options...
            **_
    ):
        super().__init__()

        # Not support for other than batch normalization for now...
        assert norm in {"batch-norm", "none", None}, f"Unsupported norm: {norm}"

        # Weight quantizer configuration: Disables quantizer if bits are None
        weight_quant = (
            {"weight_bit_width": bits} if bits else {"weight_quant": None}
        )

        # MLP block of two linear layers as the main branch of the residual
        # block
        self.mlp = torch.nn.Sequential(
            # Normalization layer preceding the main branch if configured as
            # pre-norm
            *(torch.nn.Sequential(
                # Transformer data comes in with sequence (temporal) dimension
                # before channels but is treated as an image in channels-first
                # layout
                Rearrange("b ... c -> b c ..."),
                # Batch normalization inferring the size of the embedding
                # dimension
                torch.nn.LazyBatchNorm1d(affine=False),
                # Rearrange from channels-first back to channels-last
                # sequence-first layout
                Rearrange("b c ... -> b ... c"),
            ) if norm_placement == "pre-norm" and norm is not None else []),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits)] if bits else []),
            # Quantized linear projection to the expanded embedding dimension
            LazyQuantLinear(expansion_dim, bias=bias, **weight_quant),
            # Select and instantiate activation functions from the dictionary
            # defined above
            ACTIVATIONS[activation](),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits, signed=False)] if bits else []),
            # Amount of dropout to apply at the sublayer output
            torch.nn.Dropout(p=dropout),
            # Quantized linear projection to the output embedding dimension
            LazyQuantLinear(emb_dim, bias=bias, **weight_quant),
            # No output quantizer here, see below, avoid double quantizer...
            # Amount of dropout to apply at the sublayer output
            torch.nn.Dropout(p=dropout),
        )

        # The final quantizer of the two branches needs to be shared to have
        # matching quantizer scales preceding the addition
        self.quant = torch.nn.Sequential(
            *([QuantIdentity(bit_width=bits)] if bits else []),
        )

        # Default identity post-norm will be overwritten below if configured
        self.post_norm = torch.nn.Identity()

        # Normalization layer following the residual addition if configured as
        # post-norm
        if norm_placement == "post-norm" and norm is not None:
            self.post_norm = torch.nn.Sequential(
                # Transformer data comes in with sequence (temporal) dimension
                # before channels but is treated as an image in channels-first
                # layout
                Rearrange("b ... c -> b c ..."),
                # Batch normalization inferring the size of the embedding
                # dimension
                torch.nn.LazyBatchNorm1d(affine=False),
                # Rearrange from channels-first back to channels-last
                # sequence-first layout
                Rearrange("b c ... -> b ... c"),
                # Insert optional activation quantizer if enabled
                *([QuantIdentity(bit_width=bits)] if bits else []),
            )

    def forward(self, x):
        # Pack multiple sequence/spatial dimensions into a single sequence
        # dimension
        x, ps = pack([x], "b * d")
        # Quantized MLP block with residual connection and normalization
        y = self.post_norm(self.quant(self.mlp(x)) + self.quant(x))
        # Unpack the tensor to continue with the original layer
        return unpack(y, ps, "b * d")[0]


# Multihead Self Attention block according to Vaswani et al. 2017. Comprises a
# normalization layer which can be placed in pre- or post-norm configuration.
class Attention(torch.nn.Module):
    def __init__(
            self,
            # Embedding (output) dimension of the linear layers
            emb_dim,
            # Number of attention heads
            num_heads,
            # Enable a bias added to the input and output projections
            bias=False,
            # Normalization layer preceding or following the block
            norm="batch-norm",
            # Placement of the normalization layer: pre-norm or post-norm
            norm_placement="post-norm",
            # Number of quantization bits for weights and activations (for all
            # intermediate layers)
            bits=None,
            # Dropout: probability of an element to be zeroed during training
            dropout=0.0,
            # Inserts an additional input quantizer, e.g. when the block is used
            # as the first block of the model
            input_quant=False,
            # Catches all remaining, unused configuration options...
            **_
    ):
        super().__init__()

        # Not support for other than batch normalization for now...
        assert norm in {"batch-norm", "none", None}, f"Unsupported norm: {norm}"

        # Optional input quantizer in front of the entire block
        self.input_quant = torch.nn.Identity()

        if input_quant and bits is not None:
            self.input_quant = QuantIdentity(bit_width=bits)

        # Default identity pre-norm will be overwritten below if configured
        self.pre_norm = torch.nn.Identity()

        # Normalization layer preceding the attention if configured as pre-norm
        if norm_placement == "pre-norm" and norm is not None:
            self.pre_norm = torch.nn.Sequential(
                # Packed sequential/spatial data comes in channel-last layout
                # while batch normalization expects channels-first
                Rearrange("b ... c -> b c ..."),
                # Batch normalization inferring the size of the embedding
                # dimension
                torch.nn.LazyBatchNorm1d(affine=False),
                # Insert optional activation quantizer if enabled
                *([QuantIdentity(bit_width=bits)] if bits else []),
                # Rearrange from channels-first back to channels-last
                # sequence-first layout
                Rearrange("b c ... -> b ... c"),
            )

        # Block of quantized multihead attention where all quantizers share the
        # same quantization bit-width
        self.mha = QuantMultiheadAttention(
            # Size of the embedding dimension
            emb_dim=emb_dim,
            # Number of attention heads to distribute the embeddings to
            num_heads=num_heads,
            # Enable a bias added to the input and output projections
            bias=bias,
            # Amount of dropout to apply at the attention block output, i.e.,
            # after the output projection, during training
            dropout=dropout,
            # Input, weight and bias quantization settings of input projections,
            # shared by all three input projections
            input_projection_input_quant=None,
            input_projection_weight_quant=weight_quantizer(bits),
            input_projection_bias_quant=None,
            # Quantization settings of key, query and values tensors, i.e., the
            # outputs of the input projection
            k_quant=act_quantizer(bits),
            q_quant=act_quantizer(bits),
            v_quant=act_quantizer(bits),
            # Input and output quantization of the softmax normalization of the
            # attention weights
            softmax_input_quant=act_quantizer(bits),
            softmax_output_quant=act_quantizer(bits),
            # Input, weight and bias quantization settings of output projection
            output_projection_input_quant=act_quantizer(bits),
            output_projection_weight_quant=weight_quantizer(bits),
            output_projection_bias_quant=None,
            # Output quantizer of the whole attention operation following the
            # output projection
            output_quant=None,  # Avoid double quantizer
            # Return the quantization parameters so the next layer can
            # quantize the bias
            return_quant_tensor=False
        )

        # The final quantizer of the two branches needs to be shared to have
        # matching quantizer scales preceding the addition
        self.quant = torch.nn.Sequential(
            *([QuantIdentity(bit_width=bits)] if bits else []),
        )

        # Default identity post-norm will be overwritten below if configured
        self.post_norm = torch.nn.Identity()

        # Normalization layer following the residual addition if configured as
        # post-norm
        if norm_placement == "post-norm" and norm is not None:
            self.post_norm = torch.nn.Sequential(
                # Packed sequential/spatial data comes in channel-last layout
                # while batch normalization expects channels-first
                Rearrange("b ... c -> b c ..."),
                # Batch normalization inferring the size of the embedding
                # dimension
                torch.nn.LazyBatchNorm1d(affine=False),
                # Insert optional activation quantizer if enabled
                *([QuantIdentity(bit_width=bits)] if bits else []),
                # Rearrange from channels-first back to channels-last
                # sequence-first layout
                Rearrange("b c ... -> b ... c"),
            )

    def forward(self, x):
        # Pack multiple sequence/spatial dimensions into a single sequence
        # dimension
        x, ps = pack([self.input_quant(x)], "b * d")
        # Apply pre-norm normalization once on the query, key and value input
        # before forking
        y = self.pre_norm(x)
        # Compute the self-attention operation on packed tensors
        y = self.mha(y, y, y)
        # Quantized residual addition and post-norm normalization
        y = self.post_norm(self.quant(y) + self.quant(x))
        # Unpack the tensor to continue with the original layer
        return unpack(y, ps, "b * d")[0]


# Convolutional block according to Gulati et al. 2020. Comprises a normalization
# layer which can be placed in pre- or post-norm configuration.
class Conv(torch.nn.Module):
    def __init__(
            self,
            # Embedding (output) dimension of output convolution layer
            emb_dim,
            # Expanded embedding dimension of the first convolution layer
            expansion_dim,
            # Kernel size of the depthwise convolution (second convolution)
            kernel_size=(1, 7),
            # Padding of the depthwise convolution
            padding=(0, 3),
            # Enable a bias added to the projections
            bias=True,
            # Normalization layer preceding or following the block
            norm="batch-norm",
            # Activation functions to use after the second convolution layer
            activation="relu",
            # Placement of the normalization layer: pre-norm or post-norm
            norm_placement="post-norm",
            # Number of quantization bits for weights and activations (for all
            # intermediate layers)
            bits=None,
            # Dropout: probability of an element to be zeroed during training
            dropout=0.0,
            # Catches all remaining, unused configuration options...
            **_
    ):
        super().__init__()

        # Not support for other than batch normalization for now...
        assert norm in {"batch-norm", "none", None}, f"Unsupported norm: {norm}"

        # Weight quantizer configuration: Disables quantizer if bits are None
        weight_quant = (
            {"weight_bit_width": bits} if bits else {"weight_quant": None}
        )

        # Gated Linear Unit (GLU) activation function reducing the expanded
        # embedding dimension in half
        class GLU(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # Quantizer following the sigmoid activation in the gating
                # branch
                self.quant = QuantIdentity(act_quantizer(bits))

            def forward(self, x):
                # Split the tensor along the expanded embedding dimension into
                # two equally sized parts: The gating and the gated part
                x0, x1 = torch.split(x, expansion_dim // 2, dim=1)
                # GLU gates (multiplication) the second part of the split by the
                # sigmoid activation on the first part of the split. Quantize
                # the Sigmoid and the final output after multiplication.
                return x0 * self.quant(torch.sigmoid(x1))

        # CNN block of three convolution layers as the main branch of the
        # residual block
        self.cnn = torch.nn.Sequential(
            # Transformer data comes in with sequence (temporal) dimension
            # before channels but is treated as an image in channels-first
            # layout
            Rearrange("b ... c -> b c ..."),
            # Normalization layer preceding the main branch if configured as
            # pre-norm
            *(torch.nn.Sequential(
                # Batch normalization inferring the size of the embedding
                # dimension
                torch.nn.LazyBatchNorm2d(),
            ) if norm_placement == "pre-norm" and norm is not None else []),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits)] if bits else []),
            # First quantized pointwise convolution to the expanded embedding
            # dimension
            LazyQuantConv2d(
                expansion_dim, bias=bias, kernel_size=(1, 1), **weight_quant
            ),
            # Gate Linear Unit reducing the embedding dimension by half
            GLU(),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits)] if bits else []),
            # Quantized depthwise convolution to the output embedding dimension
            LazyQuantConv2d(
                # Output the embedding dimension
                emb_dim,
                # Enable/disable the bias following the convolution
                bias=bias,
                # Configurable kernel size: Has a default (1, 7) above with
                # matching default padding
                kernel_size=kernel_size,
                # Configurable padding: Without padding this might reduce the
                # feature map size
                padding=padding,
                # Set the groups to the embedding dimension to turn this into
                # the depth-wise convolution, otherwise it is just a normal kx1
                # convolution
                groups=emb_dim,
                # Inject weight quantizer configuration
                **weight_quant
            ),
            # Normalization between depthwise convolution and activation
            # function - this is always a batch norm and not configurable
            torch.nn.LazyBatchNorm2d(),
            # Select and instantiate activation functions from the dictionary
            # defined above
            ACTIVATIONS[activation](),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits, signed=False)] if bits else []),
            # Second quantized pointwise convolution to the embedding dimension
            LazyQuantConv2d(
                emb_dim, bias=bias, kernel_size=(1, 1), **weight_quant
            ),
            # No output quantizer here, see below, avoid double quantizer...
            # Amount of dropout to apply at the sublayer output
            torch.nn.Dropout(p=dropout),
            # Rearrange from channels-first back to channels-last
            # sequence-first layout
            Rearrange("b c ... -> b ... c"),
        )

        # The final quantizer of the two branches needs to be shared to have
        # matching quantizer scales preceding the addition
        self.quant = torch.nn.Sequential(
            *([QuantIdentity(bit_width=bits)] if bits else []),
        )

        # Default identity post-norm will be overwritten below if configured
        self.post_norm = torch.nn.Identity()

        # Normalization layer following the residual addition if configured as
        # post-norm
        if norm_placement == "post-norm" and norm is not None:
            self.post_norm = torch.nn.Sequential(
                # Transformer data comes in with sequence (temporal) dimension
                # before channels but is treated as an image in channels-first
                # layout
                Rearrange("b ... c -> b c ..."),
                # Batch normalization inferring the size of the embedding
                # dimension
                torch.nn.LazyBatchNorm2d(),
                # Rearrange from channels-first back to channels-last
                # sequence-first layout
                Rearrange("b c ... -> b ... c"),
                # Insert optional activation quantizer if enabled
                *([QuantIdentity(bit_width=bits)] if bits else []),
            )

    def forward(self, x):
        return self.post_norm(self.quant(self.cnn(x)) + self.quant(x))


# Convolutional downsampling layer according to ... me? Comprises a quantized
# convolution with stride equal to kernel size to reduce the feature map size.
class Downsample(torch.nn.Module):
    def __init__(
            self,
            # Embedding (output) dimension of output convolution layer
            emb_dim,
            # Kernel size of the downsampling convolution
            kernel_size=(1, 2),
            # Padding of the convolution
            padding=(0, 0),
            # Enable a bias added to the projections
            bias=True,
            # Activation functions to use after the second convolution layer
            activation="relu",
            # Number of quantization bits for weights and activations (for all
            # intermediate layers)
            bits=None,
            # Catches all remaining, unused configuration options...
            **_
    ):
        super().__init__()

        # Weight quantizer configuration: Disables quantizer if bits are None
        weight_quant = (
            {"weight_bit_width": bits} if bits else {"weight_quant": None}
        )

        # Convolution downsampling followed by a quantized activation function
        self.conv = torch.nn.Sequential(
            # Transformer data comes in with sequence (temporal) dimension
            # before channels but is treated as an image in channels-first
            # layout
            Rearrange("b ... c -> b c ..."),
            # Quantized convolution with stride matching kernel size to reduce
            # the feature map size
            LazyQuantConv2d(
                # Output the embedding dimension
                emb_dim,
                # Enable/disable the bias following the convolution
                bias=bias,
                # Configurable kernel size: Has a default (1, 2) above with
                # matching default padding
                kernel_size=kernel_size,
                # Make stride equal to kernel size to reduce the feature map
                # size by the factor of the kernel size in each dimension
                stride=kernel_size,
                # Configurable padding: Without padding this might reduce the
                # feature map size by more than the kernel size factor
                padding=padding,
                # Inject weight quantizer configuration
                **weight_quant
            ),
            # Normalization between convolution and activation function - this
            # is always a batch norm and not configurable
            torch.nn.LazyBatchNorm2d(),
            # Select and instantiate activation functions from the dictionary
            # defined above
            ACTIVATIONS[activation](),
            # Insert optional activation quantizer if enabled
            *([QuantIdentity(bit_width=bits, signed=False)] if bits else []),
            # Rearrange from channels-first back to channels-last
            # sequence-first layout
            Rearrange("b c ... -> b ... c"),
        )

    def forward(self, x):
        return self.conv(x)


# Dictionary for looking up the configurable layer blocks by name
BLOCKS = {
    "attention": Attention, "mlp": MLP, "conv": Conv, "downsample": Downsample
}

# Original configuration of the Transformer-encoder according to Vaswani et al.
# 2017
ORIGINAL = ("attention", "mlp")
# Conformer configuration of the Transformer-encoder according to Gulati et al.
# 2020
CONFORMER = ("mlp", "attention", "conv", "mlp")

# Maps string identifiers to Transformer block configurations
TRANSFORMER_CONFIGURATIONS = {"original": ORIGINAL, "conformer": CONFORMER}
