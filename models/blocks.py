import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import init

class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm="in",
        activation="elu",
        pad_type="zero",
        upsampling=False,
        stride = 1,
        snorm=False
    ):
        super(ResBlock, self).__init__()
        self.norm = norm
        self.model = nn.ModuleList()

        if upsampling:
            self.conv1 = Conv2dTransposeBlock(
                dim_in,
                dim_out,
                3,
                2,
                1,
                norm=self.norm,
                activation=activation,
                snorm= snorm
            )

            self.conv2 = Conv2dBlock(
                dim_out,
                dim_out,
                3,
                1,
                1,
                norm=self.norm,
                activation="none",
                pad_type=pad_type,
                snorm=snorm
            )
        else:
            self.conv1 = Conv2dBlock(
                dim_in,
                dim_out,
                3,
                stride,
                1,
                norm=self.norm,
                activation=activation,
                pad_type=pad_type,
                snorm=snorm
            )

            self.conv2 = Conv2dBlock(
                dim_out,
                dim_out,
                3,
                1,
                1,
                norm=self.norm,
                activation="none",
                pad_type=pad_type,
                snorm=snorm
            )

        self.convolve_res = dim_in != dim_out or upsampling or stride != 1
        if self.convolve_res:
            if not upsampling:
                self.res_conv = Conv2dBlock(dim_in,dim_out,3,stride,1,
                                        norm="in",
                                        activation=activation,
                                        pad_type=pad_type,
                                        snorm=snorm)
            else:
                self.res_conv = Conv2dTransposeBlock(dim_in,dim_out,3,2,1,
                                        norm="in",
                                        activation=activation,
                                        snorm=snorm)


    def forward(self, x,adain_params=None):
        residual = x
        if self.convolve_res:
            residual = self.res_conv(residual)
        out = self.conv1(x,adain_params)
        out = self.conv2(out,adain_params)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="elu",
        pad_type="zero",
        use_bias=True,
        activation_first=False,
        snorm=False
    ):
        super().__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "group":
            self.norm = nn.GroupNorm(num_channels=norm_dim,num_groups=16)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if snorm:
            self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x, adain_params=None):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
        else:
            x = self.conv(self.pad(x))
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
            if self.activation:
                x = self.activation(x)
        return x

class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Conv2dTransposeBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        ks,
        st,
        padding=0,
        norm="none",
        activation="elu",
        use_bias=True,
        activation_first=False,
        snorm=False
    ):
        super().__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "group":
            self.norm = nn.GroupNorm(num_channels=norm_dim,num_groups=16)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "elu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        if snorm:
            self.conv = spectral_norm(nn.ConvTranspose2d(in_dim, out_dim, ks, st, bias=self.use_bias, padding=padding, output_padding=padding))
        else:
            self.conv = nn.ConvTranspose2d(in_dim, out_dim, ks, st, bias=self.use_bias, padding=padding,output_padding=padding)

    def forward(self, x, adain_params=None):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(x)
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
        else:
            x = self.conv(x)
            if self.norm and not isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x)
            elif isinstance(self.norm,AdaptiveInstanceNorm2d):
                x = self.norm(x, adain_params)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, adain_params):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            adain_params["weight"],
            adain_params["bias"],
            True,
            self.momentum,
            self.eps,
        )
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class AdaINLinear(nn.Module):
    def     __init__(self, in_units, target_units, use_bias=True, actfn=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_units, 2 * target_units, bias=use_bias)
        self.act_fn = actfn()

    def forward(self, x):
        out = self.act_fn(self.linear(x))
        out = {
            "weight": out[:, : out.size(1) // 2],
            "bias": out[:, out.size(1) // 2 :],
        }
        return out





class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size,upsample=False):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.upsample = upsample
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        if self.upsample:
            self.up_gate = nn.ConvTranspose2d(input_size,input_size,kernel_size,2,padding=padding, output_padding=padding)


        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)
        if self.upsample:
            init.orthogonal_(self.up_gate.weight)
            init.constant_(self.up_gate.bias, 0.)

    def forward(self, input_, prev_state):

        if self.upsample:
            input_ = self.up_gate(input_)

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers, upsampling:list=None):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()
        if upsampling is None:
            upsampling = [False]*n_layers

        self.input_size = input_size
        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        self.cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            self.cells.append(ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i],upsample=upsampling[i]))

        self.cells = nn.Sequential(*self.cells)

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (layer, batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        if hidden is None:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


# taken from official NVLabs implementation

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, config):
        super().__init__()

        param_free_norm_type = config["base_norm_spade"] if "base_norm_spade" in config else "instance"
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out