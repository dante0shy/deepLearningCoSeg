import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


# Using LSTM+non_local 275-iter=4 0.686


class non_local_block(nn.Module):
    def __init__(self):
        super(non_local_block, self).__init__()

        self.mlp1 = nn.Linear(512, 4096)
        self.mlp2 = nn.Linear(4096, 512)
        self.theta = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
        )
        self.conv1x1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0
        )
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.conv_y = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0
        )
        self.conv_lastlayer = nn.Conv2d(
            in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
        )

        self.pool_y = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, featureX, featureY):
        # 加pool
        featureY = self.pool_y(featureY)

        batch_size = featureX.size(0)  # N
        channel_size = featureX.size(1)

        theta_x = self.theta(featureX).view(
            batch_size, channel_size // 2, -1
        )  # (batch,C//2,H*W)
        theta_x = theta_x.permute(0, 2, 1)  # (batch,H2*W2,C//2)

        phi_y1 = self.phi(featureY).view(
            batch_size, channel_size // 2, -1
        )  # (batch,C//2,7*7)
        f1 = torch.matmul(theta_x, phi_y1)  # (batch,H*W,7*7)
        f_div_C1 = F.softmax(f1, dim=-1)  # normalize the last dim by softmax
        featureY = featureY.view(batch_size, channel_size, -1)  # N,512,7*7
        featureY = featureY.permute(0, 2, 1)  # N,7*7,512
        y1 = torch.matmul(f_div_C1, featureY)  # batch,H*W,C
        y1 = y1.permute(0, 2, 1).contiguous()
        # y1 = y1.view(batch_size, channel_size, 16, 16).permute(0, 2, 3, 1)  # batch,16,16,512
        # # experiment12 对比下linear和1x1卷积
        # W_y1 = self.mlp2(F.tanh(self.mlp1(y1))).permute(0, 3, 1, 2)  # batch,512,16,16
        y1 = y1.view(batch_size, channel_size, 16, 16)  # batch,512,16,16

        return y1

        # out=torch.cat((W_y1,featureX),dim=1)
        # out=F.relu(self.conv_lastlayer(out))
        # # out=W_y1*featureX
        #
        # return out


# Define some constants
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        # print(classname)
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size=3, dilation=1):
        super().__init__()
        self.input_size = input_size  # input channel
        self.hidden_size = hidden_size  # hidden channel
        self.Gates = nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 3) // 2 + dilation,
        )
        self.Gates.apply(weights_init)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (torch.zeros(state_size), torch.zeros(state_size))

        prev_hidden, prev_cell = prev_state  # previous state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        # print (gates.shape)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.pretrained_model = vgg16(pretrained=True)
        self.features, self.classifiers = (
            list(self.pretrained_model.features.children()),
            list(self.pretrained_model.classifier.children()),
        )

        self.features_map = nn.Sequential(*self.features)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp1 = nn.Linear(512, 4096)
        self.mlp2 = nn.Linear(4096, 512)
        self.upsample = nn.Upsample(16)
        self.dec = Decoder(2, 512, 2, activ="relu", pad_type="reflect")

        self.lstm_cell = ConvLSTMCell(512, 512, kernel_size=3, dilation=1)
        self.conv = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )

        self.non_local_block1 = non_local_block()
        self.conv_last = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)

    def forward(self, x, y):
        vgg_x, vgg_y, vgg_x_weight, vgg_y_weight = self.encode(x, y)
        images_recon_x, images_recon_y = self.decode(vgg_x_weight, vgg_y_weight)
        return images_recon_x, images_recon_y

    def encode(self, x, y):
        vgg_x = self.features_map(x)
        vgg_y = self.features_map(y)

        x_input = self.upsample(self.global_avg_pool(vgg_y))
        y_input = self.upsample(self.global_avg_pool(vgg_x))

        hidden_state_x = vgg_x
        cell_x = vgg_x

        hidden_state_y = vgg_y
        cell_y = vgg_y

        for i in range(4):
            hidden_state_x, cell_x = self.lstm_cell(x_input, (hidden_state_x, cell_x))
            hidden_state_y, cell_y = self.lstm_cell(y_input, (hidden_state_y, cell_y))

            non_local_x = self.non_local_block1(cell_x, cell_y)
            non_local_y = self.non_local_block1(cell_y, cell_x)

            x_input = self.upsample(self.global_avg_pool(cell_y))
            x_input = (x_input + non_local_x) / 2
            y_input = self.upsample(self.global_avg_pool(cell_x))
            y_input = (y_input + non_local_y) / 2

        # vgg_x_weight = self.global_avg_pool(vgg_x)
        # vgg_x_weight = self.upsample(F.softmax(self.mlp2(F.tanh(self.mlp1(vgg_x_weight.view(-1,512)))),dim=-1).view(-1,512,1,1))
        #
        # vgg_y_weight = self.global_avg_pool(vgg_y)
        # vgg_y_weight = self.upsample(F.softmax(self.mlp2(F.tanh(self.mlp1(vgg_y_weight.view(-1,512)))),dim=-1).view(-1,512,1,1))
        return vgg_x, vgg_y, hidden_state_x, hidden_state_y

    def decode(self, vgg_x_weight, vgg_y_weight):
        vgg_x_weight = F.relu(self.conv(vgg_x_weight))
        vgg_y_weight = F.relu(self.conv(vgg_y_weight))

        images_x = self.dec(vgg_x_weight)
        images_y = self.dec(vgg_y_weight)
        return images_x, images_y


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm="bn", activation="relu", pad_type="zero"):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_res, dim, output_dim, activ="relu", pad_type="zero"):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, "bn", activ, pad_type=pad_type)]

        self.model += [
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                dim, dim // 2, 5, 1, 2, norm="bn", activation=activ, pad_type="reflect"
            ),
        ]
        dim //= 2
        self.model += [
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                dim, dim // 2, 5, 1, 2, norm="bn", activation=activ, pad_type="reflect"
            ),
        ]
        dim //= 2
        self.model += [
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                dim, dim // 2, 5, 1, 2, norm="bn", activation=activ, pad_type="reflect"
            ),
        ]
        dim //= 2
        self.model += [
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                dim, dim // 2, 5, 1, 2, norm="bn", activation=activ, pad_type="reflect"
            ),
        ]
        dim //= 2
        self.model += [
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                dim, dim // 2, 5, 1, 2, norm="bn", activation=activ, pad_type="reflect"
            ),
        ]
        dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                dim,
                output_dim,
                7,
                1,
                3,
                norm="bn",
                activation="none",
                pad_type="reflect",
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm="ln", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)

        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "ln":
            self.norm = LayerNorm(norm_dim)
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
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias
        )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
