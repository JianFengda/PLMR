import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import dtype
import math


class ViTSelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_method: str = 'torch'):
        super().__init__()

        self.query = nn.Linear(dim, dim*num_heads, bias=bias).cuda()

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense = nn.Linear(dim, dim, bias=True).cuda()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dim = dim
        self.attention_head = num_heads
        self.gumbel_softmax = gumbel_softmax

    def forward(self, x1,k,c):
        q = self.query(x1)
        k = self.query(k)
        C = torch.cat(([c] * self.attention_head), 1)
        t = q+k
        x = torch.matmul(t, C.transpose(0,1))
        x = x / math.sqrt(self.attention_dim)
        x = self.softmax(x)
        x = self.gumbel_softmax(x,0.1,True)
        x = self.attention_dropout(x)
        x = torch.matmul(x.transpose(0,1), x1)
        x = self.dense(x)
        x = self.dropout(x)
        return x



class ViTBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon).cuda()
        self.attn = ViTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     init_method=init_method)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x,k,c):
        c1 = self.attn(self.norm1(x),self.norm1(k),c)
        c = c + self.drop_path(c1)
        return c


def gumbel_softmax(logits, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(in_channel,**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channel, **kwargs)

class LinearBatchNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
class PLMR(nn.Module):
    def __init__(self, head='mlp', feat_dim=128, num_class=0, dataset='cifar10'):
        super(PLMR, self).__init__()
        dim_in = 512
        model_fun = resnet18
        if dataset == 'cifar10':
            self.encoder = model_fun(3)
        else:
            self.encoder = model_fun(1)
        # self.encoder = model_fun()
        self.query_embed = nn.Embedding(num_class, feat_dim)
        self.fc = nn.Linear(dim_in, num_class)
        self.jl = ViTBlock(feat_dim, num_class)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))     

    def forward(self, x,x1=None):
        feat = self.encoder(x)
        logits = self.fc(feat)
        if x1 == None:
            return logits
        feat_c = self.head(feat)
        with torch.no_grad():
            feat1 = self.encoder(x1)
            feat_c1 = self.head(feat1)
        center = self.jl(feat_c,feat_c1,self.query_embed.weight)
        return logits, F.normalize(feat_c, dim=1),F.normalize(feat_c1, dim=1),F.normalize(center, dim=1)