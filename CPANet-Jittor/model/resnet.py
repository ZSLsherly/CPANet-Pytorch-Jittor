import jittor as jt
from jittor import init
from jittor import nn
import math

BatchNorm = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet50', 'resnet101']

# In Jittor, there is no direct equivalent of torch.hub or model_zoo for
# pre-trained models. You'll need to manually download the weights and load them.
# The `model_urls` dictionary is kept for reference.
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv:
    return nn.Conv(in_planes, out_planes, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes,planes,stride=1,dilation=1,downsample=None,norm_layer=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=dilation,dilation=dilation,bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.Relu()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out +=residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
            self.relu = nn.ReLU()
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.relu1 = nn.ReLU()
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.relu2 = nn.ReLU()
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
            self.relu3 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,64,layers[0],norm_layer=BatchNorm)
        self.layer2 = self._make_layer(block,128,layers[1],stride=2,norm_layer=BatchNorm)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2,norm_layer=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=BatchNorm)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1,dilation=1,norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=1,downsample=downsample,norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation=dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def resnet50(pretrained=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_path = './initmodel/resnet50_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model


def resnet101(pretrained=None, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_path = './initmodel/resnet101_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model


if __name__ == '__main__':

    resnet = resnet50(pretrained=False)  # Set pretrained to False for this test

    resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                  resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

    for n, m in resnet.layer3.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)
    for n, m in resnet.layer4.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)

    # Jittor tensor creation
    x = jt.randn(4, 3, 200, 200)
    print(x.shape)

    # Forward pass and printing shapes
    query_feat_0 = resnet.layer0(x)
    print(query_feat_0.shape)
    query_feat_1 = resnet.layer1(query_feat_0)
    print(query_feat_1.shape)
    query_feat_2 = resnet.layer2(query_feat_1)
    print(query_feat_2.shape)
    query_feat_3 = resnet.layer3(query_feat_2)
    print(query_feat_3.shape)
    query_feat_4 = resnet.layer4(query_feat_3)
    print(query_feat_4.shape)