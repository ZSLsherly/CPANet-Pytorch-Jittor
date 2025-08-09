import jittor as jt
from jittor import init
from jittor import nn

class channel_attention(nn.Module):

    def __init__(self, channel=256, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, (channel // ratio), False), nn.ReLU(), nn.Linear((channel // ratio), channel, False))
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        (b, c, h, w) = x.shape
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = (max_fc_out + avg_fc_out)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return (out * x)

class spacial_attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()
        padding = (7 // 2)
        self.conv = nn.Conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        max_pool_out = jt.max(x, dim=1, keepdims=True)
        mean_pool_out = jt.mean(x, dim=1, keepdims=True)
        pool_out = jt.concat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return (out * x)

class Cbam(nn.Module):

    def __init__(self, channel=256, ratio=16, kernel_size=7):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        self.spacial_attention = spacial_attention(kernel_size)

    def execute(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x

if (__name__ == '__main__'):
    model = Cbam()
    total = sum((p.numel() for p in model.parameters()))
    print(('Total params: %.2fM' % (total / 1000000.0)))
    input = jt.ones(4, 256, 200, 200)
    print('input_size:', input.shape)
    out = model(input)
    print('out', out.shape)