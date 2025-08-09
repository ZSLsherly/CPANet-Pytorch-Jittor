import jittor as jt
from jittor import init
from jittor import nn

class CPP(nn.Module):

    def __init__(self, in_channels, sub_sample=True, bn_layer=True):
        super(CPP, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = (in_channels // 2)
        self.g = nn.Conv(self.in_channels, self.inter_channels, (1, 1), stride=(1, 1), padding=0)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.peak_pool = nn.AdaptiveMaxPool2d(1)
        # self.cos_similarity = nn.CosineSimilarity()
        if bn_layer:
            self.W = nn.Sequential(nn.Conv(self.inter_channels, self.in_channels, (1, 1), stride=(1, 1), padding=0), nn.BatchNorm(self.in_channels))
            init.constant_(self.W[1].weight, value=0)
            init.constant_(self.W[1].bias, value=0)
        else:
            self.W = nn.Conv(self.inter_channels, self.in_channels, (1, 1), stride=(1, 1), padding=0)
            init.constant_(self.W.weight, value=0)
            init.constant_(self.W.bias, value=0)
        self.theta = nn.Conv(self.in_channels, self.inter_channels, (1, 1), stride=(1, 1), padding=0)
        self.phi = nn.Conv(self.in_channels, self.inter_channels, (1, 1), stride=(1, 1), padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.Pool((2, 2), op='maximum'))
            self.phi = nn.Sequential(self.phi, nn.Pool((2, 2), op='maximum'))

    def execute(self, x):
        '\n        :param x: (b, c, t, h, w)\n        :return:\n        '
        batch_size = x.shape[0]
        g_x = self.g(x).reshape(batch_size, self.inter_channels, - 1)
        g_x = g_x.transpose(2, 1)
        theta_x = self.theta(x).reshape(batch_size, self.inter_channels, (- 1))
        theta_x = theta_x.transpose(2, 1)
        phi_x = self.phi(x).reshape(batch_size, self.inter_channels, (- 1))
        f = jt.matmul(theta_x, phi_x)
        f_div_C = nn.softmax(f, dim=(- 1))
        y = jt.matmul(f_div_C, g_x)
        y = y.transpose(2, 1)
        y = y.reshape(batch_size, self.inter_channels, *x.shape[2:])
        W_y = self.W(y)
        z = (W_y + x)
        x3 = self.avg_pool(z)
        return x3

if (__name__ == '__main__'):
    img = jt.ones(4, 256, 25, 25)
    net = CPP(256)
    out1 = net(img)
    print(out1.shape)
    total = sum((p.numel() for p in net.parameters()))
    print(('Total params: %.4fM' % (total / 1000000.0)))
