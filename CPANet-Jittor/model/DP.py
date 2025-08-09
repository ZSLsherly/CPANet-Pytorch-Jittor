import jittor as jt
from jittor import init
from jittor import nn
import math
from model.CBAM import Cbam

class DoneUp(nn.Module):

    def __init__(self):
        super(DoneUp, self).__init__()
        reduce_channels = 256
        self.Done1 = nn.Sequential(nn.Conv(reduce_channels, reduce_channels, (2, 2), stride=2, padding=0, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.Done2 = nn.Sequential(nn.Conv(reduce_channels, reduce_channels, (2, 2), stride=2, padding=0, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.conv_cat = nn.Sequential(nn.Conv((reduce_channels * 3), reduce_channels, (1, 1), padding=0, bias=False), nn.ReLU(), nn.Dropout(p=0.1), nn.Conv(reduce_channels, reduce_channels, (3, 3), padding=1, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.res_conv = nn.Sequential(nn.Conv(256, 256, (3, 3), padding=1, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.cls = nn.Sequential(nn.Conv(256, 2, (1, 1)))
        self.Cbam = Cbam(256)
        self._init_weight()

    def execute(self, x):
        x1 = self.Done1(x)
        x1_up = nn.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.Done2(x1)
        x2_up = nn.interpolate(x2, scale_factor=4, mode='bilinear', align_corners=True)
        x_cat = jt.contrib.concat([x, x1_up, x2_up], dim=1)
        x_x = self.conv_cat(x_cat)
        x_x_r = self.res_conv(x_x)
        x_atten = self.Cbam(x_x)
        out = x_x_r + x_atten
        out = self.cls(out)
        return out

    def _init_weight(self):
        # 推荐使用 self.apply 方法来应用初始化函数
        self.apply(self._init_single_module_weight)

    def _init_single_module_weight(self, m):
        if isinstance(m, nn.Conv):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm):
            # Jittor 中对 BatchNorm 的权重和偏置进行初始化
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

if (__name__ == '__main__'):
    model = DoneUp()
    total = sum((p.numel() for p in model.parameters()))
    print(('Total params: %.2fM' % (total / 1000000.0)))
    input = jt.randn(4, 256, 200, 200)
    print('input_size:', input.shape)
    out = model(input)
    print('out', out.shape)
