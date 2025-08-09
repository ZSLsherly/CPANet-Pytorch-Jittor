import jittor as jt
from jittor.nn import init
from jittor import nn
import model.resnet as models
import model.vgg as vgg_models
# from model.CPP import CPP
# from model.DP import DoneUp

class channel_attention(nn.Module):

    def __init__(self, channel=256, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, (channel // ratio), False), nn.ReLU(), nn.Linear((channel // ratio), channel, False))
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        b, c, h, w = x.shape
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
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

class SSA(nn.Module):

    def __init__(self):
        super(SSA, self).__init__()
        reduce_channels = 256
        self.Done1 = nn.Sequential(nn.Conv(reduce_channels, reduce_channels, (2, 2), stride=2, padding=0, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.Done2 = nn.Sequential(nn.Conv(reduce_channels, reduce_channels, (2, 2), stride=2, padding=0, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.conv_cat = nn.Sequential(nn.Conv((reduce_channels * 3), reduce_channels, (1, 1), padding=0, bias=False), nn.ReLU(), nn.Dropout(p=0.1), nn.Conv(reduce_channels, reduce_channels, (3, 3), padding=1, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.res_conv = nn.Sequential(nn.Conv(256, 256, (3, 3), padding=1, bias=False), nn.ReLU(), nn.Dropout(p=0.1))
        self.cls = nn.Sequential(nn.Conv(256, 2, (1, 1)))
        self.Cbam = Cbam(256)
        self.init_weight()

    def execute(self, x):
        x1 = self.Done1(x)
        x1_up = nn.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.Done2(x1)
        x2_up = nn.interpolate(x2, scale_factor=4, mode='bilinear', align_corners=True)
        x_cat = jt.concat([x, x1_up, x2_up], dim=1)
        x_x = self.conv_cat(x_cat)
        x_x_r = self.res_conv(x_x)
        x_atten = self.Cbam(x_x)
        out = x_x_r + x_atten
        out = self.cls(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CPP(nn.Module):

    def __init__(self, in_channels, sub_sample=True, bn_layer=True):
        super(CPP, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv(self.in_channels, self.inter_channels, (1, 1), stride=(1, 1), padding=0)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        # self.cos_similarity = nn.CosineSimilarity()
        if bn_layer:
            self.W = nn.Sequential(nn.Conv(self.inter_channels, self.in_channels, (1, 1), stride=(1, 1), padding=0), nn.BatchNorm2d(self.in_channels))
            init.constant_(self.W[1].weight, value=0)
            init.constant_(self.W[1].bias, value=0)
        else:
            self.W = nn.Conv(self.inter_channels, self.in_channels, (1, 1), stride=(1, 1), padding=0)
            init.constant_(self.W.weight, value=0)
            init.constant_(self.W.bias, value=0)
        self.theta = nn.Conv(self.in_channels, self.inter_channels, (1, 1), stride=(1, 1), padding=0)
        self.phi = nn.Conv(self.in_channels, self.inter_channels, (1, 1), stride=(1, 1), padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d((2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d((2, 2)))

    def execute(self, x):
        # '\n        :param x: (b, c, t, h, w)\n        :return:\n        '
        batch_size = x.shape[0]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = nn.matmul(theta_x, phi_x)
        f_div_C = nn.softmax(f, dim=(- 1))
        y = nn.matmul(f_div_C, g_x)
        y = y.permute(0,2,1)
        y = y.view(batch_size, self.inter_channels, *x.shape[2:])
        W_y = self.W(y)
        z = (W_y + x)
        x3 = self.avg_pool(z)
        return x3

def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class cpanet(nn.Module):
    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True, shot=1, vgg=False,args=None):
        super(cpanet, self).__init__()
        # assert layers in [50, 101, 152]
        # assert classes == 2

        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg

        models.BatchNorm = nn.BatchNorm2d

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg_models.BatchNorm = nn.BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('>>>>>>>>> Using ResNet {}<<<<<<<<<'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256

        if self.vgg:
            fea_dim = 512 + 256

        else:
            fea_dim = 1024 + 512


        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, 2, kernel_size=3, padding=1, bias=False),
        )


        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )

        self.down_support = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.conv_Fsq = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )
        self.conv_queryMask = nn.Sequential(
            nn.Conv2d(reduce_dim , reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )
        self.conv_supportMask = nn.Sequential(
            nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )

        self.SSA = SSA()

        self.CPP = CPP(reduce_dim)

    def execute(self,x,s_x,s_y,y=None):

        x_size = x.size()  # [4,3,200,200]
        h = int(x_size[-1])
        w = int(x_size[-2])


        with jt.no_grad():
            query_feat_0 = self.layer0(x)  # [4,128,50,50]
            query_feat_1 = self.layer1(query_feat_0)  # [4,256,50,50]
            query_feat_2 = self.layer2(query_feat_1)  # [4,512,25,25]
            query_feat_3 = self.layer3(query_feat_2)  # [4,1024,25,25]

            if self.vgg:
                query_feat_2 = nn.interpolate(query_feat_2,
                                             size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear',
                                             align_corners=True)
        query_feat = jt.concat([query_feat_3, query_feat_2], dim=1)
        query_feat = self.down_query(query_feat)
        gt_list = []
        gt_down_list = []
        supp_feat_list = []
        supp_a_list = []
        # ----- 5shot ----- #
        for i in range(self.shot):
            supp_gt = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            gt_list.append(supp_gt)

            with jt.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)

                if self.vgg:
                    supp_feat_2 = nn.interpolate(supp_feat_2,
                                                size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear',
                                                align_corners=True)
            supp_gt_down = nn.interpolate(supp_gt, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',align_corners=True)
            gt_down_list.append(supp_gt_down)

            supp_feat = jt.concat([supp_feat_3, supp_feat_2], dim=1)
            supp_feat = self.down_support(supp_feat)  # [4,256,25,25]
            supp_feat_list.append(supp_feat)


            supp_feat_mask = supp_feat * supp_gt_down
            supp_a = self.CPP(supp_feat_mask)  # [1,256,1,1]

            supp_a_list.append(supp_a)

        supp_i = jt.zeros_like(supp_a_list[0])   # 【1，256，1，1】
        for i in range(self.shot):
            supp_i += supp_a_list[i]
        supp_ap = supp_i/len(supp_i)
        supp_ap = supp_ap.expand(query_feat.shape[0], 256, query_feat.shape[-2], query_feat.shape[-1])

        query_supp = jt.concat([supp_ap, query_feat], dim=1)
        query_out = self.conv_Fsq(query_supp)  # 1,256,200,200

        query_pred_mask = self.conv_queryMask(query_out)
        query_pred_mask = nn.interpolate(query_pred_mask, size=(h, w), mode='bilinear', align_corners=True)  # [1,256,200,200]

        query_pred_mask = self.SSA(query_pred_mask)
        query_pred_mask_save = jt.argmax(query_pred_mask[0].permute(1, 2, 0), dim=-1)[0].detach().numpy()
        query_pred_mask_save[query_pred_mask_save!=0] = 255
        query_pred_mask_save[query_pred_mask_save==0] = 0

        supp_pred_mask_list = []
        if self.is_training():
            for i in range(self.shot):
                supp_s_i = supp_a_list[i]
                supp_feat_i = supp_feat_list[i]
                supp_s = supp_s_i.expand(supp_feat_i.shape[0], 256, supp_feat_i.shape[-2], supp_feat_i.shape[-1])
                supp_gcn = jt.concat([supp_feat_i, supp_s, supp_s] ,dim=1)
                supp_out = self.conv_supportMask(supp_gcn)
                supp_out = nn.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)  # [1,256,200,200]

                supp_pred_mask = self.cls(supp_out)
                supp_pred_mask_list.append(supp_pred_mask)

        alpah = 0.6
        if self.is_training():
            supp_loss_list = []
            loss = 0.
            for i in range(self.shot):
                supp_loss = self.criterion(supp_pred_mask_list[i], gt_list[i].squeeze(1).long())
                loss += supp_loss
            aux_loss = loss/self.shot
            main_loss = self.criterion(query_pred_mask, y.long())
            # for j in range(10):
            #     for k in range(10):
            #         print(f"{query_pred_mask[0,0,j,k]} {query_pred_mask[0,1,j,k]}")

            return jt.argmax(query_pred_mask, dim=1)[0], main_loss + alpah * aux_loss
        else:
            return query_pred_mask, query_pred_mask_save

if __name__ == '__main__':
    model = cpanet(vgg=True) # 以 vgg 为例
    total = sum(p.numel() for p in model.parameters())

    # 创建所有需要的假数据
    x = jt.randn(4, 3, 200, 200)
    s_x = jt.randn(4, 1, 3, 200, 200) # 假设 shot=1
    s_y = jt.randint(0, 2, (4, 1, 200, 200))

    # 切换到评估模式
    model.eval()

    # forward 函数在 eval 模式下返回两个值
    query_pred, query_pred_save = model(x, s_x, s_y)

