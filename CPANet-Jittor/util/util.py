import os
import numpy as np
from PIL import Image
import jittor as jt
from jittor import nn, init

jt.flags.use_cuda = 1  # 根据需要开启 CUDA

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=4, scale_lr=10., warmup=False, warmup_step=500):
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter / warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    if curr_iter % 50 == 0:
        print(f'Base LR: {base_lr:.4f}, Curr LR: {lr:.4f}, Warmup: {warmup and curr_iter < warmup_step}.')

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr


# def intersectionAndUnion(output, target, K, ignore_index=255):
#     assert (output.ndim in [1, 2, 3])
#     assert output.shape == target.shape
#     output = output.view(-1)
#     target = target.view(-1)
#     output[target == ignore_index] = ignore_index
#     intersection = output[output == target]
#     area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
#     area_output, _ = np.histogram(output, bins=np.arange(K+1))
#     area_target, _ = np.histogram(target, bins=np.arange(K+1))
#     area_union = area_output + area_target - area_intersection
#     return area_intersection, area_union, area_target

def histc_(input, bins, min=0., max=0.):
    if min == 0 and max == 0:
        min, max = input.min(), input.max()
    assert min < max
    bin_length = (max - min) / bins
    histc = jt.floor((input[jt.logical_and(input >= min, input <= max)] - min) / bin_length)
    histc = jt.minimum(histc, bins - 1).int().reshape(-1)
    hist = jt.ones_like(histc).float().reindex_reduce("add", [bins, ], ["@e0(i0)"], extras=[histc])
    if hist.sum() != histc.shape[0]:
        hist[-1] += 1
    return hist

def intersectionAndUnion(output, target, K, ignore_index=255):
    #计算交集、并集和target区域
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = histc_(intersection,bins=K , min=0, max = K- 1)
    area_output = histc_(output, bins=K, min=0, max=K - 1)
    area_target = histc_(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection , area_union, area_target



def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = jt.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = jt.histc(output, bins=K, min=0, max=K - 1)
    area_target = jt.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    for m in model.modules():
        # Conv
        if isinstance(m, (nn.Conv, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif conv == 'xavier':
                init.xavier_normal_(m.weight)
            else:
                raise ValueError("Invalid conv init type")
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0)

        # BatchNorm
        elif isinstance(m, (nn.BatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if batchnorm == 'normal':
                init.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                init.constant_(m.weight, 1.0)
            else:
                raise ValueError("Invalid batchnorm init type")
            init.constant_(m.bias, 0)

        # Linear
        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif linear == 'xavier':
                init.xavier_normal_(m.weight)
            else:
                raise ValueError("Invalid linear init type")
            if m.bias is not None:
                init.constant_(m.bias, 0)

        # Jittor 不原生支持 LSTM
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        init.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        init.xavier_normal_(param)
                    else:
                        raise ValueError("Invalid lstm init type")
                elif 'bias' in name:
                    init.constant_(param, 0)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color
