import jittor as jt
from jittor import init
import argparse
import logging
import os
import random
import time
from datetime import datetime
import cv2
import numpy as np
from jittor import nn
from tensorboardX import SummaryWriter
from model.CPANet import cpanet
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from util import config, dataset, transform

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


def to_item(x):
    """把 Jittor Var / numpy 数组 转为 python float"""
    if isinstance(x, jt.Var):
        return float(x.numpy())
    elif isinstance(x, np.ndarray):
        return float(x)
    return x

def get_parser():
    parser = argparse.ArgumentParser(description='Few-Shot Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/SSD/fold0_vgg16.yaml', help='config file')
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = '[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed((args.manual_seed + worker_id))


def main_process():
    return True


def main():
    args = get_parser()
    assert (args.classes > 1)
    if (args.manual_seed is not None):
        # Jittor equivalent of setting random seeds
        jt.set_global_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)

    main_worker(args)




def main_worker(argss):
    global args
    args = argss

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model = cpanet(layers=args.layers, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=args.ignore_label), pretrained=True,
                   shot=args.shot, vgg=args.vgg,args=args)

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False

    optimizer = jt.optim.SGD(
        [
            {'params': model.down_query.parameters()},
            {'params': model.down_support.parameters()},
            {'params': model.CPP.parameters()},
            {'params': model.cls.parameters()},
            {'params': model.conv_Fsq.parameters()},
            {'params': model.conv_queryMask.parameters()},
            {'params': model.SSA.parameters()},
            {'params': model.conv_supportMask.parameters()},
        ],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info('\x1b[1;36m >>>>>>Creating model ...\x1b[0m')
    logger.info('\x1b[1;36m >>>>>>Classes: {}\x1b[0m'.format(args.classes))
    logger.info(model)
    print(args)
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = jt.load(args.weight)
            model.load_parameters(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = jt.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_parameters(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [(item * value_scale) for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [(item * value_scale) for item in std]
    assert (args.split in [0, 1, 2])
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    # train_transform = [transform.Resize(size=args.val_size),transform.RandomHorizontalFlip(), transform.ToTensor(), transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.train_list,
                                 transform=train_transform, mode='train')
    train_loader = jt.dataset.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, drop_last=True)
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose(
                [transform.Resize(size=args.val_size), transform.ToTensor(), transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([transform.test_Resize(size=args.val_size), transform.ToTensor(),
                                               transform.Normalize(mean=mean, std=std)])
        val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.val_list,
                                   transform=val_transform, mode='val')
        val_loader = jt.dataset.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers)
    max_iou = 0.0
    max_fbiou = 0
    best_epoch = 0
    filename = 'CPANet.pth'


    # 进行文件记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.save_path, f"train_val_log_{timestamp}.txt")
    log_file = open(log_path, "w")  # 写表头
    log_file.write("# Training and Validation Metrics Log\n")
    log_file.write("# None 代表该轮没有进行验证\n\n")
    log_file.write("Epoch,Train_Loss,Train_mIoU,Train_mAcc,Train_AllAcc,"
                   "Val_Loss,Val_mIoU,Val_mAcc,Val_AllAcc,Val_Class_mIoU\n")


    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            jt.set_global_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, epoch,optimizer)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
        if args.evaluate:
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if class_miou > max_iou:
                max_iou = class_miou
                best_epoch = epoch
                if os.path.exists(filename):
                        os.remove(filename)
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou)+ '.pkl'
                logger.info(('Saving checkpoint to: ' + filename))
                jt.save({'epoch': epoch, 'state_dict': model.state_dict()},
                           filename)
            if (mIoU_val > max_fbiou):
                max_fbiou = mIoU_val
            logger.info('Best Epoch {:.1f}, Best IOU {:.4f} Best FB-IoU {:4F}'.format(best_epoch, max_iou, max_fbiou))

            log_file.write(
                f"Epoch {epoch_log}: "
                f"{to_item(loss_train):.4f},{to_item(mIoU_train):.4f},{to_item(mAcc_train):.4f},{to_item(allAcc_train):.4f},"
                f"{to_item(loss_val) if to_item(loss_val) is not None else 'None'},"
                f"{to_item(mIoU_val) if to_item(mIoU_val) is not None else 'None'},"
                f"{to_item(mAcc_val) if to_item(mAcc_val) is not None else 'None'},"
                f"{to_item(allAcc_val) if to_item(allAcc_val) is not None else 'None'},"
                f"{to_item(class_miou) if to_item(class_miou) is not None else 'None'}\n"
            )
            log_file.flush()

    log_file.close()
    filename = (args.save_path + '/final.pkl')
    logger.info(('Saving checkpoint to: ' + filename))
    jt.save({'epoch': args.epochs, 'state_dict': model.state_dict()}, filename)


def train(train_loader, model, epoch,optimizer):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<<')
    multiprocessing_distributed = False
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = (args.epochs * len(train_loader))
    for (i, (input, target, s_input, s_mask, subcls, _)) in enumerate(train_loader):
        data_time.update((time.time() - end))
        current_iter = (((epoch * len(train_loader)) + i) + 1)

        # input,s_input,s_mask,target=input.stop_grad(),s_input.stop_grad(),s_mask.stop_grad(),target.stop_grad()

        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer,args.base_lr,current_iter,max_iter,power=args.power,
                               warmup=args.warmup,warmup_step=len(train_loader) // 2)

        output, main_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        if (not multiprocessing_distributed):
            main_loss = jt.mean(main_loss)
        optimizer.step(main_loss)
        loss = main_loss

        n = input.shape[0]
        (intersection, union, target) = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        (intersection, union, target) = (intersection.numpy(), union.numpy(), target.numpy())
        (intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target))
        accuracy = (sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10))
        main_loss_meter.update(main_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update((time.time() - end))
        end = time.time()
        remain_iter = (max_iter - current_iter)
        remain_time = (remain_iter * batch_time.avg)
        (t_m, t_s) = divmod(remain_time, 60)
        (t_h, t_m) = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if ((((i + 1) % 10) == 0) and main_process()):
            logger.info(
                'Epoch: [{}/{}][{}/{}] Data {data_time.val:.3f} ({data_time.avg:.3f}) Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) Remain {remain_time} MainLoss {main_loss_meter.val:.4f} AuxLoss {aux_loss_meter.val:.4f} Loss {loss_meter.val:.4f} Accuracy {accuracy:.4f}.'.format(
                    (epoch + 1), args.epochs, (i + 1), len(train_loader)//args.batch_size, batch_time=batch_time, data_time=data_time,
                    remain_time=remain_time, main_loss_meter=main_loss_meter, aux_loss_meter=aux_loss_meter,
                    loss_meter=loss_meter, accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean((intersection / (union + 1e-10))), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean((intersection / (target + 1e-10))), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)
    iou_class = (intersection_meter.sum / (union_meter.sum + 1e-10))
    accuracy_class = (intersection_meter.sum / (target_meter.sum + 1e-10))
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
                                                                                           mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou： {:.4f} - accuracy： {:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Train <<<<<<<<<<<<<<<<<')
    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    split_gap = 4
    class_intersection_meter = ([0] * split_gap)
    class_union_meter = ([0] * split_gap)

    if ((args.manual_seed is not None) and args.fix_random_seed_val):
        jt.set_global_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)
    model.eval()
    end = time.time()
    test_num = len(val_loader)
    assert ((test_num % args.batch_size_val) == 0)
    iter_num = 0
    total_time = 0
    for (i, (input, target, s_input, s_mask, subcls, ori_label)) in enumerate(val_loader):
        iter_num += 1
        data_time.update(time.time() - end)
        start_time = time.time()
        output, _ = model(s_x=s_input, s_y=s_mask, x=input, y=target)

        total_time = (total_time + 1)
        model_time.update((time.time() - start_time))

        if args.ori_resize:
            longerside = max(ori_label.shape[1], ori_label.shape[2])
            backmask = jt.ones(ori_label.shape[0], longerside, longerside)*255
            backmask[0, :ori_label.shape[1], :ori_label.shape[2]] = ori_label[0]
            target = backmask.clone().long()
        output = nn.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)
        loss = jt.mean(loss)
        output=jt.argmax(output, dim=1)[0]
        (intersection, union, new_target) = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        (intersection, union, target, new_target) = (
        intersection.numpy(), union.numpy(), target.numpy(), new_target.numpy())
        (intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target))
        subcls = subcls[0].numpy()[0]
        class_intersection_meter[((subcls - 1) % split_gap)] += intersection[1]
        class_union_meter[((subcls - 1) % split_gap)] += union[1]
        accuracy = (sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10))
        loss_meter.update(loss.item(), input.shape[0])
        batch_time.update((time.time() - end))
        end = time.time()
        if ((i + 1) % 10) == 0:
            logger.info(
                'Test: [{}/{}] Data {data_time.val:.3f} ({data_time.avg:.3f}) Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) Accuracy {accuracy:.4f}.'.format(
                    (iter_num * args.batch_size_val), test_num, data_time=data_time, batch_time=batch_time,
                    loss_meter=loss_meter, accuracy=accuracy))
    iou_class = (intersection_meter.sum / (union_meter.sum + 1e-10))
    accuracy_class = (intersection_meter.sum / (target_meter.sum + 1e-10))
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = (class_intersection_meter[i] / (class_union_meter[i] + 1e-10))
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = ((class_miou * 1.0) / len(class_intersection_meter))
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format((i + 1), class_iou_class[i]))
    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    return (loss_meter.avg, mIoU, mAcc, allAcc, class_miou)


if (__name__ == '__main__'):
    main()
