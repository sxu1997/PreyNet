# refernce from Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# author: Shuang Xu
# data: 2021-11-28
import os
import torch
from torch import nn
from apex import amp

import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from model.B_TwoStage_P66_20_SC_emb_att import SAM_ResNet
from reference.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, adjust_learning_rate_poly, adjust_lr_4
from eval.metric import *
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import sys
import random
from utils import pytorch_iou, pytorch_ssim
import math

#
#


def set_rand_seed(seed=1024):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True   # 保证每次返回得的卷积算法是确定的


def w_err_loss(pred, mask, pred_err):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    pred1 = torch.sigmoid(pred)
    err_gt = torch.abs(pred1.detach() - mask)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred_err, err_gt)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    return (wbce).mean()


# sal loss: bce+iou
def sal_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)

    return bce + iou.mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step, test_best_mae, test_best_fm, test_best_sm, test_best_epoch
    model.train()
    loss_all = 0
    loss_stage2_all = 0
    loss_stage1_0_all = 0
    loss_stage1_1_all = 0
    loss_stage1_2_all = 0
    loss_stage1_3_all = 0
    loss_err_0_all = 0
    loss_err_1_all = 0
    loss_err_2_all = 0
    loss_err_3_all = 0

    E_score0_all = 0
    E_score1_all = 0
    E_score2_all = 0
    E_score3_all = 0

    epoch_step = 0

    for i, (images, gts) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = images.cuda()
        gts = gts.cuda()

        preds, stage1_pre0, stage1_pre1, stage1_pre2, stage1_pre3, \
        err_map0, err_map1, err_map2, err_map3 = model(images)

        loss_stage2 = sal_loss(preds, gts)
        loss_stage1_0 = sal_loss(stage1_pre0, gts)
        loss_stage1_1 = sal_loss(stage1_pre1, gts)
        loss_stage1_2 = sal_loss(stage1_pre2, gts)
        loss_stage1_3 = sal_loss(stage1_pre3, gts)
        loss_errmap0 = w_err_loss(stage1_pre0, gts, err_map0)
        loss_errmap1 = w_err_loss(stage1_pre1, gts, err_map1)
        loss_errmap2 = w_err_loss(stage1_pre2, gts, err_map2)
        loss_errmap3 = w_err_loss(stage1_pre3, gts, err_map3)

        # err map使用sigmoid
        # 计算for-score\total-score\final-score
        for_score0 = (torch.sum(torch.sum(torch.sigmoid(err_map0) * gts, dim=(1, 2, 3)) / torch.sum(gts, dim=(
        1, 2, 3))) / opt.batchsize).item()
        score0 = (torch.sum(torch.sigmoid(err_map0)).float() / (opt.trainsize * opt.trainsize * opt.batchsize)).item()
        E_score0 = (for_score0 + score0) / 2.0

        for_score1 = (torch.sum(torch.sum(torch.sigmoid(err_map1) * gts, dim=(1, 2, 3)) / torch.sum(gts, dim=(
        1, 2, 3))) / opt.batchsize).item()
        score1 = (torch.sum(torch.sigmoid(err_map1)).float() / (opt.trainsize * opt.trainsize * opt.batchsize)).item()
        E_score1 = (for_score1 + score1) / 2.0

        for_score2 = (torch.sum(torch.sum(torch.sigmoid(err_map2) * gts, dim=(1, 2, 3)) / torch.sum(gts, dim=(
        1, 2, 3))) / opt.batchsize).item()
        score2 = (torch.sum(torch.sigmoid(err_map2)).float() / (opt.trainsize * opt.trainsize * opt.batchsize)).item()
        E_score2 = (for_score2 + score2) / 2.0

        for_score3 = (torch.sum(torch.sum(torch.sigmoid(err_map3) * gts, dim=(1, 2, 3)) / torch.sum(gts, dim=(
        1, 2, 3))) / opt.batchsize).item()
        score3 = (torch.sum(torch.sigmoid(err_map3)).float() / (opt.trainsize * opt.trainsize * opt.batchsize)).item()
        E_score3 = (for_score3 + score3) / 2.0

        # softmax
        scores = torch.tensor([1 - E_score0, 1 - E_score1, 1 - E_score2, 1 - E_score3])
        func = nn.Softmax()
        soft_scores = func(scores)
        R_score0, R_score1, R_score2, R_score3 = soft_scores[0], soft_scores[1], soft_scores[2], soft_scores[3]
        loss = 1 * loss_stage2 + R_score0 * loss_stage1_0 + R_score1 * loss_stage1_1 + R_score2 * loss_stage1_2 + R_score3 * loss_stage1_3 \
               + loss_errmap0 + loss_errmap1 + loss_errmap2 + loss_errmap3

        # 1:1
        # loss = 1 * loss_stage2 + loss_stage1_0 + loss_stage1_1 + loss_stage1_2 + loss_stage1_3 \
        #        + loss_errmap0 + loss_errmap1 + loss_errmap2 + loss_errmap3

        # F3Net
        # loss = 1 * loss_stage2 + 0.5 * loss_stage1_0 + 0.25 * loss_stage1_1 + 0.125 * loss_stage1_2 + 0.0625 * loss_stage1_3 \
        #        + loss_errmap0 + loss_errmap1 + loss_errmap2 + loss_errmap3

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        step += 1
        epoch_step += 1

        loss_all += loss.data

        loss_stage2_all += loss_stage2.data
        loss_stage1_0_all += loss_stage1_0.data
        loss_stage1_1_all += loss_stage1_1.data
        loss_stage1_2_all += loss_stage1_2.data
        loss_stage1_3_all += loss_stage1_3.data
        loss_err_0_all += loss_errmap0.data
        loss_err_1_all += loss_errmap1.data
        loss_err_2_all += loss_errmap2.data
        loss_err_3_all += loss_errmap3.data


        E_score0_all += E_score0
        E_score1_all += E_score1
        E_score2_all += E_score2
        E_score3_all += E_score3

        if i % 20 == 0 or i == total_step or i == 1:
            # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}'.
            #       format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
            logging.info(
                '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_all: {:.4f}, loss_stage2: {:.4f},'
                'loss_stage1_0: {:.4f}, loss_stage1_1: {:.4f}, loss_stage1_2: {:.4f},loss_stage1_3: {:.4f},'
                'E_score0: {:.4f}, E_score1: {:.4f}, E_score2: {:.4f}, E_score3: {:.4f}, loss_errmap0: {:.4f}, loss_errmap1: {:.4f}, loss_errmap2: {:.4f}, loss_errmap3: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_stage2.data,
                           loss_stage1_0.data, loss_stage1_1.data, loss_stage1_2.data, loss_stage1_3.data, E_score0,
                           E_score1, E_score2, E_score3,
                           loss_errmap0.data, loss_errmap1.data, loss_errmap2.data, loss_errmap3.data))

    loss_all /= epoch_step

    loss_stage2_all /= epoch_step
    loss_stage1_0_all /= epoch_step
    loss_stage1_1_all /= epoch_step
    loss_stage1_2_all /= epoch_step
    loss_stage1_3_all /= epoch_step
    loss_err_0_all /= epoch_step
    loss_err_1_all /= epoch_step
    loss_err_2_all /= epoch_step
    loss_err_3_all /= epoch_step

    E_score0_all /= epoch_step
    E_score1_all /= epoch_step
    E_score2_all /= epoch_step
    E_score3_all /= epoch_step

    #
    logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, Loss_stage2_AVG: {:.4f}, '
                 'Loss1_0_AVG: {:.4f}, Loss1_1_AVG: {:.4f}, Loss1_2_AVG: {:.4f}, Loss1_3_AVG: {:.4f},'
                 'E_score0_AVG: {:.4f}, E_score1_AVG: {:.4f}, E_score2_AVG: {:.4f}, E_score3_AVG: {:.4f}, Loss_err_0_AVG: {:.4f}, Loss_err_1_AVG: {:.4f}, Loss_err_2_AVG: {:.4f}, Loss_err_3_AVG: {:.4f}'.format
                 (epoch, opt.epoch, loss_all, loss_stage2_all,
                  loss_stage1_0_all, loss_stage1_1_all, loss_stage1_2_all, loss_stage1_3_all,
                  E_score0_all, E_score1_all, E_score2_all, E_score3_all, loss_err_0_all, loss_err_1_all,
                  loss_err_2_all, loss_err_3_all))

    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)


    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, _, _, _, _, _, _, _, _ = model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size

        writer.add_scalar('MAE', mae, global_step=epoch)
        return mae


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='/home/lewis/0_xs/COD_data/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/home/lewis/0_xs/COD_data/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--val_root_2', type=str,
                        default='/home/lewis/0_xs/COD_data/TestDataset/CAMO-small/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/75_01_score/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')

    cudnn.benchmark = True
    seed = 1024
    set_rand_seed(seed)

    # build the model
    model = SAM_ResNet().cuda()


    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)

    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    val_loader2 = test_dataset(image_root=opt.val_root_2 + 'Imgs/',
                               gt_root=opt.val_root_2 + 'GT/',
                               testsize=opt.trainsize, )

    total_step = len(train_loader)


    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(opt.save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))
    logging.info('Random Seed: {}'.format(seed))

    step = 0
    writer = SummaryWriter(save_path + 'summary')


    test_best_mae = 100
    test_best_epoch = 0
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    print("Start train...")
    try:
        for epoch in range(1, opt.epoch):
            # cur_lr = adjust_lr_4(optimizer, opt.lr, epoch)
            # 调整学习率poly
            cur_lr = adjust_learning_rate_poly(optimizer, epoch, opt.epoch, opt.lr, power=0.9)
            # cur_lr = adjust_lr_3(optimizer, opt.lr, epoch)
            writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
            logging.info("lr:{:.8f}".format(cur_lr))
            train(train_loader, model, optimizer, epoch, save_path, writer)
            logging.info("Start test...")
            # val(val_loader, model, epoch, save_path, writer)
            MAE_small = val(val_loader2, model, epoch, save_path, writer)
            MAE = val(val_loader, model, epoch, save_path, writer)
            logging.info(
                "CAMO-small: MAE:{:.4f}".format(MAE_small))
            logging.info(
                "CAMO: MAE:{:.4f}".format(MAE))
            if MAE < test_best_mae:
                test_best_mae = MAE
                test_best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')

            logging.info("Best CAMO: MAE:{:.4f}, correspond EPOCH:{} ".format(test_best_mae, test_best_epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt...')
        logging.info("Best CAMO: MAE:{:.4f}, correspond EPOCH:{}".format(
            test_best_mae, test_best_epoch))

# 打印学习率
# except KeyboardInterrupt:
#     print('Keyboard Interrupt: save model and exit.')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
#     print('Save checkpoints successfully!')
#     raise
