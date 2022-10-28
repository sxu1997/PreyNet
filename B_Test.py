import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
from scipy import misc
# import cv2
from model.B_Supp_05_TwoStage_P66_20_SC_emb_att import SAM_ResNet
# from model.PFNet import PFNet
from reference.data_val import test_dataset
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/78_05_supp/Net_epoch_79.pth')
opt = parser.parse_args()

for _data_name in ['COD10K']:
    # 'CHAMELEON', 'CAMO', 'COD10K', 'NC4K'
    data_path = '/home/lewis/0_xs/COD_data/TestDataset/{}/'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = SAM_ResNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res2, _, _, _, _, _, _, _, _ = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        imageio.imwrite(save_path+name, res)
