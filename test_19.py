#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, load_checkpoints, save_checkpoints, Discriminator_G, Discriminator_L

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images, sm_image, combine_images

from datetime import datetime
import random

'''
HPM = Human Parsing Module
GMM = GMM
TOM = Try On Module
'''
'''
def random_crop(reals, fakes, winsize):
    y, x = [random.randint(reals.size(i)//4, int(reals.size(i)*0.75)-winsize-1) for i in (2, 3)]
    return reals[:,:,y:y+winsize,x:x+winsize], fakes[:,:,y:y+winsize,x:x+winsize]
'''

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = datetime.now().strftime("%m%d_%H%M"))
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-d', '--debug', type=str, default="debug")
    parser.add_argument('-w', '--winsize', type=int, default=48)

    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument('-s', "--stage", required=True)
    parser.add_argument("--data_list", default = "train_pairs_shuffled_seed_326.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 1000)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def test_hpm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    image_seg_dir = os.path.join(save_dir, 'image-seg')
    if not os.path.exists(image_seg_dir):
        os.makedirs(image_seg_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(image_seg_dir):
        os.makedirs(image_seg_dir)
    
    for step, inputs in enumerate(test_loader.data_loader):

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()

        #generate segmentation
        segmentation = model(torch.cat([agnostic, c],1))
        save_images(segmentation, im_names, image_seg_dir)
        #generate mask
        warped_mask = segmentation
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 


def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        wm = inputs['warped_mask'].cuda()
        c = inputs['cloth'].cuda()
            
        grid, theta = model(wm, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')

        save_images(warped_cloth, c_names, warp_cloth_dir) 


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        p_tryon = model(torch.cat([agnostic, c],1))
        save_images(p_tryon, im_names, try_on_dir) 


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
    
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt, 1)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        test_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'HPM':
        model = UnetGenerator(25, 3, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        test_hpm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 3, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()

