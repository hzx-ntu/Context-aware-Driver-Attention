# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhongxu Hu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

#from lib.core.evaluate import accuracy
from lib.core.cal_sauc import SAUC
from lib.core.cal_auc import AUC
from lib.core.cal_cc import CC
from lib.core.cal_nss import NSS
from saliency_metrics import *
import scipy.io

import torch.nn as nn

import cv2
from skimage import filters


logger = logging.getLogger(__name__)


def train_sal(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_img, target_map) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        

        input_img=input_img.cuda()
        target_map = target_map.cuda()#.float()
        
        input_img = torch.autograd.Variable(input_img, requires_grad = False)
        target_map = torch.autograd.Variable(target_map, requires_grad = False)
        
        # compute output
        input_face=torch.zeros([input_img.shape[0],3,224,224]).cuda()
        input_eyel=torch.zeros([input_img.shape[0],3,224,224]).cuda()
        input_eyer=torch.zeros([input_img.shape[0],3,224,224]).cuda()
        input_grid=torch.zeros([input_img.shape[0],25,25]).cuda()
        
        outputs = model(input_face,input_eyel,input_eyer,input_grid,input_img,True)
        
        loss = criterion(outputs, target_map)


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input_img.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input_img.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

def validate_sal(config, val_loader,  model, criterion, output_dir,
             tb_log_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    #auc=AverageMeter()
    cc=AverageMeter()
    #nss=AverageMeter()
    kl=AverageMeter()
    
    #cal_auc=AUC()
    #cal_cc=CC()
    #cal_nss=NSS()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_img, target_map) in enumerate(val_loader):
            
            input_img=input_img.cuda()
            target_map = target_map.cuda()
            input_img = torch.autograd.Variable(input_img, requires_grad = False)
            target_map = torch.autograd.Variable(target_map, requires_grad = False)
            
            # compute output
            
            input_face=torch.zeros([input_img.shape[0],3,224,224]).cuda()
            input_eyel=torch.zeros([input_img.shape[0],3,224,224]).cuda()
            input_eyer=torch.zeros([input_img.shape[0],3,224,224]).cuda()
            input_grid=torch.zeros([input_img.shape[0],25,25]).cuda()
            
            outputs = model(input_face,input_eyel,input_eyer,input_grid,input_img,True)
            
            loss = criterion(outputs, target_map)
            

            num_images = input_img.size(0)
            losses.update(loss.item(), num_images)
            
            for idx in range(outputs.shape[0]):
                
                #auc_avg=cal_auc(np.squeeze(outputs[idx,:,:,:]),np.squeeze(target_map[idx,:,:,:]))
                cc_avg=cal_cc_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]))
                #nss_avg=cal_nss(np.squeeze(outputs[idx,:,:,:]),np.squeeze(target_map[idx,:,:,:]))
                kl_avg=cal_kldiv_torch(torch.squeeze(outputs[idx,:,:,:]),torch.squeeze(target_map[idx,:,:,:]))
                
                cnt=1
                #auc.update(auc_avg, cnt)
                cc.update(cc_avg, cnt)
                #nss.update(nss_avg, cnt)
                kl.update(kl_avg,cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'KLDiv {kl.val:.4f} ({kl.avg:.4f})\t' \
                      'CC {cc.val:.3f} ({cc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          kl=kl, cc=cc)
                logger.info(msg)
                
    perf_indicator=cc.avg
    
    return perf_indicator

def train_saleye(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (imFace, imEyeL, imEyeR, imScene,img_sal, faceGrid, gaze, gaze_cls) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if imFace==[]:
           continue

        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()#.float()
        imEyeR = imEyeR.cuda()#.float()
        imScene = imScene.cuda()
        img_sal=img_sal.cuda()
        faceGrid=faceGrid.cuda()
        gaze=gaze.cuda()
        gaze_cls=gaze_cls.cuda()
        
        output_gaze_cls,output_gaze,output_sal = model(imFace,imEyeL,imEyeR,faceGrid,imScene,False)
        
        
        loss = criterion(output_gaze,gaze,output_gaze_cls,gaze_cls[:,0],output_sal, img_sal)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), imFace.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=imFace.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)


def validate_saleye(config, val_loader, model, criterion, output_dir,tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    
    softmax=nn.Softmax(dim=1).cuda()
    
    error_all=[]
    
    output_all=torch.zeros([1,2]).cuda()
    gaze_all=torch.zeros([1,2]).cuda()
    
    gaze_grid=[16,9]
    #pf_params=np.array([1024,576, 70, 39.5, 26, 7])
    pf_params=np.array([1920,1080, 70, 39.5, 26, 7])
    
    w_index = torch.range(0,gaze_grid[0]-1).repeat(gaze_grid[1],1)
    w_index = torch.FloatTensor(w_index.view(gaze_grid[0]*gaze_grid[1])).cuda()
    h_index = torch.range(0,gaze_grid[1]-1).view(gaze_grid[1],1).repeat(1,gaze_grid[0])
    h_index = torch.FloatTensor(h_index.view(gaze_grid[0]*gaze_grid[1])).cuda()
    
    oIndex = 0
    for i, (imFace, imEyeL, imEyeR, imScene,img_sal, faceGrid, gaze, gaze_cls) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if imFace==[]:
           continue
        imFace = imFace.cuda()
        imEyeL = imEyeL.cuda()
        imEyeR = imEyeR.cuda()
        imScene=imScene.cuda()
        faceGrid = faceGrid.cuda()
        gaze = gaze.cuda()
        gaze_cls=gaze_cls.cuda()
        img_sal=img_sal.cuda()
        
        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        imScene = torch.autograd.Variable(imScene, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)
        img_sal = torch.autograd.Variable(img_sal, requires_grad = False)

        # compute output
        with torch.no_grad():
            output_gaze_cls,output_gaze,output_sal = model(imFace,imEyeL,imEyeR,faceGrid,imScene,False)
        
        loss = criterion(output_gaze,gaze,output_gaze_cls,gaze_cls[:,0],output_sal, img_sal)
        
        output_all=torch.cat([output_all,output_gaze],dim=0)
        gaze_all=torch.cat([gaze_all,gaze],dim=0)
        
        #output transfer
        output_w = (output_gaze[:,0])/gaze_grid[0] * pf_params[2] - pf_params[4]
        output_h = (output_gaze[:,1])/gaze_grid[1] * pf_params[3] + pf_params[5]
        output_wh=torch.stack([output_w,output_h],dim=1)
        
        gaze_w = (gaze[:,0])/gaze_grid[0] * pf_params[2] - pf_params[4]
        gaze_h = (gaze[:,1])/gaze_grid[1] * pf_params[3] + pf_params[5]
        gaze_wh=torch.stack([gaze_w,gaze_h],dim=1)
        
        #output_wh=(output_wh+pred_wh)/2
        
        lossLin = output_wh - gaze_wh
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))
        
        error_all.append(lossLin.item())

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
        
        #if lossLin.item()>8:
        #   print(p_idx)
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i%50==0:
           print('Test :[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                       loss=losses,lossLin=lossesLin))
                       
    return lossesLin.avg


def normalize(x):
    x -= x.min()
    x /= x.max()

def post_process(pred):
    pred = filters.gaussian(pred, 5)
    normalize(pred)
    pred = (pred * 255).astype(np.uint8)
    return pred

def loadMat(matPath):
    raw_fixs=scipy.io.loadmat(matPath)
    raw_gaze=raw_fixs['gaze']
    fixs=[]
    for i in range(raw_gaze.shape[0]):
        sub_fix=raw_gaze[i]['fixations'][0]
        #if (sub_fix.shape[0])==0:
        fixs.append(sub_fix.tolist())
    
    fixs = [fix for subj in fixs for fix in subj]
    
    return fixs

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
        self.avg = self.sum / self.count if self.count != 0 else 0
