import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

import lib.models as models

class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class EYESALModel(nn.Module):

    def __init__(self,cfg):
        super(EYESALModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        
        self.gaze_grid=[16,9]
        
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*12*12*64, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64+128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.gaze_grid[0]*self.gaze_grid[1]),
            )
        self.salModel=models.bdda_res_mhfnet.get_net(cfg,False)
        #Joning saliencymap
        self.sal_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # Joining everything
        self.output = nn.Sequential(
            nn.Linear(self.gaze_grid[0]*self.gaze_grid[1]+16*9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )
        
        #self.sigmoid=nn.Sigmoid()
        self.w_index = torch.range(0,self.gaze_grid[0]-1).repeat(self.gaze_grid[1],1)
        self.w_index = torch.FloatTensor(self.w_index.view(self.gaze_grid[0]*self.gaze_grid[1])).cuda()
        self.h_index = torch.range(0,self.gaze_grid[1]-1).view(self.gaze_grid[1],1).repeat(1,self.gaze_grid[0])
        self.h_index = torch.FloatTensor(self.h_index.view(self.gaze_grid[0]*self.gaze_grid[1])).cuda()
        
        self.softmax=nn.Softmax(dim=1).cuda()
        self.relu=nn.ReLU(inplace=True)

    def forward(self, faces, eyesLeft, eyesRight, faceGrids, img , onlysal=False):
        
        #Sal Net
        Sal=self.salModel(img)
        
        if onlysal:
           return Sal
        
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)
        
        xGaze=self.relu(x)
        
        xSal=Sal
        xSal=self.sal_layer(xSal)
        xSal=xSal.view(xSal.shape[0],-1)
        xFusion=self.output(torch.cat((xGaze,xSal),1))
        
        '''
        #x = self.sigmoid(x)
        x_cls=self.softmax(x)
        x_w=torch.sum(x_cls*self.w_index,1)
        x_h=torch.sum(x_cls*self.h_index,1)
        x_wh=torch.stack([x_w,x_h],1)
        #x = torch.reshape(x,(-1,8,16))
        '''
        
        return x,xFusion,Sal
