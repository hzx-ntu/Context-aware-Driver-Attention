import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import random
import math

from lib.dataset.heatmap import *

class BDDA(data.Dataset):
    def __init__(self, split = 'train', imSize=(144,256),aug=False):

        self.dataPath = './dataset/bdda/BDDA'
        self.imSize = imSize
        
        self.aug=aug
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformImg = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.transformSal = transforms.Compose([
            transforms.Resize((36,64)),
            transforms.ToTensor(),
        ])
        
        
        if split == 'train':
            self.indices=self._get_db(True)
        else:
            self.indices=self._get_db(False)
        

        print('Loaded BDDA dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def __getitem__(self, index):
        sample = self.indices[index]
        
        imPath=sample['image']
        img = self.loadImage(imPath)
        
        img_sal=Image.open(sample['sal']).convert('L')
        
        '''
        if self.aug:
           
           if random.random()>0.5:
              img=img.transpose(Image.FLIP_LEFT_RIGHT)
              img_sal=img_sal.transpose(Image.FLIP_LEFT_RIGHT)
           
           if random.random()>0.5:
              
              left=math.floor(random.random()*17)*1
              top=math.floor(random.random()*10)*1
              
              img=img.crop((left,top,left+596,top+279))
              img_sal=img_sal.crop((left,top,left+596,top+279))
        '''
        
        img=self.transformImg(img)
        img_sal=self.transformSal(img_sal)

        return img, img_sal
    
        
    def __len__(self):
        return len(self.indices)
    
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(True)
        else:
            # use bbox from detection
            gt_db = self._load_samples(False)
        return gt_db
    
    def _load_samples(self,is_train=False):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]

        if is_train:
           org_file='training'
        else:
           org_file='test'
        
        idx_list=os.listdir('%s/%s/camera_frames'%(self.dataPath,org_file))

        for v_idx in idx_list:
            
            
            file_list=[f for f in os.listdir('%s/%s/camera_frames/%s'%(self.dataPath,org_file,v_idx)) if f.endswith('.jpg')]
            
            file_list.sort()
            
            for i, img_idx in enumerate(file_list):
                   
                    if i <5:
                       continue

                    image_path=os.path.join(self.dataPath,'%s'%org_file,'camera_frames','%s'%v_idx,img_idx)
                    
                    sal_path=os.path.join(self.dataPath,'%s'%org_file,'gazemap_frames','%s'%v_idx,img_idx)
                    
                    if os.path.exists(image_path) and  os.path.exists(sal_path):
                          
                          gt_db.append({
                              'image':image_path,
                              'sal':sal_path,
                              })
                
        return gt_db
