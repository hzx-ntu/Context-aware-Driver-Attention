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

def collate_fn_filter(batch):  
    data=[]
    target=[]
    mean=[]
    
    for item in batch:
        img_sal=item[1]
        if torch.max(img_sal)>0.8:
           data.append(item[0])
           target.append(item[1])
           mean.append(item[2])
    if len(data)==0:
       return [[],[],[]]       
    data = torch.stack(data)
    target = torch.stack(target)
    mean = torch.stack(mean)
    return [data, target,mean]

class DREYEVE(data.Dataset):
    def __init__(self, split = 'train', imSize=(144,256),aug=False):

        self.dataPath = './dataset/dreyeve'
        self.dataPath_org = './dataset/AImageLab'
        self.imSize = imSize
        
        self.aug=aug
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformImg = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.transformSal = transforms.Compose([
            transforms.Resize((144,256)),
            transforms.ToTensor(),
        ])
        
        
        if split == 'train':
            self.indices=self._get_db(True)
        else:
            self.indices=self._get_db(False)
        

        print('Loaded DR(EYE)VE dataset split "%s" with %d records...' % (split, len(self.indices)))
    

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im
    
    def updateindex(self, index):
        new_index=index+1
        
        if new_index>=len(self.indices):
           new_index=index-1
        
        return new_index


    def __getitem__(self, index):
    
        sample = self.indices[index]
        #imScene = self.loadImage(sample['image'])
        
        img_sal=Image.open(sample['sal']).convert('L')
        mean_gt=Image.open(sample['mean_gt']).convert('L')
        
        imPath=sample['image']
        img = self.loadImage(imPath)
        
        
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
        mean_gt=self.transformSal(mean_gt)

        return img, img_sal,mean_gt
    
        
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
    
    def _load_subsequences(self):
        
        f=open('%s/subsequences.txt'%(self.dataPath))
        lines=f.readlines()
        
        unatten_list_all=[]
        unatten_idx=[]
        start_idx=1
        for line in lines:

            nums=line.split('\t')
            if not int(nums[0])==start_idx:
            	unatten_list_all.append(unatten_idx)
            	unatten_idx=[]
            	start_idx=int(nums[0])
        

            unatten_idx+=list(range(int(nums[1])-1,int(nums[2])))
        
        unatten_list_all.append(unatten_idx)

        return unatten_list_all
            
            
    
    def _load_samples(self,is_train=False):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]
        
        unatten_list=self._load_subsequences()

        if is_train:
           idx_list=range(1,38)
        else:
           idx_list=range(38,75)
        
        for v_idx in idx_list:
            
            file_list=unatten_list[v_idx-1]
                
            for img_idx in range(7500):
                    
                if img_idx not in file_list:

                    image_path=os.path.join(self.dataPath,'%02d'%v_idx,'frames','frame%05d.jpg'%img_idx)
                    
                    sal_path=os.path.join(self.dataPath_org,'%02d'%v_idx,'gt_org','frame%d.jpg'%img_idx)
                    
                    meangt_path=os.path.join(self.dataPath,'mean_gt','mean_gt_%02d.png'%v_idx)
                    
                    if os.path.exists(image_path) and  os.path.exists(sal_path):
                          
                          gt_db.append({
                              'image':image_path,
                              'sal':sal_path,
                              'mean_gt':meangt_path,
                              })
                
        return gt_db
