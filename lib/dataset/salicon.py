import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import struct
import math
#from utils.heatmap import gen_heatmap
import random
from ..utils.salicon_utils import padding_fixation
import scipy
import cv2


class SALICON(data.Dataset):
    def __init__(self, split = 'train', imSize=(192,256),aug=False):

        self.imSize = imSize
        self.wlp_root='./dataset/salicon'
        self.cat_root='./dataset/cat2000/trainSet'

        print('Loading SALICON dataset...')

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformImg = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        
        self.transformMap = transforms.Compose([
            transforms.Resize((int(self.imSize[0]/1),int(self.imSize[1]/1))), #resize(h,w)
            transforms.ToTensor(),
        ])
        
        
        self.transformFixMap = transforms.Compose([
            #transforms.Resize((int(self.imSize[0]/1),int(self.imSize[1]/1))),
            transforms.ToTensor(),
        ])
        
        self.split=split
        
        self.indices=self._get_db(split)
        
        self.aug=aug
            
        print('Loaded SALICON dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path, mode='RGB'):
        try:
            im = Image.open(path).convert(mode)
        except OSError:
            raise RuntimeError('Could not read image: ' + path)

        return im
        

    def __getitem__(self, index):
        sample = self.indices[index]

        imgPath=sample['img']
        raw_img = self.loadImage(imgPath,'RGB')
        
        mapPath=sample['map']
        raw_map = self.loadImage(mapPath,'L')
        
        
        fixPath=sample['fix']
        raw_gaze = scipy.io.loadmat(fixPath)["gaze"]
        
        fixations=[]
        for i in range(raw_gaze.shape[0]):
            sub_fix=raw_gaze[i]['fixations'][0]
            fixations.append(sub_fix.tolist())
        fixations = [fix for subj in fixations for fix in subj]
        
        fix_map = self.buildFixMap(fixations,False)
        #fix_map = self.resize_fixation(fix_map,self.imSize[0]/1,self.imSize[1]/1)
        #fix_map = Image.fromarray(np.uint8(fix_map*255))
        
        if self.aug:
           if random.random() > 0.5:
                raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                raw_map = raw_map.transpose(Image.FLIP_LEFT_RIGHT)
                #fix_map = fix_map.transpose(Image.FLIP_LEFT_RIGHT)
                fix_map = np.flip(fix_map,1)
           
           rand=random.random()
           if rand > 0.8:
              left=math.floor(random.random()*5)*10
              top=math.floor(random.random()*4)*10
              
              scale_r=480/self.imSize[0]
              scale_c=640/self.imSize[1]
              
              raw_img=raw_img.crop((left,top,left+600,top+450))
              raw_map=raw_map.crop((left,top,left+600,top+450))
              #fix_map=fix_map.crop((left,top,left+600,top+450))
              fix_map = fix_map[top:top+450,left:left+600]
           
           elif rand>0.5:
              
              left=math.floor(random.random()*5)*5
              top=math.floor(random.random()*4)*5
              
              scale_r=480/self.imSize[0]
              scale_c=640/self.imSize[1]
              
              raw_img=raw_img.crop((left,top,left+620,top+465))
              raw_map=raw_map.crop((left,top,left+620,top+465))
              #fix_map=fix_map.crop((left,top,left+600,top+450))
              fix_map = fix_map[top:top+465,left:left+620]
           
        
        fix_map = self.resize_fixation(fix_map,self.imSize[0]/1,self.imSize[1]/1)
        
        
        img=self.transformImg(raw_img)
        gt_map=self.transformMap(raw_map)
        #gt_map_2=self.transformMap_2(raw_map)
        #gt_fixmap=self.transformFixMap(fix_map)
        gt_fixmap=torch.BoolTensor(fix_map)
        #gt_fixmap_2=self.transformFixMap(fix_map_2)
        #gt_fixs=torch.LongTensor(fixations)
        imgID=torch.IntTensor([sample['imgID']])

        return img,gt_map,gt_fixmap,imgID#,gt_map_2#,gt_fixmap_2
    
    def __len__(self):
        return len(self.indices)
        
    def _get_db(self,split):
        
        gt_db = self._load_samples(split,self.wlp_root)
        
        return gt_db
        
    def _load_samples(self,idx_list,root_path):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]
        
        imgs_path=os.path.join(root_path,'images',idx_list)
        maps_path=os.path.join(root_path,'maps',idx_list)
        fixs_path=os.path.join(root_path,'fixs',idx_list)
        
        images = [imgs_path +'/' + f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_path +'/' + f for f in os.listdir(maps_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_path +'/' + f for f in os.listdir(fixs_path) if f.endswith('.mat')]
        
        
        images.sort()
        maps.sort()
        fixs.sort()
        

        for idx in range(len(images)):
                imgID=images[idx].split('/')[-1]
                imgID=imgID.split('.')[0]
                imgID=imgID.split('_')[2]
                gt_db.append({
                    'imgID': int(imgID),
                    'img':images[idx],
                    'map':maps[idx],
                    'fix':fixs[idx],
                    })
            
                
        return gt_db
    
    def resize_fixation(self,fix_map, rows=480, cols=640):
    
        out = np.zeros((int(rows), int(cols)))
        factor_scale_r = rows / fix_map.shape[0]
        factor_scale_c = cols / fix_map.shape[1]
    
        coords = np.argwhere(fix_map)
        for coord in coords:
            r = int(np.round(coord[0]*factor_scale_r))
            c = int(np.round(coord[1]*factor_scale_c))
            if r == rows:
                r -= 1
            if c == cols:
                c -= 1
            out[r, c] = 1
    
        return out
    
    
    def buildFixMap(self,fixs,blur=False,sigma=19):
        """
        TODO: Build Saliency Map based on fixation annotations
        refer to format spec to see the format of fixations
        """
        if len(fixs) == 0:
            return 0

        #create saliency map
        sal_map = np.zeros((480,640))

        for x,y in fixs:
            sal_map[y-1][x-1] = 1
        if blur:
            sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
            sal_map -= np.min(sal_map)
            sal_map /= np.max(sal_map)
        return sal_map
