import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re

from lib.dataset.heatmap import *

MEAN_PATH = '.'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)

def collate_fn_filter_d(batch):  
    
    data=[]
    target=[]
    face=[]
    eyel=[]
    eyer=[]
    grid=[]
    gaze=[]
    gaze_cls=[]
    
    for item in batch:
        img_sal=item[4]
        if torch.max(img_sal)>0.8:
           face.append(item[0])
           eyel.append(item[1])
           eyer.append(item[2])
           data.append(item[3])
           target.append(item[4])
           grid.append(item[5])
           gaze.append(item[6])
           gaze_cls.append(item[7])
    if len(data)==0:
       return [[],[],[],[],[],[],[],[]]       
    face = torch.stack(face)
    eyel = torch.stack(eyel)
    eyer = torch.stack(eyer)
    data = torch.stack(data)
    target = torch.stack(target)
    grid = torch.stack(grid)
    gaze = torch.stack(gaze)
    gaze_cls = torch.stack(gaze_cls)
    return [face,eyel,eyer,data, target,grid,gaze,gaze_cls]


class DREYEVED(data.Dataset):
    def __init__(self, split = 'train', imSize=(224,224), gridSize=(25, 25)):

        self.dataPath = './dataset/dreyeve-d'
        self.dataPath_org = './dataset/AImageLab'
        self.imSize = imSize
        self.gridSize = gridSize
        
        self.gaze_grid=[16,9]
        
        self.trainidx=[6,7,10,11,12,26,27,35,40,47]
        self.testidx=[53,60,64,65,70,72,74]
        
        self.pf_params=np.array([1920,1080, 70, 39.5, 26, 7]) # resolution (w,h), physic size(w,h)cm, left up point to camera center(x,y)cm
        
        #self.cali=False
        self.eye_w=28
        self.mirror=True

        self.faceMean = loadMetadata('./lib/dataset/mean_face_224.mat')['image_mean']
        self.eyeLeftMean = loadMetadata('./lib/dataset/mean_left_224.mat')['image_mean']
        self.eyeRightMean = loadMetadata('./lib/dataset/mean_right_224.mat')['image_mean']
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transformImg = transforms.Compose([
            transforms.Resize((144,256)),
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
        

        print('Loaded DR(EYE)VE-D dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        sample = self.indices[index]
        
        imScene = self.loadImage(sample['image'])
        imScene = self.transformImg(imScene)
        
        
        imFace=self.loadImage(sample['face'])
        imEyeL=self.loadImage(sample['eyel'])
        imEyeR=self.loadImage(sample['eyer'])
        
        img_sal=Image.open(sample['sal']).convert('L')
        img_sal=self.transformSal(img_sal)
        
        kps=np.load(sample['kps'])
        kps = np.array(kps, np.int32)
        
        bounding_box=kps[0:4]
        
        if self.mirror:
           bounding_box[0]=1920-(bounding_box[0]+bounding_box[2])
        

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)
        
        pf_param=self.pf_params
        gaze = sample['gt']
        gaze = np.array(gaze, np.float32)
        
        #to class idx
        bin_w = np.array(range(0, 1920, int(pf_param[0]/self.gaze_grid[0])))
        bin_h = np.array(range(0, 1080, int(pf_param[1]/self.gaze_grid[1])))
        gaze_w = np.digitize(gaze[0], bin_w) - 1
        gaze_h = np.digitize(gaze[1], bin_h) - 1
        gaze_cls=np.array([gaze_h*self.gaze_grid[0]+gaze_w])
        
        
        #to centermeter
        gaze=gaze/[pf_param[0],pf_param[1]] * self.gaze_grid
        

        params=bounding_box/[76.8,43.2,76.8,43.2]
        faceGrid = self.makeGrid(params)

        # to tensor
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)
        gaze_cls = torch.LongTensor(gaze_cls)

        return imFace, imEyeL, imEyeR, imScene,img_sal, faceGrid, gaze,gaze_cls#,p_idx
    
        
    def __len__(self):
        return len(self.indices)
    
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(self.trainidx,True)
        else:
            # use bbox from detection
            gt_db = self._load_samples(self.testidx)
        return gt_db
    
    def _load_subsequences(self):
        
        f=open('./dataset/dreyeve/subsequences.txt')
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
    
    def _load_samples(self,idx_list,is_train=False):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]
        pf_param=self.pf_params
        
        unatten_list=self._load_subsequences()

        for p_id,idx in enumerate(idx_list):
                
            lines=[]
            with open('%s/DREYEVE_DATA_PART/%02d/summary.txt'%(self.dataPath,idx)) as f:
                 lines = f.readlines()
            
            file_list=unatten_list[idx-1]
            
            for i,line in enumerate(lines):
               
                if i <10 or i > len(lines)-10:
                   continue
                   
                lbls=line.split(' ')
                if len(lbls)<3:
                   continue
                
                frame_no=int(float(lbls[0]))
                
                if frame_no in file_list:
                   continue

                image_path=os.path.join(self.dataPath,'DREYEVE_DATA_PART','%02d'%idx,'frame','frame%05d.jpg'%(frame_no))
                face_path=os.path.join(self.dataPath,'DREYEVE_FACE_DETECT','%02d_crop'%idx,'frame%05d_face.jpg'%(frame_no))
                eyel_path=os.path.join(self.dataPath,'DREYEVE_FACE_DETECT','%02d_crop'%idx,'frame%05d_eyel.jpg'%(frame_no))
                eyer_path=os.path.join(self.dataPath,'DREYEVE_FACE_DETECT','%02d_crop'%idx,'frame%05d_eyer.jpg'%(frame_no))
                kps_path=os.path.join(self.dataPath,'DREYEVE_FACE_DETECT','%02d'%idx,'frame%05d.jpg.npy'%(frame_no))
                sal_path=os.path.join(self.dataPath_org,'%02d'%idx,'gt_org','frame%d.jpg'%(frame_no))
                
                
                if os.path.exists(image_path)and os.path.exists(face_path) and \
                   os.path.exists(kps_path) and os.path.exists(sal_path):
                      
                      gt_db.append({
                          'image':image_path,
                          'face':face_path,
                          'eyel':eyel_path,
                          'eyer':eyer_path,
                          'kps':kps_path,
                          'gt':[int(float(lbls[1])),int(float(lbls[2]))],
                          #'param': pf_param,
                          'sal':sal_path,
                          'p_idx': idx,
                          })
                
        return gt_db
