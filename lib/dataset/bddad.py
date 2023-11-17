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


Train_Name_List = ['Yanxin','Siyi','LYW','HZX','DWY','lhd','Yiran','CH','GZH','DZM','Haohan']


TrainNo = [[2, 3, 4, 5, 10, 11, 12, 13, 15, 17, 20, 22, 23, 27, 28, 29, 30, 31, 32, 34, 35, 38, 40, 44, 45, 47, 51, 52, 56, 141, 145, 146, 147, 148],
[59, 60, 61, 111, 113, 114, 117, 118, 120, 122, 124, 125, 126, 127, 129, 131, 134, 136, 137, 139],
[62, 63, 64, 65, 69, 71, 72, 73, 74, 76, 78, 79, 90, 94, 102, 105, 106, 107, 109],
[151, 152, 153, 155, 156, 159, 161, 162, 163, 164, 165, 166, 168, 169, 173, 174, 175, 176, 177, 178],
[181, 183, 184, 188, 190, 194, 195, 196, 198, 200, 201, 202, 206, 215, 217, 218, 221, 222, 223, 226],
[227, 228, 229, 230, 231, 234, 235, 237, 239, 243, 244, 245, 246, 247, 248],
[250, 251, 252, 253, 255, 256, 259, 262, 263, 264, 266, 268, 269, 271, 272, 273, 277, 278, 279, 280],
[280, 283, 285, 286, 287, 288, 290, 291, 293, 296, 300, 301, 303, 304, 305, 306, 309, 314, 315, 316],
[321, 322, 323, 325, 329, 331, 337, 339, 340, 345, 346, 347, 348, 349, 350, 352, 353, 354, 356, 357],
[358, 359, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 373, 374, 375],
[376, 378, 379, 380, 383, 386, 387, 388, 389, 392, 394, 395, 403, 405, 406, 408, 410, 411, 413, 415]]


Test_Name_List = ['Yuxuan','Jim','Cdd','Sjj']

TestNo = [[19, 25, 33, 37, 42, 53, 57, 58, 85, 88, 89, 93, 100],
[108, 112, 115, 116, 128, 132, 135, 138, 171, 191, 199, 205, 212],
[233, 238, 249, 258, 313, 335, 338, 343, 372],
[282, 297, 298, 302, 377, 384, 391, 398 ,400, 409, 424, 442]]

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


class BDDAD(data.Dataset):
    def __init__(self, split = 'train', imSize=(224,224), gridSize=(25, 25)):

        self.dataPath = './dataset/bdda-d'
        self.dataPath_org = './dataset/bdda/BDDA'
        self.imSize = imSize
        self.gridSize = gridSize
        
        self.gaze_grid=[16,9]
        
        self.trainidx=TrainNo
        self.trainname=Train_Name_List
        self.testidx=TestNo
        self.testname=Test_Name_List
        
        self.pf_params=np.array([1024,576, 70, 39.5, 26, 7]) # resolution (w,h), physic size(w,h)cm, left up point to camera center(x,y)cm
        
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
        

        print('Loaded BDDA-D dataset split "%s" with %d records...' % (split, len(self.indices)))

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
        
        p_idx= sample['p_idx']
        
        #imFacePath=sample['face']
        #imFaceALL = self.loadImage(imFacePath)
        
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
        bin_w = np.array(range(0, 1024, int(self.pf_params[0]/self.gaze_grid[0])))
        bin_h = np.array(range(0, 576, int(self.pf_params[1]/self.gaze_grid[1])))
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
        p_idx=torch.LongTensor(np.array([p_idx]))

        return imFace, imEyeL, imEyeR, imScene,img_sal, faceGrid, gaze,gaze_cls
    
        
    def __len__(self):
        return len(self.indices)
    
    def _get_db(self,is_train):
        if is_train:
            # use ground truth bbox
            gt_db = self._load_samples(self.trainidx,self.trainname,True)
        else:
            # use bbox from detection
            gt_db = self._load_samples(self.testidx,self.testname)
        return gt_db
    
    def _load_samples(self,idx_list,name_list,is_train=False):
        # get images and groundtruths(position of center of head, pitch,yaw,roll,rotation matrix)
        gt_db=[]
        pf_param=self.pf_params
        
        if is_train:
           org_file='training'
        else:
           org_file='test'

        for p_id,person in enumerate(idx_list):
            
            for idx in person:
                
                lines=[]
                with open('%s/%05d/summary.txt'%(self.dataPath,idx)) as f:
                     lines = f.readlines()
                
                for i,line in enumerate(lines):
                   
                    if i <10 or i > len(lines)-10:
                       continue
                       
                    lbls=line.split(' ')
                    if len(lbls)<3:
                       continue
                    frame_no=int(lbls[0])

                    image_path=os.path.join(self.dataPath,'%05d'%idx,'frame','frame%05d.jpg'%(frame_no))
                    face_path=os.path.join(self.dataPath,'%05d'%idx,'face_crop','frame%05d_face.jpg'%(frame_no))
                    eyel_path=os.path.join(self.dataPath,'%05d'%idx,'face_crop','frame%05d_eyel.jpg'%(frame_no))
                    eyer_path=os.path.join(self.dataPath,'%05d'%idx,'face_crop','frame%05d_eyer.jpg'%(frame_no))
                    kps_path=os.path.join(self.dataPath,'%05d'%idx,'face','frame%05d.jpg.npy'%(frame_no))
                    sal_path=os.path.join(self.dataPath_org,org_file,'gazemap_frames','%05d'%idx,'frame%05d.jpg'%(frame_no))
                    
                    if os.path.exists(image_path)and os.path.exists(face_path) and \
                       os.path.exists(kps_path) and os.path.exists(sal_path):
                          
                          gt_db.append({
                              'image':image_path,
                              'face':face_path,
                              'eyel':eyel_path,
                              'eyer':eyer_path,
                              'kps':kps_path,
                              'gt':[int(lbls[1]),int(lbls[2])],
                              #'param': pf_param,
                              'sal':sal_path,
                              'p_idx': idx,
                              })
                
        return gt_db
