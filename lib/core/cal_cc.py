#!/usr/bin/env python
#
# File Name : cal_cc.py
#
# Description : Computes CC metric #

# Author : Zhongxu Hu

import numpy as np
import scipy.ndimage
import math


class CC():
    '''
    Class for computing CC score for saliency maps
    '''
    def __init__(self):
        
        self.width=640
        self.height=480

    def calc_score(self, gtsAnn, resAnn):
        """
        Computer CC score. A simple implementation
        :param gtsAnn : ground-truth fixation map
        :param resAnn : predicted saliency map
        :return score: int : score
        """
        
        fixationMap = gtsAnn - np.mean(gtsAnn)
        if np.max(fixationMap) > 0:
            fixationMap = fixationMap / np.std(fixationMap)
        salMap = resAnn - np.mean(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.std(salMap)
        
        cc=np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]
        #cc=(salMap*fixationMap).sum() / math.sqrt((salMap*salMap).sum() * (fixationMap*fixationMap).sum())

        return cc

    def compute_score(self, gt_sals,res_salmaps):
        """
        Computes CC score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values
        :returns: average_score: float (mean CC score computed by averaging scores for all the images)
        """
        #width=res_salmaps.shape[3]
        #height=res_salmaps.shape[2]
        
        score = []
        #gt_fixs=self.resize_fixations(gt_fixs,height,width)
        
        for idx in range(res_salmaps.shape[0]):
            #fixations  = gt_fixs[idx]
            #fixationmap = self.buildFixMap(fixations,width,height)
            salMap = np.squeeze(res_salmaps[idx,:,:])
            gt_sal = np.squeeze(gt_sals[idx,:,:])
            
            score.append(self.calc_score(gt_sal,salMap))
        average_score = np.mean(np.array(score))
        return average_score#, np.array(score)

    def method(self):
        return "CC"
    
    def buildFixMap(self,fixs,width,height,blur=False,sigma=19):
        """
        TODO: Build Saliency Map based on fixation annotations
        refer to format spec to see the format of fixations
        """
        if len(fixs) == 0:
            return 0

        #create saliency map
        sal_map = np.zeros((height,width))

        for x,y in fixs:
            sal_map[y-1][x-1] = 1
        if blur:
            sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
            sal_map -= np.min(sal_map)
            sal_map /= np.max(sal_map)
        return sal_map
    def resize_fixations(self,fixations,rows,cols):
       
        factor_scale_r = rows / 480
        factor_scale_c = cols / 640
        
        fixations_resize=[]
        for fixation in fixations:
            fixation_resize=[]
            for fix in fixation:
                c = int(np.round(fix[0]*factor_scale_c))
                r = int(np.round(fix[1]*factor_scale_r))
                
                if r == rows:
                   r -= 1
                if c == cols:
                   c -= 1
                fixation_resize.append([c,r])
            fixations_resize.append(fixation_resize)
        
        return fixations_resize



if __name__=="__main__":
    cc = CC()
    #more tests here
