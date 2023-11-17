#!/usr/bin/env python
#
# File Name : cal_nss.py
#
# Description : Computes NSS metric #

# Author : Zhongxu Hu

import numpy as np
import scipy.ndimage


class NSS():
    '''
    Class for computing NSS score for saliency maps

    '''
    def __init__(self):
        
        self.width=640
        self.height=480

    def calc_score(self, gtsAnn, resAnn):
        """
        Computer NSS score.
        :param gtsAnn : ground-truth annotations
        :param resAnn : predicted saliency map
        :return score: int : NSS score
        """

        salMap = resAnn - np.mean(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.std(salMap)
        return np.mean([ salMap[y-1][x-1] for x,y in gtsAnn ])

    def compute_score(self, gt_fixs,res_salmaps):
        """
        Computes NSS score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : saliency map predictions with "image name" key and ndarray as values
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        width=res_salmaps.shape[3]
        height=res_salmaps.shape[2]
        gt_fixs=self.resize_fixations(gt_fixs,height,width)
        
        score = []
        for idx in range(len(gt_fixs)):
            fixations  = gt_fixs[idx]
            salMap = np.squeeze(res_salmaps[idx,:,:])
            #height,width = (480,640)
            #mapheight,mapwidth = np.shape(salMap)
            #salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
            score.append(self.calc_score(fixations,salMap))
        average_score = np.mean(np.array(score))
        return average_score#, np.array(score)
        
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

    def method(self):
        return "NSS"



if __name__=="__main__":
    nss = NSS()
