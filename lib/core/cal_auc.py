#!/usr/bin/env python
#
# File Name : cal_auc.py
#
# Description : Computes AUC metric

# Author : zhongxu hu

import numpy as np
import scipy.ndimage


class AUC():
    '''
    Class for computing AUC score for saliency maps

    '''
    def __init__(self):
        
        self.stepSize=0.01
        self.Nrand=100000

    def calc_score(self, gtsAnn, resAnn, stepSize=.01, Nrand=100000):
        """
        Computer AUC score.
        :param gtsAnn : ground-truth annotations
        :param resAnn : predicted saliency map
        :return score: int : score
        """

        salMap = resAnn - np.min(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.max(salMap)

        S = salMap.reshape(-1)
        Sth = np.asarray([ salMap[y-1][x-1] for x,y in gtsAnn ])

        Nfixations = len(gtsAnn)
        Npixels = len(S)

        # sal map values at random locations
        randfix = S[np.random.randint(Npixels, size=Nrand)]

        allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
        allthreshes = allthreshes[::-1]
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[-1]=1.0
        fp[-1]=1.0
        tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
        fp[1:-1]=[float(np.sum(randfix >= thresh))/Nrand for thresh in allthreshes]

        auc = np.trapz(tp,fp)
        return auc

    def compute_score(self, gt_fixs,res_salmaps):
        """
        Computes AUC score for a given set of predictions and fixations
        :param gts : dict : fixation points with "image name" key and list of points as values
        :param res : dict : salmap predictions with "image name" key and ndarray as values
        :returns: average_score: float (mean NSS score computed by averaging scores for all the images)
        """
        self.width=res_salmaps.shape[3]
        self.height=res_salmaps.shape[2]
        score = []
        gt_fixs=self.resize_fixations(gt_fixs,self.height,self.width)

        for idx in range(len(gt_fixs)):
            fixations  = gt_fixs[idx]
            salMap = np.squeeze(res_salmaps[idx,:,:])
            #height,width = (480,640)
            #mapheight,mapwidth = np.shape(salMap)
            #salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
            score.append(self.calc_score(fixations,salMap))
        average_score = np.mean(np.array(score))
        return average_score#, np.array(score)

    def method(self):
        return "AUC"
        
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
    auc = AUC()
    #more tests here
