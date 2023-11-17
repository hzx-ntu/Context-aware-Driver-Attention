#!/usr/bin/env python
#
# File Name : cal_sauc.py
#
# Description : Computes shuffled AUC metric #

# Author : Zhongxu Hu

import numpy as np
import scipy.ndimage
class SAUC():
    '''
    Class for computing SAUC score for saliency maps
    '''
    def __init__(self):
        self.stepSize=0.01
        
        self.width=640
        self.height=480


    def calc_score(self, gtsAnn, resAnn, shufMap, stepSize=.01):
        """
        Computer SAUC score. A simple implementation
        :param gtsAnn : list of fixation annotataions
        :param resAnn : list only contains one element: the result annotation - predicted saliency map
        :return score: int : score
        """

        salMap = resAnn - np.min(resAnn)
        if np.max(salMap) > 0:
            salMap = salMap / np.max(salMap)
        Sth = np.asarray([ salMap[y-1][x-1] for x,y in gtsAnn])
        Nfixations = len(gtsAnn)

        others = np.copy(shufMap)
        for x,y in gtsAnn:
            others[y-1][x-1] = 0

        ind = np.nonzero(others) # find fixation locations on other images
        nFix = shufMap[ind]
        randfix = salMap[ind]
        Nothers = sum(nFix)

        allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
        allthreshes = allthreshes[::-1]
        tp = np.zeros(len(allthreshes)+2)
        fp = np.zeros(len(allthreshes)+2)
        tp[-1]=1.0
        fp[-1]=1.0
        tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
        fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

        auc = np.trapz(tp,fp)
        return auc
    

    def compute_score(self,gt_fixs,res_salmaps, shufMap=np.zeros((48,64))):
        """
        Computes SAUC score for a given set of predictions and fixations
        :param gtsAnn : ground-truth annotations
        :param resAnn : predicted saliency map
        :returns: average_score: float (mean sAUC score computed by averaging scores for all the images)
        """
        
        self.width=res_salmaps.shape[3]
        self.height=res_salmaps.shape[2]
        score = []
        gt_fixs=self.resize_fixations(gt_fixs,self.height,self.width)
        shufMap=np.zeros((self.height,self.width))
        # we assume all image sizes are 640x480
        for gt_fix in gt_fixs:
            shufMap += self.buildFixMap(gt_fix, False)

        score = []
        for idx in range(len(gt_fixs)):
            fixations  = gt_fixs[idx]
            salMap = np.squeeze(res_salmaps[idx,:,:,:])
            #height,width = (480,640)
            #mapheight,mapwidth = np.shape(salMap)
            #salMap = scipy.ndimage.zoom(salMap, (float(height)/mapheight, float(width)/mapwidth), order=3)
            score.append(self.calc_score(fixations,salMap,shufMap))
        average_score = np.mean(np.array(score))
        return average_score#, np.array(score)

    def method(self):
        return "SAUC"
    
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
    
    def buildFixMap(self,fixs,blur=False,sigma=19):
        """
        TODO: Build Saliency Map based on fixation annotations
        refer to format spec to see the format of fixations
        """
        if len(fixs) == 0:
            return 0

        #create saliency map
        sal_map = np.zeros((self.height,self.width))

        for x,y in fixs:
            sal_map[y-1][x-1] = 1
        if blur:
            sal_map = ndimage.filters.gaussian_filter(sal_map, sigma)
            sal_map -= np.min(sal_map)
            sal_map /= np.max(sal_map)
        return sal_map



if __name__=="__main__":
    sauc = SAUC()
    #more tests here
