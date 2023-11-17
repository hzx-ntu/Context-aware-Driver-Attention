# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

import torch
import torch.nn as nn
import math

class Gaussian_Prior(nn.Module):
     def __init__(self,num,constraint=True):
         super(Gaussian_Prior,self).__init__()
         
         self.num_gaussian=num
         self.weights=torch.nn.Parameter(torch.FloatTensor(torch.rand(self.num_gaussian*4)))
         #self.epsilon= torch.from_numpy(np.array([1e-10])).cuda()
         #self.pi=np.pi
         #self.epsilon=torch.FloatTensor([1e-10]).cuda()
         self.scaling=6
         self.constraint=constraint
     
     def forward(self,b,h,w):
        
        self.b=b
        self.w=w
        self.h=h

        mu_x = self.weights[:self.num_gaussian]
        mu_y = self.weights[self.num_gaussian:self.num_gaussian*2]
        sigma_x = self.weights[self.num_gaussian*2:self.num_gaussian*3]
        sigma_y = self.weights[self.num_gaussian*3:]
        
        mu_x=mu_x.view([-1,1,1])
        mu_y=mu_y.view([-1,1,1])
        sigma_x=sigma_x.view([-1,1,1])
        sigma_y=sigma_y.view([-1,1,1])


        e = self.h / self.w
        e1 = (1 - e) / 2
        e2 = e1 + e
        
        if self.constraint:
            mu_x = torch.clip(mu_x, 0.25, 0.75) #0.25, 0.75
            mu_y = torch.clip(mu_y, 0.35, 0.65) #0.35, 0.65
    
            sigma_x = torch.clip(sigma_x, 0.1, 0.9)
            sigma_y = torch.clip(sigma_y, 0.2, 0.8)
        else:
            mu_x = torch.clip(mu_x, 0.05, 0.95) #0.25, 0.75
            mu_y = torch.clip(mu_y, 0.05, 0.95) #0.35, 0.65
    
            sigma_x = torch.clip(sigma_x, 0.1, 0.9)
            sigma_y = torch.clip(sigma_y, 0.1, 0.9)

        x_t = torch.mm(torch.ones((self.h, 1)), self._linspace(0, 1.0, self.w).view(1,-1)).cuda()
        y_t = torch.mm(self._linspace(e1, e2, self.h).view(-1,1), torch.ones((1, self.w))).cuda()
        
        x_t=x_t.repeat(self.num_gaussian,1,1)
        y_t=y_t.repeat(self.num_gaussian,1,1)
        

        #eps=self.epsilon.repeat(self.num_gaussian)
        '''
        gaussian = 1 / (torch.add(2 * math.pi * sigma_x * sigma_y, self.epsilon)) * \
                   torch.exp(-((x_t - mu_x) ** 2 / (torch.add(2 * sigma_x ** 2, self.epsilon)) +
                           (y_t - mu_y) ** 2 / (torch.add(2 * sigma_y ** 2 , self.epsilon))))
        '''
        
        gaussian = 1 / (2 * math.pi * sigma_x * sigma_y) * \
                   torch.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2) +
                           (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 )))

        max_gauss = torch.max(gaussian,2,keepdim=True)[0]
        max_gauss = torch.max(max_gauss,1,keepdim=True)[0]
        gaussian = gaussian / max_gauss
        
        #gaussian*=self.scaling
        
        output = gaussian.repeat(self.b,1,1,1)

        return output
     def _linspace(self,start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        step = (stop - start) / (num - 1)
        return torch.arange(num) * step + start

class Gaussian_Prior_Conv(nn.Module):
     def __init__(self,num):
         super(Gaussian_Prior_Conv,self).__init__()
         
         self.num_gaussian=num
         #self.epsilon= torch.from_numpy(np.array([1e-10])).cuda()
         #self.pi=np.pi
         #self.epsilon=torch.FloatTensor([1e-10]).cuda()
         
     
     def forward(self,weights,b,h,w):
        
        self.b=b
        self.w=w
        self.h=h
        
        mu_x = weights[:,:self.num_gaussian]
        mu_y = weights[:,self.num_gaussian:self.num_gaussian*2]
        sigma_x = weights[:,self.num_gaussian*2:self.num_gaussian*3]
        sigma_y = weights[:,self.num_gaussian*3:]
        
        mu_x=mu_x.view([-1,self.num_gaussian,1,1])
        mu_y=mu_y.view([-1,self.num_gaussian,1,1])
        sigma_x=sigma_x.view([-1,self.num_gaussian,1,1])
        sigma_y=sigma_y.view([-1,self.num_gaussian,1,1])


        e = self.h / self.w
        e1 = (1 - e) / 2
        e2 = e1 + e
        
        
        mu_x = torch.clip(mu_x, 0.25, 0.75)
        mu_y = torch.clip(mu_y, 0.35, 0.65)
    
        sigma_x = torch.clip(sigma_x, 0.1, 0.9)
        sigma_y = torch.clip(sigma_y, 0.2, 0.8)

        x_t = torch.mm(torch.ones((self.h, 1)), self._linspace(0, 1.0, self.w).view(1,-1)).cuda()
        y_t = torch.mm(self._linspace(e1, e2, self.h).view(-1,1), torch.ones((1, self.w))).cuda()
        
        x_t=x_t.repeat(self.num_gaussian,1,1)
        y_t=y_t.repeat(self.num_gaussian,1,1)
        
        x_t=x_t.repeat(self.b,1,1,1)
        y_t=y_t.repeat(self.b,1,1,1)
        

        #eps=self.epsilon.repeat(self.num_gaussian)
        '''
        gaussian = 1 / (torch.add(2 * math.pi * sigma_x * sigma_y, self.epsilon)) * \
                   torch.exp(-((x_t - mu_x) ** 2 / (torch.add(2 * sigma_x ** 2, self.epsilon)) +
                           (y_t - mu_y) ** 2 / (torch.add(2 * sigma_y ** 2 , self.epsilon))))
        '''
        
        gaussian = 1 / (2 * math.pi * sigma_x * sigma_y) * \
                   torch.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2) +
                           (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 )))

        max_gauss = torch.max(gaussian,3,keepdim=True)[0]
        max_gauss = torch.max(max_gauss,2,keepdim=True)[0]
        gaussian = gaussian / max_gauss
        
        #output = gaussian.repeat(self.b,1,1,1)

        return gaussian
     def _linspace(self,start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        step = (stop - start) / (num - 1)
        return torch.arange(num) * step + start