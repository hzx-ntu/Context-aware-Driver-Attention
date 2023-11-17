
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class SalLoss(nn.Module):
    def __init__(self):
        super(SalLoss, self).__init__()
        self.eps = 1e-6

    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).sum()
        return loss 

    def CC_loss(self, input, target):
        input = (input - input.mean()) / input.std()  
        target = (target - target.mean()) / target.std()
        loss = (input * target).sum() / (torch.sqrt((input*input).sum() * (target * target).sum()))
        loss = 1 - loss
        return loss

    def NSS_loss(self, input, target):
        ref = (target - target.mean()) / target.std()
        input = (input - input.mean()) / input.std()
        loss = (ref*target - input*target).sum() / target.sum()
        return loss 

    def forward(self, input, smap, fix):
        kl = 0
        cc = 0
        nss = 0
        for p, f, s in zip(input, fix, smap):
            kl += 1.0*self.KL_loss(p, s)
            cc += 0.5*self.CC_loss(p, s)
            nss += 0.2*self.NSS_loss(p, f)
        return (kl + cc + nss) / input.size(0)

class JOINTLoss(nn.Module):
    """docstring for PYRMSELoss"""
    def __init__(self):
        super(JOINTLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_m = nn.CrossEntropyLoss()
        #self.criterion_m_mp = nn.CrossEntropyLoss(weight=self.weights_mp)
        #self.criterion_kl=nn.KLDivLoss(size_average=True, reduce=True)
        
    def forward(self,output,target,output_m,target_m):
        loss=0
        
        loss += 0.5*self.criterion(output,target)
        
        loss += self.criterion_m(output_m, target_m)
        
        return loss

class JOINTSALLoss(nn.Module):
    """docstring for PYRMSELoss"""
    def __init__(self):
        super(JOINTSALLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_m = nn.CrossEntropyLoss()
        self.kl_div=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        
    def forward(self,output,target,output_m,target_m,output_sal,target_sal):
        loss=0
        
        loss += 0.5*self.criterion(output,target)
        loss += self.criterion_m(output_m, target_m)
        
        loss += 1.0 * self.kl_div(output_sal,target_sal)
        loss += 0.2 * self.cc(output_sal,target_sal)
        
        return loss


class SalMSELoss(nn.Module):
    def __init__(self):
        super(SalMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        #self.kl_div=KL_divergence()
        self.kl_div=nn.KLDivLoss()
        self.cc=MyCorrCoef()
        self.nss=NSS()
        #self.bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]))
    
    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).mean()
        return loss 

    def forward(self, output, target_map,target_fix):
        
        loss=0
        #target_map=torch.unsqueeze(target_map,dim=1)
        loss += 5.0 * self.kl_div(output,target_map)
        #loss += 5.0 * self.criterion(output, target_map)
        #loss += 1.0 * self.bce(output[:,2,:,:].squeeze(),target_fix.squeeze())
        loss += 1.0 * self.cc(output,target_map)
        loss += 1.0 * self.nss(output,target_fix)
        

        return loss

class SalKLCCLoss(nn.Module):
    def __init__(self):
        super(SalKLCCLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        #self.kl_div=KL_divergence()
        self.kl_div=nn.KLDivLoss()
        self.cc=MyCorrCoef()
    
    def KL_loss(self, input, target):
        input = input / input.sum() 
        target = target / target.sum()
        loss = (target * torch.log(target/(input+self.eps) + self.eps)).mean()
        return loss 

    def forward(self, output, target_map):
        
        loss=0
        #target_map=torch.unsqueeze(target_map,dim=1)
        loss += 10.0 * self.kl_div(output,target_map)
        loss += 2.0 * self.cc(output,target_map)
        

        return loss



# KL-Divergence Loss
class KL_divergence(nn.Module):
    def __init__(self):
       super(KL_divergence,self).__init__()
       self.eps=1e-10
       
    def forward(self,y_pred_org,y_true_org):
        y_pred=torch.squeeze(y_pred_org)
        y_true=torch.squeeze(y_true_org)
        max_y_pred=torch.max(y_pred,2,keepdim=True)[0]
        max_y_pred=torch.max(max_y_pred,1,keepdim=True)[0]
        
        min_y_pred=torch.min(y_pred,2,keepdim=True)[0]
        min_y_pred=torch.min(min_y_pred,1,keepdim=True)[0]
        
        y_pred -= min_y_pred
        y_pred /= (max_y_pred-min_y_pred)
        
        
        sum_y_true=torch.sum(y_true,(1,2),keepdim=True)
        sum_y_pred=torch.sum(y_pred,(1,2),keepdim=True)
        
        y_true /= (sum_y_true + self.eps)
        y_pred /= (sum_y_pred + self.eps)
    
        return torch.mean(torch.sum(y_true * torch.log((y_true / (y_pred + self.eps)) + self.eps), (1,2)))

class MyCorrCoef(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyCorrCoef, self).__init__()

    def forward(self, pred, target):
        '''
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        CC = []

        for i in range(input.size(0)):
            im = input[i] - torch.mean(input[i])
            tm = target[i] - torch.mean(target[i])

            CC.append(-1.0*torch.sum(im * tm) / (torch.sqrt(torch.sum(im ** 2))
                                            * torch.sqrt(torch.sum(tm ** 2))))
            CC[i].unsqueeze_(0)

        CC = torch.cat(CC,0)
        CC = torch.mean(CC)

        return CC
        '''
        size = pred.size()
        new_size = (-1, size[-1] * size[-2])
        pred = pred.reshape(new_size)
        target = target.reshape(new_size)
    
        cc = []
        for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
            xm, ym = x - x.mean(), y - y.mean()
            r_num = torch.mean(xm * ym)
            r_den = torch.sqrt(
                torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
            r = -1.0*r_num / r_den
            cc.append(r)
    
        cc = torch.stack(cc)
        cc = cc.reshape(size[:2]).mean()
        return cc  # 1 - torch.square(r)

# Correlation Coefficient Loss
class Correlation_coefficient(nn.Module):
    def __init__(self):
        super(Correlation_coefficient,self).__init__()
        self.eps=1e-10
        
    def forward(self,y_pred,y_true):
        y_pred=torch.squeeze(y_pred)
        y_true=torch.squeeze(y_true)
        
        shape_r_out=y_pred.shape[1]
        shape_c_out=y_pred.shape[2]
        
        max_y_pred=torch.max(y_pred,2,keepdim=True)[0]
        max_y_pred=torch.max(max_y_pred,1,keepdim=True)[0]
        
        min_y_pred=torch.min(y_pred,2,keepdim=True)[0]
        min_y_pred=torch.min(min_y_pred,1,keepdim=True)[0]
        
        y_pred = y_pred-min_y_pred
        y_pred = y_pred/(max_y_pred-min_y_pred)
        
        
        sum_y_true=torch.sum(y_true,(1,2),keepdim=True)
        sum_y_pred=torch.sum(y_pred,(1,2),keepdim=True)
        
        y_true = y_true/(sum_y_true + self.eps)
        y_pred = y_pred/(sum_y_pred + self.eps)
    
        N = shape_r_out * shape_c_out
        sum_prod = torch.sum(y_true * y_pred,(1,2),keepdim=True)
        
        sum_x = torch.sum(y_true,(1,2),keepdim=True)
        sum_y = torch.sum(y_pred,(1,2),keepdim=True)
        
        sum_x_square = torch.sum(torch.square(y_true),(1,2),keepdim=True)
        sum_y_square = torch.sum(torch.square(y_pred),(1,2),keepdim=True)
    
        num = sum_prod - ((sum_x * sum_y) / N)
        
        den = torch.sqrt((sum_x_square - torch.square(sum_x) / N) * (sum_y_square - torch.square(sum_y) / N))
    
        return -1.0 * torch.mean(num / den)


# Normalized Scanpath Saliency Loss
class NSS(nn.Module):
    def __init__(self):
        super(NSS,self).__init__()
        self.eps=1e-10
        #self.flatten=nn.Flatten()
    
    def forward(self,pred,fixations):
        
        '''
        y_pred=torch.squeeze(y_pred)
        y_true=torch.squeeze(y_true)
        
        max_y_pred=torch.max(y_pred,2,keepdim=True)[0]
        max_y_pred=torch.max(max_y_pred,1,keepdim=True)[0]
        
        min_y_pred=torch.min(y_pred,2,keepdim=True)[0]
        min_y_pred=torch.min(min_y_pred,1,keepdim=True)[0]
        
        y_pred = y_pred-min_y_pred
        y_pred = y_pred/(max_y_pred-min_y_pred)
        
        
        #y_pred_flatten = self.flatten(y_pred)
    
        y_mean = torch.mean(y_pred, (1,2), keepdim=True)
        y_std = torch.mean(y_pred, (1,2), keepdim=True)

    
        y_pred = (y_pred - y_mean) / (y_std + self.eps)
    
        return -1.0*torch.mean(torch.sum(y_true * y_pred, (1,2)) / torch.sum(y_true, (1,2)))
        '''
        size = pred.size()
        new_size = (-1, size[-1] * size[-2])
        pred = pred.reshape(new_size)
        fixations = fixations.reshape(new_size)
    
        pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
        results = []
        for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                          torch.unbind(fixations, 0)):
            if mask.sum() == 0:
                print("No fixations.")
                results.append(torch.ones([]).float().to(fixations.device))
                continue
            
            nss_ = torch.masked_select(this_pred_normed, mask)
            nss_ = nss_.mean(-1)
            results.append(-1.0*nss_)
        results = torch.stack(results)
        results = results.reshape(size[:2]).mean()
        #results = torch.mean(results)
        
        return results

class MyNormScanSali(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyNormScanSali, self).__init__()

    def forward(self, input, target):

        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)

        NSS = []
        target_logic = torch.zeros(input.size(0), input.size(1) ).cuda()

        for i in range(input.size(0)):

            # normalize the predicted maps
            input_norm = (input[i] - torch.mean(input[i])) / torch.std(input[i])

            # compute the logic matrix of fixs
            for m in range(input.size(1)):
                if target[i,m] != 0:
                    target_logic[i,m] = 1

            NSS.append(torch.mean(-1.0*torch.mul(input_norm, target_logic[i])))
            NSS[i].unsqueeze_(0)

        NSS = torch.cat(NSS, 0)
        NSS = torch.mean(NSS)

        return NSS
    


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
