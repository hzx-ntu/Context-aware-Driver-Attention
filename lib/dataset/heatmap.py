import numpy as np
from math import exp, log, sqrt, ceil
from matplotlib import pyplot as plt

from scipy.stats import multivariate_normal

def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))

def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)
  
def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    # k1 = multivariate_normal(mean=m1, cov=593.109206084)
    k1 = multivariate_normal(mean=m1, cov=s1)
    #     zz = k1.pdf(array_like_hm)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height,width))
    return img

def test(width, height, x, y, array_like_hm):
    dmax = 10
    edge_value = 0.00001
    sigma = cal_sigma(dmax, edge_value)
    
    return draw_heatmap(width, height, x, y, sigma, array_like_hm)

def gen_heatmap(width,height,x,y,sigma):
    #edited by hu zhongxu, which to generate the heatmap
    m1=(x,y)
    s1=np.eye(2)*pow(sigma,2)

    #k1=multivariate_normal(mean=m1,cov=s1)

    x_rg=np.arange(width,dtype=np.float)
    y_rg=np.arange(height,dtype=np.float)
    xx,yy=np.meshgrid(x_rg,y_rg)

    xxyy=np.c_[xx.ravel(),yy.ravel()]

    zz=gaussian(xxyy,m1,sigma)
    img=zz.reshape((height,width))

    return img
def gen_heatmap_ang(width,height,x,y,r,value):
    m1=(x,y)
    x_rg=np.arange(width,dtype=np.float)
    y_rg=np.arange(height,dtype=np.float)
    xx,yy=np.meshgrid(x_rg,y_rg)

    xxyy=np.c_[xx.ravel(),yy.ravel()]
    
    xxyy -= m1
    x_term = xxyy[:,0] ** 2
    y_term = xxyy[:,1] ** 2
    
    xy_term=x_term+y_term
    
    exp_value=[value if xy<r**2 else 0 for xy in xy_term]
    exp_value=np.array(exp_value)

    return exp_value.reshape((height,width))


