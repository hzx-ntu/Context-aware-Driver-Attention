
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

#import _init_paths
from lib.config import cfg
from lib.config import update_config

from lib.core.loss import SalKLCCLoss,JOINTSALLoss
from lib.core.function_bdda_eye import train_sal,validate_sal,train_saleye,validate_saleye

from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary
from lib.dataset.bdda import BDDA
from lib.dataset.bddad import BDDAD

import lib.models as models
from lib.models.eyemodel_sal import EYESALModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = EYESALModel(cfg)


    # define loss function (criterion) and optimizer
    criterion_sal = SalKLCCLoss().cuda()
    criterion_saleye = JOINTSALLoss().cuda()

    # Data loading code
    
    train_dataset_bdda = BDDA(split='train', aug = True)
    valid_dataset_bdda = BDDA(split='test', aug = False)
    
   
    train_loader_bdda = torch.utils.data.DataLoader(
        train_dataset_bdda,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=True,
        num_workers=cfg.WORKERS, pin_memory=True)

    valid_loader_bdda = torch.utils.data.DataLoader(
        valid_dataset_bdda,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=False,
        num_workers=cfg.WORKERS, pin_memory=True)
    
    
    train_dataset_bddad = BDDAD(split='train')
    valid_dataset_bddad = BDDAD(split='test')
    
   
    train_loader_bddad = torch.utils.data.DataLoader(
        train_dataset_bddad,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=True,
        num_workers=cfg.WORKERS, pin_memory=True)

    valid_loader_bddad = torch.utils.data.DataLoader(
        valid_dataset_bddad,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=False,
        num_workers=cfg.WORKERS, pin_memory=True)

    best_perf = 6.0
    best_model = False
    last_epoch = -1
    
    optimizer_sal = torch.optim.Adam(model.salModel.parameters(),lr=1e-4) ####
    optimizer_saleye = torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9, weight_decay=1e-4)
    
    doTest=True
    if doTest:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
        #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    
    else:
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
        )
    
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
    
            optimizer_saleye.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
    
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_saleye, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    
    
    
    load_eyemodel=False
    if load_eyemodel:
        saved = load_checkpoint_eyemodel()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']

            try:
                model.module.load_state_dict(state,strict=False)
            except:
                model.load_state_dict(state,strict=False)
            
            epoch = saved['epoch']
        else:
            print('Warning: Could not read checkpoint!')
    
    if doTest:
        # evaluate on validation set
        perf_indicator = validate_saleye(
            cfg, valid_loader_bddad, model, criterion_saleye,
            final_output_dir, tb_log_dir
        )
        
        print(perf_indicator)
        return
    

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()
        
        # train sal module for one epoch
        if epoch==0:
           train_sal(cfg, train_loader_bdda, model, criterion_sal, optimizer_sal, epoch,
                final_output_dir, tb_log_dir)
        
        # train for one epoch
        train_saleye(cfg, train_loader_bddad, model, criterion_saleye, optimizer_saleye, epoch,
              final_output_dir, tb_log_dir)
        

        # evaluate on validation set
        perf_indicator = validate_saleye(
            cfg, valid_loader_bddad, model, criterion_saleye,
            final_output_dir, tb_log_dir
        )

        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer_saleye.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)

def load_checkpoint_eyemodel(filename='checkpoint_org.pth.tar'):
    filename = os.path.join('./output/bddad', filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state

if __name__ == '__main__':
    main()
