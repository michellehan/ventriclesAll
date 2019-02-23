import subprocess
import re
import argparse
import os
import shutil
import time
import math
import logging
import gc
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.dataset import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms.functional as visionF
from torchvision.transforms import ToPILImage
from PIL import Image

import models as models
#from models.unet import *
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

cudnn.benchmark = True
LOG = logging.getLogger('ventNormals')

args = None
best_dice = 0
global_step = 0

def main(context):
    global args
    global train_exp, exp
    global global_step
    global best_dice

    ############ set random seed for GPUs for reproducible
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


    #########################################################################
    ############ Define logging/running/etc #################################
    #########################################################################
    if args.flag == 'full': train_exp = '%s_%s_cls%d' %(args.arch, args.encoder, args.num_classes)
    else: train_exp = '%s_%s_cls%d' %(args.flag, args.encoder, args.num_classes)
    gen_exp = 'lr{}_d{}_m{}_b{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size)

    ############ setup output paths
    output_path = '../log/{}/{}'.format(train_exp, gen_exp)
    checkpoint_outputpath = '../ckpt/{}/{}'.format(train_exp, gen_exp)
    test_pred_path = '../test_pred/{}/{}'.format(train_exp, gen_exp)
    mask_path = '../masks/{}/{}'.format(train_exp, gen_exp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    if not os.path.exists(checkpoint_outputpath): os.makedirs(checkpoint_outputpath)
    if not os.path.exists(test_pred_path): os.makedirs(test_pred_path)
    if not os.path.exists(mask_path): os.makedirs(mask_path)
    parameters = subprocess.Popen("python %s" %args.parameters, shell=True, stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    parameters = parameters.replace(" --", "\n--")
    
    txt_file = os.path.join(output_path, 'log_{}.txt'.format(gen_exp))
    if args.log:
        assert not(args.evaluate * args.log)
        sys.stdout = open(txt_file, "w")
    LOG.info('\n\n\n==============================================================================')
    LOG.info('\n==============================================================================\n')

    ch_sysout(gen_exp, args)
    print('==============================================================================')
    print('==============================================================================\n')
    if not args.evaluate:
        LOG.info('Training Experiements Log Output Folder: {}'.format(output_path))
        LOG.info('Training Experiements Checkpoint Output Folder: {}\n'.format(checkpoint_outputpath))
        print('Training Experiements Log Output Folder: %s' %output_path)
        print('Training Experiements Checkpoint Output Folder: %s\n' %checkpoint_outputpath)
    else:
        LOG.info('Testing Prediction Results Output Folder: {}'.format(test_pred_path))
        LOG.info('Prediction Masks Results Output Folder: {}'.format(mask_path))
        print('Testing Prediction Results Output Folder:\t', test_pred_path)
        print('Prediction Masks Results Output Folder:\t', mask_path)
    LOG.info('Parameters file: {}'.format(args.parameters))
    LOG.info('\n{}'.format(parameters))
    print('Parameters file: %s' %args.parameters)
    print('%s' %parameters)



    #########################################################################
    ############ Create your model ##########################################
    #########################################################################
    dataset_config = datasets.__dict__[args.dataset]()
    modeldict_1 = setup(args, 1, output_path, checkpoint_outputpath, LOG, context)
    modeldict_2 = setup(args, 2, output_path, checkpoint_outputpath, LOG, context)
    modeldict_3 = setup(args, 3, output_path, checkpoint_outputpath, LOG, context)
    modeldict_4 = setup(args, 4, output_path, checkpoint_outputpath, LOG, context)
    modeldict_5 = setup(args, 5, output_path, checkpoint_outputpath, LOG, context)
    model_list = [modeldict_1, modeldict_2, modeldict_3, modeldict_4, modeldict_5]

    ch_sysout(gen_exp, args)
    for m, modeldict in enumerate(model_list):
        model = modeldict['model']
        optimizer = modeldict['optimizer']
        exp = modeldict['exp']
        checkpoint_path = modeldict['checkpoint_path']
        validation_log = modeldict['validation_log']

        if args.evaluate or args.resume != 0:
            args.resume = '{}/{}_{}.ckpt'.format(checkpoint_outputpath, args.ckpt, exp)

        ############ optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                if m == 0: LOG.info('=> loading checkpoints from: {}'.format(checkpoint_outputpath))
                best_dice, model, optimizer = load_checkpoint(args, model, optimizer, best_dice='overall')
            else: print("=> no checkpoint found at '{}'".format(args.resume))



    ############ For just saving masks
    if args.only_mask:
        if args.eval_tag is not None:
            res_exp = 'lr{}_d{}_m{}_b{}_{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size, args.eval_tag)
        else: res_exp = 'lr{}_d{}_m{}_b{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size)

        result = os.path.join(test_pred_path, 'test_{}.npz'.format(res_exp))
        m_result = os.path.join(test_pred_path, 'm_test_{}.npz'.format(res_exp))
        dice = float( np.load(result)['dice'] )
        m_dice = float( np.load(m_result)['dice'] )

        if dice >= m_dice:
            target = torch.tensor( np.load(result)['target'] )
            pred = torch.tensor( np.load(result)['pred'] )
            LOG.info('Saving masks for OVERALL ensemble [dice: {}]'.format(round(dice, 4)))
            print('\nSaving masks for OVERALL ensemble [dice: {}]'.format(round(dice, 4)))
        else:
            target = torch.tensor( np.load(m_result)['target'] )
            pred = torch.tensor( np.load(m_result)['pred'] )
            LOG.info('Saving masks for MODEL-WISE ensemble [dice: {}]'.format(round(m_dice, 4)))
            print('\nSaving masks for MODEL-WISE ensemble [dice: {}]'.format(round(dice, 4)))

        save_mask(args, target, pred, mask_path)
        return



    ############ For test dataset evaluation
    if args.evaluate:
        eval_loader = eval_create_data_loaders(**dataset_config, args=args)

        LOG.info("\nEvaluating the primary OVERALL model:")
        dice, target, pred, outnames, volumes = validate(eval_loader, model_list, validation_log, global_step, args.start_epoch, gen_exp, save_pred=True)
        if args.eval_tag is not None:
            res_exp = 'lr{}_d{}_m{}_b{}_{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size, args.eval_tag)
        else: res_exp = 'lr{}_d{}_m{}_b{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size)

        result_file = os.path.join(test_pred_path, 'test_{}.npz'.format(res_exp))
        print('Saving testing predction to: {}\n'.format(result_file))
        LOG.info('Saving testing predction to: {}\n'.format(result_file))
        np.savez(result_file, target=target, pred=pred, name=outnames, dice=dice)

        gc.collect()
        torch.cuda.empty_cache()

        for m, modeldict in enumerate(model_list):
            model = modeldict['model']
            optimizer = modeldict['optimizer']
            exp = modeldict['exp']
            checkpoint_path = modeldict['checkpoint_path']
            validation_log = modeldict['validation_log']

            args.resume = '{}/{}_{}.ckpt'.format(checkpoint_outputpath, 'm_' + args.ckpt, exp)
            if os.path.isfile(args.resume):
                if m == 0: LOG.info('=> loading checkpoints from: {}'.format(checkpoint_outputpath))
                best_dice_m, model, optimizer = load_checkpoint(args, model, optimizer, best_dice='model')
            else: print("=> no checkpoint found at '{}'".format(args.resume))

        LOG.info("\nEvaluating the primary MODEL-WISE model:")
        m_dice, m_target, m_pred, outnames, volumes = validate(eval_loader, model_list, validation_log, global_step, args.start_epoch, gen_exp, save_pred=True)

        if args.eval_tag is not None:
            res_exp = 'lr{}_d{}_m{}_b{}_{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size, args.eval_tag)
        else: res_exp = 'lr{}_d{}_m{}_b{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size)

        result_file = os.path.join(test_pred_path, 'm_test_{}.npz'.format(res_exp))
        print('Saving testing predction to: {}'.format(result_file))
        LOG.info('Saving testing predction to: {}\n'.format(result_file))
        np.savez(result_file, target=m_target, pred=m_pred, name=outnames, dice=m_dice)
       

        if dice >= m_dice: 
            LOG.info('Saving masks for OVERALL ensemble [dice: {}]'.format(round(dice, 4)))
            print('\nSaving masks for OVERALL ensemble [dice: {}]'.format(round(dice, 4)))
        else:
            target = m_target
            pred = m_pred
            LOG.info('Saving masks for MODEL-WISE ensemble [dice: {}]'.format(round(m_dice, 4)))
            print('\nSaving masks for MODEL-WISE ensemble [dice: {}]'.format(round(dice, 4)))
        save_mask(args, target, pred, mask_path)

        return




    #########################################################################
    ############# Train your model ##########################################
    #########################################################################
    
    #dataset_config = datasets.__dict__[args.dataset]()
    train_loader, val_loader = create_data_loaders(**dataset_config, args=args)

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
     
        for i, modeldict in enumerate(model_list):
            model = modeldict['model']
            optimizer = modeldict['optimizer']
            exp = modeldict['exp']
            training_log = modeldict['training_log']
            validation_log = modeldict['validation_log']

            train(train_loader, model, optimizer, epoch, training_log, exp)
            LOG.info("--- training epoch in %s seconds ---\n" % (time.time() - start_time))

#            is_best = True
#            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0 or epoch == 0:
#                save_checkpoint({
#                    'epoch': epoch + 1,
#                    'global_step': global_step,
#                    'arch': args.arch,
#                    'state_dict': model.state_dict(),
#                    'best_dice': best_dice,
#                    'optimizer' : optimizer.state_dict(),
#                }, is_best, checkpoint_outputpath, epoch + 1, exp)


        ############ evaluate as you go
        if (args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0) or epoch == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            val_dice = validate(val_loader, model_list, validation_log, global_step, epoch + 1, gen_exp)
            LOG.info("--- validation in %s seconds ---\n" % (time.time() - start_time))
           
            is_best = val_dice > best_dice
            best_dice = max(val_dice, best_dice)
            #LOG.info("--- current dice: %s\t best dice: %s\t is_best %s" %(val_dice, best_dice, is_best))
            
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0 or epoch == 0:
            for i, modeldict in enumerate(model_list):
                model = modeldict['model']
                optimizer = modeldict['optimizer']
                exp = modeldict['exp']

                val_dice_m = modeldict['val_dice_m']
                best_dice_m = modeldict['best_dice_m']
                is_best_m = val_dice_m > best_dice_m
                modeldict['best_dice_m'] = max(val_dice_m, best_dice_m)
                LOG.info("--- checkpoints saved: ---")

                #best ensemble performance
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_dice': best_dice,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_outputpath, epoch + 1, exp)

                #best individual performances
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_dice_m': modeldict['best_dice_m'],
                    'optimizer' : optimizer.state_dict(),
                }, is_best_m, checkpoint_outputpath, epoch + 1, exp, saveboth=False)

                LOG.info("\n")




def setup(args, modelnum, output_path, checkpoint_outputpath, LOG, context):
    if args.encoder: model_type = args.arch + "_" + args.encoder
    else: model_type = args.arch
    LOG.info("=> creating model '{}'".format(model_type))

    exp = 'lr{}_d{}_m{}_b{}_model{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size, modelnum)
    if args.logit_distance_cost > 0: exp = exp + "_res%.2f" %(args.logit_distance_cost)

    txt_file = os.path.join(output_path, 'log_{}.txt'.format(exp))
    if args.log:
        assert not(args.evaluate * args.log)
        sys.stdout = open(txt_file, "w")
        print('************* Log into txt file: %s' %(txt_file))
        print('************* Checkpoints saved to: %s/[checkpoint/best]_%s.ckpt' %(checkpoint_outputpath, exp))


    ############ Define logging files
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")

    model = models.__dict__[model_type](num_classes = args.num_classes)
    model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    best_dice_m = 0
    dt = {
          'model':model, 'optimizer':optimizer, 
          'best_dice_m':best_dice_m,
          'exp':exp, 'txt_file':txt_file, 'checkpoint_path':checkpoint_path,
          'training_log':training_log, 'validation_log':validation_log
          }
    return dt



def ch_sysout(exp, args):
    if args.flag == 'full': train_exp = '%s_%s_cls%d' %(args.arch, args.encoder, args.num_classes)
    else: train_exp = '%s_%s_cls%d' %(args.flag, args.encoder, args.num_classes)
    gen_exp = 'lr{}_d{}_m{}_b{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size)
    output_path = '../log/{}/{}'.format(train_exp, gen_exp)

    txt_file = os.path.join(output_path, 'log_{}.txt'.format(exp))
    #if args.log:
    #    assert not(args.evaluate * args.log)
    sys.stdout = open(txt_file, "a")








########################################################################
############# Load your data ############################################
#########################################################################

############ Create data loader for training and validation
def create_data_loaders(train_transformation,
                        target_transformation,
                        eval_transformation,
                        eval_target_transformation,
                        args):

    ############ training / testing diruse the same test dataset in official split
    print('\nTraining/Validation Dataset: ', args.raw_dir)
    print('Training/Validation Segmentations: ', args.segs_dir)
    print('Training csv: ', args.train_csv)
    print('Validation csv: ', args.val_csv, '\n')

    ############ Training dataset
    #train_dataset = datasets.Ventricles(args.train_csv, args.raw_dir, args.segs_dir, train_transformation, target_transformation, train=True)
    train_dataset = datasets.Ventricles(args.train_csv, args.raw_dir, args.segs_dir, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,      
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    ############ Validation dataset
    val_dataset = datasets.Ventricles(args.val_csv, args.raw_dir, args.segs_dir, eval_transformation, eval_target_transformation)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=2 * args.workers,  # Needs images twice as fast
                                            pin_memory=True,
                                            drop_last=False)    

    return train_loader, val_loader



############ Create data loader for evaluation
def eval_create_data_loaders(train_transformation,
                            target_transformation,
                            eval_transformation,
                            eval_target_transformation,
                            args):

    print('Test Dataset: ', args.raw_dir)
    print('Test Segmentations: ', args.segs_dir)
    print('Test csv: ', args.test_csv, '\n')

    eval_dataset = datasets.Ventricles(args.test_csv, args.raw_dir, args.segs_dir, eval_transformation, eval_target_transformation)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=2 * args.workers,  # Needs images twice as fast
                                                pin_memory=True,
                                                drop_last=False)

    return eval_loader






#########################################################################
############# Specify paramter updates ##################################
#########################################################################
def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epochi):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.25 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def dice_loss(pred, target):
    smooth = 1.

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = (iflat * iflat).sum()
    B_sum = (tflat * tflat).sum()
    #A_sum = iflat.sum()
    #B_sum = tflat.sum()
    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    score = 1 - dice.sum()
    return score





#########################################################################
############# Train your model ##########################################
#########################################################################
def train(train_loader, model, optimizer, epoch, log, exp):
    global global_step

    ch_sysout(exp, args)
    meters = AverageMeterSet()
    model.train()
    end = time.time()

    #criterion = nn.BCELoss()
    #class_weights = torch.FloatTensor([1, 25]).cuda()
    #m = nn.LogSoftmax()
    #criterion = nn.CrossEntropyLoss(weight=class_weights)

    for i, (input, target) in enumerate(train_loader):
        input_var = Variable(input).cuda()
        target_var = torch.autograd.Variable(target).float().cuda()

        ### target to onehot for model: nn.MultiLabelSoftMarginLoss
        #target_var = torch.autograd.Variable(target).long()
        #one_hot = torch.zeros( target_var.size(0), args.num_classes, target_var.size(2), target_var.size(3) )
        #target_onehot = one_hot.scatter_(1, target_var.data, 1)
        #target_var = Variable(target_onehot).cuda()

        #### set learning rate
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        ### model inference to output logits prediction
        model_out = model(input_var)
        model_out = torch.sigmoid(model_out) #since binary 
        loss = dice_loss(model_out, target_var)
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        ### compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### update EMA model with current primary model, not training EMA model
        global_step += 1

        ### measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()


        ############# print progress to console
        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'lr {meters[lr]:.6f}\t'
                'Loss {meters[loss]:.4f}\t'
                .format(epoch, i, len(train_loader), meters=meters)
            )

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })
        if i % (args.print_freq * 2) == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'lr {meters[lr]:.6f}\t'
                'Loss {meters[loss]:.4f}\t'
                .format(epoch, i, len(train_loader), meters=meters)
            )

        del input_var, target_var, model_out
        gc.collect()
        torch.cuda.empty_cache()




#########################################################################
############# Evaluate your model #######################################
#########################################################################
def validate(eval_loader, model_list, log, global_step, epoch, gen_exp, save_pred=False):
    #if torch.cuda.device_count() == 2:
    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')
    #if torch.cuda.device_count() == 3:
    #cuda2 = torch.device('cuda:2')

    meters = AverageMeterSet()
    #model.eval() #switch to evaluate mode

    if save_pred: tag = "Test"
    else: tag = "Val"

    with torch.no_grad():
        target_total = torch.randn(0, 1).double().cpu()
        pred_total = torch.randn(0, 1).double().cpu()
        for _, modeldict in enumerate(model_list):
            modeldict['target_total_m'] = torch.randn(0,1).double().cuda()
            modeldict['pred_total_m'] = torch.randn(0,1).double().cuda(cuda1)

        end = time.time()
        index = 0
        
        volumes = {}
        subjs = []
        for i, (input, target) in enumerate(eval_loader):
            gc.collect()
            torch.cuda.empty_cache()

            model_total= torch.zeros( Variable(input).size() ).float().cuda(cuda1)
            targets = torch.autograd.Variable(target).float().cuda()

            for _, modeldict in enumerate(model_list):
                model = modeldict['model']
                exp = modeldict['exp']
                target_total_m = modeldict['target_total_m']
                pred_total_m = modeldict['pred_total_m']
                model.eval() 

                model_output = model(Variable(input).cuda())
                model_output = torch.sigmoid(model_output) #since binary 
                #model_output = torch.round(model_output)
                model_total += model_output.cuda(cuda1)

                model_loss = dice_loss(model_output, targets)
                model_dice = 1 - model_loss
                ch_sysout(exp, args)
                if i % 50 == 0:
                    print('[{0}][{1}/{2}]\t'
                          'Dice {3}\t'
                           .format(tag, i, len(eval_loader), round(model_dice.item(), 4) ))
            
                modeldict['target_total_m'] = torch.cat((target_total_m, targets.data.double().cuda()), 0)
                modeldict['pred_total_m'] = torch.cat((pred_total_m, model_output.data.double().cuda(cuda1)), 0)
                del model_output, target_total_m, pred_total_m
            

            model_preround = model_total / len(model_list)
            model_preround = model_preround[:,0,:,:].unsqueeze(1)
            model_out = torch.round( model_preround ) #get average of predictions\
            target_total = torch.cat((target_total, targets.data.double().cpu()), 0)
            pred_total = torch.cat((pred_total, model_out.data.double().cpu()), 0)

            #loss = dice_loss(model_out, targets)
            loss = dice_loss( model_preround, targets.cuda(cuda1))
            dice = 1 - loss
            meters.update('loss', loss.item())


            del input, target, model_total, model_preround
            gc.collect()
            torch.cuda.empty_cache()
            
            ############# print progress to console
            ch_sysout(gen_exp, args)
            if i % 50 == 0:
                LOG.info('[{0}][{1}/{2}]\t'
                         'Dice {3}\t'
                          .format(tag, str(i).zfill(3), len(eval_loader), round(dice.item(), 4) ))
                print('[{0}][{1}/{2}]\t'
                      'Dice {3}\t'
                       .format(tag, str(i).zfill(3), len(eval_loader), round(dice.item(), 4) ))


            if save_pred:
                outnames = pd.read_csv(args.test_csv, header=None)    
                target_var = targets.cpu()            
                pred_var = model_out.cpu()

                for sample in range(pred_var.size(0)):
                    outname = outnames.iloc[index,0].split(".")[0]
                    pred = float( pred_var[sample].sum() )
                    target = float( target_var[sample].sum() )

                    subjname = outname.split("-")[0]
                    if subjname not in subjs: 
                        subjs.append(subjname)
                        volumes[subjname] = [pred, target] 
                    else:
                        volumes[subjname] = [volumes[subjname][0] + pred, 
                                             volumes[subjname][1] + target]
                    
                    index += 1

            del model_out, targets
            gc.collect()
            torch.cuda.empty_cache()


        #import matplotlib.pyplot as plt
        #preds = pred_total.cpu().numpy().flatten()
        #r = np.random.choice(preds.shape[0], 5000, replace=False)
        #p = preds[r]
        #plt.hist(p, bins="auto")
        #plt.show()
        overall_loss = dice_loss(pred_total.cuda(cuda1), target_total.cuda(cuda1))
        overall_dice = 1 - overall_loss
        overall_dice = round(overall_dice.item(), 4)

        ch_sysout(gen_exp, args)
        if save_pred: 
            LOG.info('[{}] Overall Test Dice score: {}\n\n'.format(epoch, overall_dice))
            print('[{}] Overall Test Dice score: {}\n'.format(epoch, overall_dice))
         
            for subj, vols in volumes.items():
                pred = vols[0]
                target = vols[1]
                diff = round( abs(pred - target) / target * 100 , 2)
                print('[%s]\tpred: %s\ttarget: %s\terror: %s percent' 
                      %(subj, pred, target, diff))

            return overall_dice, target_total.detach(), pred_total.detach(), outnames, volumes
        else: 
            for _, modeldict in enumerate(model_list):
                exp = modeldict['exp']
                target_total_m = modeldict['target_total_m']
                pred_total_m = modeldict['pred_total_m'] 
                overall_loss_m = dice_loss(pred_total_m.cuda(), target_total_m.cuda())
                overall_dice_m = 1 - overall_loss_m
                overall_dice_m = round(overall_dice_m.item(), 4)

                modeldict['val_dice_m'] = overall_dice_m
                ch_sysout(exp, args)
                print('Overall Validation Dice score: {}\n'.format(overall_dice_m))

            ch_sysout(gen_exp, args)
            LOG.info('Overall Validation Dice score: {}'.format(overall_dice))
            print('[{}] Overall Validation Dice score: {}\n'.format(epoch, overall_dice))
            return overall_dice



def save_mask(args, pred_var, target_var, mask_path):
    outnames = pd.read_csv(args.test_csv, header=None)
    subjs = []
    total = pred_var.size(0)

    for sample in range(total):
        outname = outnames.iloc[sample,0].split(".")[0]
        outpath = mask_path + "/" + outname + ".png"
        outpath_pred = mask_path + "/pred_" + outname + ".png"

        subjname = outname.split("-")[0]
        if subjname not in subjs: 
            subjs.append(subjname)
            LOG.info('[{}/{}] {}'.format(sample, total, subjname))

        #pred = torch.round(pred_var[sample]).float().cpu()
        pred = pred_var[sample].float().cpu()
        target = target_var[sample].float().cpu()

        TP = torch.mul(pred, target)
        FP = pred - TP
        FN = target - TP

        TP = ToPILImage()(TP).convert('RGB') #green
        FP = ToPILImage()(FP).convert('RGB') #red
        FN = ToPILImage()(FN).convert('RGB') #blue

        numpixels = TP.size[0] * TP.size[1]
        colors = []
        mask_colors = []
        for pixel in range(numpixels):
            tp = TP.getdata()[pixel]
            fp = FP.getdata()[pixel]
            fn = FN.getdata()[pixel]

            if tp > (0,0,0): 
                colors.append( (65,190,215) )
                mask_colors.append( (65,190,215) )
            elif fp > (0,0,0): 
                colors.append( (250,180,60) )
                mask_colors.append( (0,0,0) )
            elif fn > (0,0,0): 
                colors.append( (200,60,85) )
                mask_colors.append( (0,0,0) )
            else: 
                colors.append( (0,0,0) )
                mask_colors.append( (0,0,0) )
            #colors.append( (fp[0], tp[0], fn[0]) )
            
        
        mask = Image.new(TP.mode, TP.size)
        mask.putdata(colors)
        mask.save(outpath)

        mask_pred = Image.new(TP.mode, TP.size)
        mask_pred.putdata(mask_colors)
        mask_pred.save(outpath_pred)


        del target, pred, TP, FP, FN, colors, mask, mask_pred
        gc.collect()
        torch.cuda.empty_cache()

    LOG.info('==============================================================================')
    LOG.info('==============================================================================\n\n\n\n\n')
    return







#########################################################################
############# Loading and saving ########################################
#########################################################################
def save_checkpoint(state, is_best, dirpath, epoch, exp, saveboth=True):
    filename = '{}/{}_{}.ckpt'                 ### only save the last checkpoint and the best one (best EMA-prec1)
    checkpoint_path = filename.format(dirpath, 'checkpoint', exp)
    best_path = filename.format(dirpath, 'best', exp)

    if saveboth:
        torch.save(state, checkpoint_path)
        LOG.info("%s" % checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, best_path)
            LOG.info("%s" % best_path)
        #else: LOG.info("\n")
    else:
        if is_best:
            best_path = filename.format(dirpath, 'm_best', exp)
            torch.save(state, best_path)
            LOG.info("%s" % best_path)

def load_checkpoint(args, model, optimizer, best_dice='overall'):
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if best_dice == "model": best_dice = checkpoint['best_dice_m']
        else: best_dice = checkpoint['best_dice']
        
        LOG.info("{} (epoch {})".format(args.resume, checkpoint['epoch']))
        print("{} (epoch {})".format(args.resume, checkpoint['epoch']))
        return best_dice, model, optimizer

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
