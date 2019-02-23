import subprocess
import re
import argparse
import os
import shutil
import time
import math
import logging
import random
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

    ############ setup output paths
    output_path = '../log/{}'.format(train_exp)
    checkpoint_outputpath = '../ckpt/{}'.format(train_exp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    if not os.path.exists(checkpoint_outputpath): os.makedirs(checkpoint_outputpath)
    print('Training Experiements Log Output Folder：  ', output_path)
    print('Training Experiements Checkpoint Output Folder：  ', checkpoint_outputpath)

    exp = 'lr{}_d{}_m{}_b{}'.format(args.lr, args.lr_decay, args.momentum, args.batch_size)
    if args.logit_distance_cost > 0: exp = exp + "_res%.2f" %(args.logit_distance_cost)


    if args.log:
        assert not(args.evaluate * args.log)
        txt_file = os.path.join(output_path, 'log_{}.txt'.format(exp))
        sys.stdout = open(txt_file, "w")
        print('************* Log into txt file: %s' %(txt_file))
        print('************* Checkpoints saved to: %s/[checkpoint/best]_%s.ckpt' %(checkpoint_outputpath, exp))
    if args.evaluate:
        #test_pred_path = '../test_pred/{}'.format(train_exp)
        #mask_path = '{}/{}'.format(args.mask_dir, train_exp)
        test_pred_path = '../test_pred/{}'.format(exp)
        mask_path = '../masks/{}'.format(exp)
        if not os.path.exists(test_pred_path): os.makedirs(test_pred_path)
        if not os.path.exists(mask_path): os.makedirs(mask_path)
        print('Testing Prediction Results Output Folder：  ', test_pred_path)
        print('Prediction Masks Results Output Folder：  ', mask_path)


    ############ Define logging files
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")

    if args.parameters is not None:
        parameters = subprocess.Popen("python %s" %args.parameters, shell=True, stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
        parameters = parameters.replace(" --", "\n--")
        LOG.info('parameters provided: \n{0}\n'.format(parameters))
        print('\nparameters provided: \n%s' %parameters)

    ############ Load dataset config (data_path, transform...)
    dataset_config = datasets.__dict__[args.dataset]()






    #########################################################################
    ############ Create your model ##########################################
    #########################################################################
    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    model = models.__dict__[args.arch](encoder = args.encoder, num_classes = args.num_classes)
    model = nn.DataParallel(model).cuda()


    #########################################################################
    ############ Define loss function (criterion) and optimizer ############
    #########################################################################   
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.evaluate:
        args.resume = '{}/{}_{}.ckpt'.format(checkpoint_outputpath, args.ckpt, exp)

    ############ optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            LOG.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_dice = checkpoint['best_dice']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    ############ For test dataset evaluation
    if args.evaluate:
        print('Evaluation {}.ckpt'.format(args.ckpt))
        eval_loader = eval_create_data_loaders(**dataset_config, args=args)

        LOG.info("Evaluating the primary model:")
        #_, auc, target, pred = validate(eval_loader, model, validation_log, global_step, args.start_epoch, save_pred=True)
        dice, target, pred, outnames = validate(eval_loader, model, validation_log, global_step, args.start_epoch, save_pred=True, mask_path=mask_path)

        #target = target.astype(np.int32, copy=False)
        result_file = os.path.join(test_pred_path, 'test_{}.npz'.format(exp))
        print('Saving testing predction to: {}'.format(result_file))
        np.savez(result_file, target=target, pred=pred, name=outnames, dice=dice)



        return



    #########################################################################
    ############# Train your model ##########################################
    #########################################################################
    train_loader, val_loader = create_data_loaders(**dataset_config, args=args)
    
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        
        ############ train for one epoch
        train(train_loader, model, optimizer, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---\n" % (time.time() - start_time))

        ############ evaluate as you go
        if (args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0) or epoch == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            val_dice = validate(val_loader, model, validation_log, global_step, epoch + 1)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
           
            is_best = val_dice > best_dice
            best_dice = max(val_dice, best_dice)
            #LOG.info("--- current dice: %s\t best dice: %s\t is_best %s" %(val_dice, best_dice, is_best))
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0 or epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_dice': best_dice,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_outputpath, epoch + 1)






#########################################################################
############# Load your data ############################################
#########################################################################

############ Create data loader for training and validation
def create_data_loaders(train_transformation,
                        target_transformation,
                        eval_transformation,
                        args):

    ############ training / testing diruse the same test dataset in official split
    print('Training/validation Dataset: ', args.raw_dir)
    print('Training/validation Segmentations: ', args.segs_dir)
    print('Training csv: ', args.train_csv)
    print('Validation csv: ', args.val_csv, '\n')

    ############ Training dataset
    train_dataset = datasets.Ventricles(args.train_csv, args.raw_dir, args.segs_dir, train_transformation, target_transformation, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,      
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    ############ Validation dataset
    val_dataset = datasets.Ventricles(args.val_csv, args.raw_dir, args.segs_dir, eval_transformation, target_transformation)
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
                            args):

    print('Test Dataset: ', args.raw_dir)
    print('Test Segmentations: ', args.segs_dir)
    print('Test csv: ', args.test_csv, '\n')

    eval_dataset = datasets.Ventricles(args.test_csv, args.raw_dir, args.segs_dir, eval_transformation, target_transformation)
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

    #score = 1 - ( dice.sum() / target.size(0) )
    score = 1 - dice.sum()
    return score


def tversky_loss(y_pred, y_true):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    #num = K.sum(p0*g0, (0,1,2,3))
    #den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T



#########################################################################
############# Train your model ##########################################
#########################################################################
def train(train_loader, model, optimizer, epoch, log):
    global global_step

    meters = AverageMeterSet()
    model.train()
    end = time.time()

    #criterion = nn.BCELoss()
    class_weights = torch.FloatTensor([1, 25]).cuda()
    m = nn.LogSoftmax()
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    for i, (input, target) in enumerate(train_loader):
        ### target to onehot for model: nn.MultiLabelSoftMarginLoss
        target_var = torch.autograd.Variable(target).float().cuda()

        #target_var = torch.autograd.Variable(target).long()
        #one_hot = torch.zeros( target_var.size(0), args.num_classes, target_var.size(2), target_var.size(3) )
        #target_onehot = one_hot.scatter_(1, target_var.data, 1)
        #target_var = Variable(target_onehot).cuda()

        input_var = Variable(input).cuda()

        #### set learning rate
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        ### model inference to output logits prediction
        model_out = model(input_var)


        #LOG.info('target shape', target_var.shape)
        #LOG.info('target', target_var[0])
        #LOG.info('model shape', model_out.shape)
        #LOG.info('model', model_out[0])

        #model_out = torch.sigmoid(model_out) #since binary 
        #loss = dice_loss(model_out, target_var)

        #LOG.info('model sigmoid', model_out[0])

        loss = criterion(model_out, target_var.long().squeeze(1))


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
                .format(epoch, i, len(train_loader), meters=meters))

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })






#########################################################################
############# Evaluate your model #######################################
#########################################################################
#def validate(eval_loader, model, log, global_step, epoch, save_pred=False):
#    meters = AverageMeterSet()
#    model.eval() #switch to evaluate mode
#
#    end = time.time()
#    for i, (input, target) in enumerate(eval_loader):
#        target_var = torch.autograd.Variable(target).long()
#        one_hot = torch.zeros( target_var.size(0), args.num_classes, target_var.size(2), target_var.size(3) )
#        target_onehot = one_hot.scatter_(1, target_var.data, 1)

#        input_var = Variable(input).cuda()
#        target_var = Variable(target_onehot).cuda()

        ### compute output and loss
 #       model_output = model(input_var)
 #       loss = dice_loss(model_output, target_var)
 #       meters.update('loss', loss.item())

        ############# print progress to console
 #       if i % args.print_freq == 0:
 #           LOG.info(
 #               'Test: [{0}/{1}]\t'
 #               'Class {meters[loss]:.4f}\t'
 #               .format(i, len(eval_loader), meters=meters))


 #   print('[{}/{}]: Dice score: {} %\n'.format(epoch, args.epochs, args.num_classes, loss))
 #   log.record(epoch, {'step': global_step})
 #   return loss




#########################################################################
############# Evaluate your model #######################################
#########################################################################
def validate(eval_loader, model, log, global_step, epoch, save_pred=False, mask_path=None):
    meters = AverageMeterSet()
    model.eval() #switch to evaluate mode

    target_total = torch.randn(0, 1).double().cuda()
    pred_total = torch.randn(0, 1).double().cuda()
    

    end = time.time()
    index = 0
    for i, (input, target) in enumerate(eval_loader):
        inputs = Variable(input).cuda()
        targets = torch.autograd.Variable(target).float().cuda()

        #target_var = torch.autograd.Variable(target).long()
        #one_hot = torch.zeros( target_var.size(0), args.num_classes, target_var.size(2), target_var.size(3) )
        #target_onehot = one_hot.scatter_(1, target_var.data, 1)
        #target_var = Variable(target_onehot).cuda()

        ### compute output and loss
        model_output = model(inputs)
        #LOG.info('model shape', model_output.shape)
        #LOG.info('mode', model_output[0,:,0:10,0:10])
        #LOG.info('targets', targets[0,:,0:10,0:10])


        model_out = torch.sigmoid(model_output) #since binary 
        model_out = model_out[:,1,:,:].unsqueeze(1)
        #LOG.info('mode', model_out[0,:,0:10,0:10])
        #LOG.info('model squeeze', model_out.squeeze(1)[0,:,0:10,0:10])
        #t = torch.Tensor([0.55]).cuda()
        #model_out = (model_out > t).float() * 1
        #model_out = torch.round(model_out)
        
        #correction = torch.Tensor([1]).cuda()
        #model_out = torch.round(correction - model_out)
        #print('model corrected', model_out[0])
        #print('targets', targets[0])

        loss = dice_loss(model_out, targets)
        dice = 1 - loss
        meters.update('loss', loss.item())

        target_total = torch.cat((target_total, targets.data.double()), 0)
        pred_total = torch.cat((pred_total, model_out.data.double()), 0)
        #LOG.info('loss = %s' %loss.item())
        #LOG.info('dice = %s' %dice)

        #print('\ntargets', targets.shape)
        #print('model', model_out.shape)
        #print('targets total', target_total.shape)
        #print('pred total', pred_total.shape)

        outnames = pd.read_csv(args.test_csv, header=None)
        target_var = targets.cpu()            
        pred_var = model_out.cpu()
        #if save_pred:
        if False:
            #sm = nn.Softmax()
            #sm_output= sm(model_output)
            #model_sigmoid = torch.sigmoid(model_output) #since binary 

            #target_var = targets.cpu()            
            #model_pred = model_out.cpu()

            #print(model_pred.shape[0])
            #print(model_pred[0])
            #print(target[0])

            #model_pred = np.argmax(model_sigmoid.detach(), axis = 1) #un one hot?
            #model_pred = torch.cat((model_pred, model_sigmoid.data.double()), 0)
            #print(model_pred)

            for sample in range(model_pred.size(0)):
                outname = outnames.iloc[index,0].split(".")[0]
                outpath = mask_path + "/" + outname + ".png"

                pred = pred_var[sample].float()
                target = target_var[sample].float()
            
                TP = torch.mul(pred, target)
                FP = pred - TP
                FN = target - TP


                #print('pred', model_pred[0].float())
                #print('target', target_var[0].float())

                ######
                #pred = model_pred[sample].unsqueeze(0)
                #print(pred)
                #pred = pred.data.cpu()
                #print(pred)
                #print(pred.size())
                #print(target.size())
                #print(pred.type())
                #pixels = visionF.to_pil_image(pred).convert('RGB')
                #pixels = ToPILImage()(pred).convert('RGB')

                TP = ToPILImage()(TP).convert('RGB')
                FP = ToPILImage()(FP).convert('RGB')
                FN = ToPILImage()(FN).convert('RGB')

                numpixels = TP.size[0] * TP.size[1]
                colors = []
                for pixel in range(numpixels):
                    tp = TP.getdata()[pixel]
                    fp = FP.getdata()[pixel]
                    fn = FN.getdata()[pixel]

                    colors.append( (fp[0], tp[0], fn[0]) )
                    
                    #p = pred[pixel]
                    #pred_by_pixel.append(p)
                    #colors.append( (color[0], 0, 0) )
                #    if color not black: colors.append(red)
                #    else: colors.append(black)

                mask = Image.new(TP.mode, TP.size)
                mask.putdata(colors)
                mask.save(outpath)

                index = index + 1


            #target_total = torch.cat((target_total, target_var.data.double()), 0)
            #pred_total = torch.cat((pred_total, model_pred.data.double()), 0)

            #LOG.info('pred preview {}'.format(pred_total.data))
            #LOG.info('model_output.size {}'.format(model_output.size()))
            #LOG.info('model_sigmoid.size {}'.format(model_sigmoid.size()))
        

        ############# print progress to console
        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Dice {2}\t'
                .format(i, len(eval_loader), round(dice.item(), 4) )
            )

    #import matplotlib.pyplot as plt
    #preds = pred_total.cpu().numpy()
    #plt.hist(preds.flatten(), bins="auto")
    #plt.show()

    overall_loss = dice_loss(pred_total, target_total)
    overall_dice = 1 - overall_loss

    print('[{}/{}] Overall Dice score: {}\n'.format(epoch, args.epochs, round(overall_dice.item(), 4) ))
    if args.log: LOG.info('[{}/{}] Overall Dice score: {}'.format(epoch, args.epochs, round(overall_dice.item(), 4) ))


    if save_pred: 
        return overall_dice.item(), target_var.detach(), pred_var.detach(), outnames
    else: 
        return overall_dice.item()


#########################################################################
############# Loading and saving ########################################
#########################################################################
def save_checkpoint(state, is_best, dirpath, epoch):
    filename = '{}/{}_{}.ckpt'                 ### only save the last checkpoint and the best one (best EMA-prec1)
    checkpoint_path = filename.format(dirpath, 'checkpoint', exp)
    best_path = filename.format(dirpath, 'best', exp)

    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---\n" % best_path)
    else: LOG.info("\n")

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
