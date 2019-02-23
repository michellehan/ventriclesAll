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

import models as models
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
    if args.flag == 'full': train_exp = 'full_cls%d' %(args.num_classes)
    else: train_exp = '%s_%.2f_cls%d' %(args.flag, args.train_portion, args.num_classes)

    ############ setup output paths
    output_path = '../log/{}'.format(train_exp)
    checkpoint_outputpath = '../ckpt/{}'.format(train_exp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    if not os.path.exists(checkpoint_outputpath): os.makedirs(checkpoint_outputpath)
    print('Training Experiements Log Output Folder：  ', output_path)
    print('Training Experiements Checkpoint Output Folder：  ', checkpoint_outputpath)

    #exp = '{}_{}_b{}_label{}_{}'.format(args.flag, args.arch, args.batch_size, args.labeled_batch_size, args.final_model)
    exp = '{}_lr{}_m{}_b{}'.format(args.arch, args.lr, args.momentum, args.batch_size)
    if args.logit_distance_cost > 0: exp = exp + "_res%.2f" %(args.logit_distance_cost)


    if args.log:
        assert not(args.evaluate * args.log)
        txt_file = os.path.join(output_path, 'log_{}.txt'.format(exp))
        sys.stdout = open(txt_file, "w")
        print('************* Log into txt file: %s' %(txt_file))

    if args.evaluate:
        test_pred_path = '../test_pred/{}'.format(train_exp)
        if not os.path.exists(test_pred_path): os.makedirs(test_pred_path)
        print('Testing Prediction Results Output Folder：  ', test_pred_path)


    ############ Define logging files
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")

    if args.parameters is not None:
        parameters = subprocess.Popen("python %s" %args.parameters, shell=True, stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
        parameters = parameters.replace(" --", "\n--")
        LOG.info('parameters provided: {0}\n'.format(parameters))
        print('\nparameters provided: %s\n' %parameters)

    ############ Load dataset config (data_path, transform...)
    dataset_config = datasets.__dict__[args.dataset]()
    #print('dataset_config', dataset_config)
    #LOG.info('dataset_config: {}'.format(dataset_config))





    #########################################################################
    ############ Create your model ##########################################
    #########################################################################
    LOG.info("=> creating model '{arch}'".format(arch=args.arch))
    model = models.__dict__[args.arch](num_classes = args.num_classes)
    #model = [args.arch]()
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
        _, auc, target, pred = validate(eval_loader, model, validation_log, global_step, args.start_epoch, model_flag='Primary Model', save_pred=True)

        target = target.astype(np.int32, copy=False)
        result_file = os.path.join(test_pred_path, 'test_{}.npz'.format(exp))
        print('Saving testing predction to: {}'.format(result_file))
        np.savez(result_file, target=target, pred=pred, auc=auc)

        return




    #########################################################################
    ############# Train your model ##########################################
    #########################################################################
    train_loader, val_loader = create_data_loaders(**dataset_config, args=args)

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        ############ train for one epoch
        train(train_loader, model, optimizer, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        ############ evaluate as you go
        if (args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0) or epoch == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            val_dice = validate(val_loader, model, validation_log, global_step, epoch + 1, model_flag='Primary Model')
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))

            is_best = val_dice > best_dice
            best_dice = max(val_dice, best_dice)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
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
    print('Validation csv: ', args.val_csv)

    #LOG.info('train trasnform: {}'.format(train_transformation))
    #LOG.info('val transform: {}'.format(eval_transformation))

    ############ Customized training dataset
    train_dataset = datasets.Ventricles(args.train_csv, args.raw_dir, args.segs_dir, train_transformation, target_transformation)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,      ### no custormized sampler, just batchsize
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    ############ Customized validation dataset
    val_dataset = datasets.Ventricles(args.val_csv, args.raw_dir, args.segs_dir, eval_transformation, target_transformation)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=2 * args.workers,  # Needs images twice as fast
                                            pin_memory=True,
                                            drop_last=False)        # set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size

    #args.class_to_idx = train_dataset.class_to_idx
    return train_loader, val_loader






############ Create data loader for evaluation
def eval_create_data_loaders(train_transformation,
                            eval_transformation,
                            args):

    print('Test Dataset: ', args.raw_dir)
    print('Test Segmentations: ', args.segs_dir)
    print('Test csv: ', args.test_csv)

    eval_dataset = datasets.IVCdataset(args.test_csv, args.raw_dir, args.segs_dir, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=2 * args.workers,  # Needs images twice as fast
                                                pin_memory=True,
                                                drop_last=False)

    #args.class_to_idx = eval_dataset.class_to_idx
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
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    #A_sum = torch.sum(tflat * iflat)
    #B_sum = torch.sum(tflat * tflat)
    #dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    #return 1 - dice


    dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    score = 1 - dice.sum() / target.size(0)

    return score


#########################################################################
############# Train your model ##########################################
#########################################################################
def train(train_loader, model, optimizer, epoch, log):
#def train(train_loader, epoch, **kwargs):
    global global_step

    meters = AverageMeterSet()
    model.train()
    end = time.time()


    for i, (input, target) in enumerate(train_loader):
        ### target to onehot for model: nn.MultiLabelSoftMarginLoss
        target_var = torch.autograd.Variable(target).long()

        onehot_dim = (target_var.size(0), args.num_classes, target_var.size(2), target_var.size(3))
        one_hot = torch.zeros(onehot_dim)
        target_onehot = one_hot.scatter_(1, target_var.data, 1)
        #LOG.info('target_onehot size {}'.format(target_onehot.size()))

        input_var = Variable(input).cuda()
        labeled_target_var = Variable(target_onehot).cuda()



        #### set learning rate
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        ### extract labeled samples from each batch
        #input_var = torch.autograd.Variable(input)

        #labeled_target = target[-args.labeled_batch_size:, ...]
        #labeled_target_var = torch.autograd.Variable(labeled_target.cuda(async=True))
        #labeled_target_var = torch.autograd.Variable(target)
        


        #LOG.info('input var {}'.format(input_var.size()))
        #LOG.info('labeled_target_var {}'.format(labeled_target_var.size()))

        ############# sanity check for labeled_target_var
        # print(labeled_target_var.data)
        # print(target_var.data)
        # assert all(labeled_target_var.data.ne(NO_LABEL))
        minibatch_size = target.size()[0]
        #labeled_minibatch_size = labeled_target.size()[0]

        ### because set drop_last = True, so last minibatch will be dropped
        #assert minibatch_size == args.batch_size, print(i, minibatch_size, labeled_minibatch_size)
        #assert minibatch_size == args.batch_size, print(i, minibatch_size)
        #assert labeled_minibatch_size == args.labeled_batch_size == (target.sum(dim=1) > 0).sum()
        #meters.update('labeled_minibatch_size', labeled_minibatch_size)

        ### model inference to output logits prediction
        #LOG.info("here")
        model_out = model(input_var)
        #LOG.info('model {}'.format(model_out.size()))
        #LOG.info('labeled_target_var {}'.format(labeled_target_var.size()))

        ############# classification loss of labeled samples
        ### create a new variable, but the gradient flow is wrong!!!
        ### class_logit = Variable(model_out.data[-args.labeled_batch_size:, ...], requires_grad=True)
        #mask = Variable(torch.LongTensor(list(range(minibatch_size-labeled_minibatch_size, minibatch_size)))).cuda()
        #class_logit = model_out.index_select(0, mask)
        #assert class_logit.data.size()[0] == labeled_minibatch_size
        #assert class_logit.requires_grad == True
        #assert mask.requires_grad == False


        ############# compute losses
        ### for classification loss of labeled samples
        ### if class_criterion is reduce=False, then to get the average loss, should devide (labeled_batch_size * num_classes)
        ### this is different from cross_entropy_loss for single-label, because for multi-label, each samples have multiple output numbers
        #loss = soft_dice(class_logit, labeled_target_var)
        #loss = soft_dice(model_out, labeled_target_var)
        
        loss = dice_loss(model_out, labeled_target_var)
        #LOG.info('loss {}'.format(loss))


        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        ### compute gradient and do SGD step
        optimizer.zero_grad()
        #LOG.info('finished optimizer.zero_grad()')
        loss.backward()
        #LOG.info('finished loss.backward()')
        optimizer.step()
        #LOG.info('finished optimizer.step()')

        ### update EMA model with current primary model, not training EMA model
        global_step += 1
        #LOG.info('global_step {}'.format(global_step))

        ### measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()


        ############# print progress to console
        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'lr {meters[lr]:.6f}\t'
                'Loss {meters[loss]:.4f}\t'
                #'Cons {meters[cons_loss]:.4f}\t'
                # 'Res {meters[res_loss]:.4f}'
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
def validate(eval_loader, model, log, global_step, epoch, model_flag, save_pred=False):
#def validate(eval_loader, global_step, epoch, log, model_flag, save_pred=False, **kwargs):

    class_criterion = nn.MultiLabelSoftMarginLoss().cuda() #Multi-LAbel Classification
    # class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda() #Single-Label

    meters = AverageMeterSet()
    model.eval() #switch to evaluate mode

    ### record all prediction and target for computing AUC score
    target_total = torch.randn(0, args.num_classes).double().cuda()
    pred_total = torch.randn(0, args.num_classes).double().cuda()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        ############# ADDED
        #target to onehot for model: nn.MultiLabelSoftMarginLoss
        #one_hot = torch.FloatTensor(target.size(0), args.num_classes).zero_()
        #target = one_hot.scatter_(1, target.data, 1)
        ############# END ADDED
        target_var = torch.autograd.Variable(target).long()

        onehot_dim = (target_var.size(0), args.num_classes, target_var.size(2), target_var.size(3))
        one_hot = torch.zeros(onehot_dim)
        target_onehot = one_hot.scatter_(1, target_var.data, 1)
        #LOG.info('target_onehot size {}'.format(target_onehot.size()))

        input = Variable(input).cuda()
        target = Variable(target_onehot).cuda()



        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        #with torch.no_grad():
        #    input_var = input
        #    target_var = target.cuda(async=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = (target_var.data.sum(dim=1) > 0).sum()
        #assert labeled_minibatch_size > 0
        ### because set drop_last = False for all evaluation dataset, so the last minibatch might be imcomplete
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        ### compute output and loss
        model_output = model(input_var)
        loss = dice_loss(model_output, target_var)
        meters.update('loss', loss.item(), labeled_minibatch_size)

        # ### record all prediction and target for computing AUC score
        # #sigmoid_output = torch.sigmoid(model_output)
        # sm = nn.Softmax() #softmax for multi-class
        # sigmoid_output = sm(model_output)
        # target_total = torch.cat((target_total, target_var.data.double()), 0)
        # pred_total = torch.cat((pred_total, sigmoid_output.data.double()), 0)

        ############# print progress to console
        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Class {meters[loss]:.4f}\t'
                .format(i, len(eval_loader), meters=meters))

    # ### computer AUC_ROC score
    # pred_array = pred_total.cpu().numpy()
    # #print('predictions', pred_array)
    # pred_array = pred_array.argmax(axis=1)
    # print('predictions\t', pred_array)
    # labels = np.array(range(0,args.num_classes))
    # pred_array = LabelBinarizer().fit(labels).transform(pred_array)
    # #print('predictions transformed', pred_array)
    # target_array = target_total.cpu().numpy()

    # #print('predictions', pred_array)
    # print('answers\t\t', target_array.argmax(axis=1))

    # auc_list = roc_auc_score(y_true=target_array, y_score=pred_array, average=None)
    # auc_list = accuracy_score(y_true=target_array, y_pred=pred_array) #calculated accuracy for multi-class
    # f1_weighted = f1_score(y_true=target_array, y_pred=pred_array, average='weighted') #calculated accuracy for multi-class with class imbalance
    # f1_list = f1_score(y_true=target_array, y_pred=pred_array, average=None) #calculated accuracy for multi-class with class imbalance
    # print(f1_list)
    # print('f1 weighted: ', f1_weighted)
    # print('accuracy', auc_list)
    # for score in auc_list: print("%.4f" %(score))

    #LOG.info(
    #    'acc_list: {}\n'
    #    'f1_list: {}\n'
    #    'f1 weighted: {}'
    #    .format(auc_list, f1_list, f1_weighted))

    ## ignore the "no_finding" class
    # auc_list[args.class_to_idx['No_Finding']] = 0.0

    # average_auc = np.sum(auc_list) / np.double(args.num_classes - 1)
    # print('{}: Average AUC score for {} classes: {} %'.format(model_flag, args.num_classes-1, 100*average_auc))
    # meters.update('auc', 100*average_auc)

    # average_auc = np.sum(auc_list) / np.double(args.num_classes)
    # average_auc = auc_list
    # #print('{} [epoch {}/{}]: Avg AUC for {} classes: {} %\n'.format(model_flag, epoch, args.epochs, args.num_classes, 100*average_auc))
    # print('{} [{}/{}]: Accuracy for {} classes: {} %\n'.format(model_flag, epoch, args.epochs, args.num_classes, 100*average_auc))
    # meters.update('auc', 100*average_auc)

    print('[{}/{}]: Dice score: {} %\n'.format(epoch, args.epochs, args.num_classes, loss))
    # meters.update('auc', 100*average_auc)



    #meters.update('auc', average_auc)

    # LOG.info(' * AUC {auc.avg:.4f}'.format(auc=meters['auc']))
    # LOG.info(' * Acc {auc.avg:.4f}'.format(auc=meters['auc']))
    log.record(epoch, {'step': global_step})
#    log.record(epoch, {
#        'step': global_step,
#        **meters.values(),
#        **meters.averages(),
#        **meters.sums()
#    })

    # if save_pred:
    #     #print(auc)
    #     return average_auc, auc_list, target_array, pred_array
    # else:
    #     return average_auc

    return loss




#########################################################################
############# Loading and saving ########################################
#########################################################################
def save_checkpoint(state, is_best, dirpath, epoch):
    # filename = 'checkpoint.{}.ckpt'.format(epoch)
    filename = '{}/{}_{}.ckpt'                 ### only save the last checkpoint and the best one (best EMA-prec1)
    checkpoint_path = filename.format(dirpath, 'checkpoint', exp)
    best_path = filename.format(dirpath, 'best', exp)

    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


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
