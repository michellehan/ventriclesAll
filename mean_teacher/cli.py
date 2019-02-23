import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--parameters', default=None)

    parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--raw-dir', metavar='DIR', default='',
                    help='path to train/validation set')
    parser.add_argument('--segs-dir', metavar='DIR', default='',
                    help='path to segmentations')
    parser.add_argument('--train-csv', metavar='DIR', default='',
                    help='path to train labels')
    parser.add_argument('--val-csv', metavar='DIR', default='',
                    help='path to validation labels')
    parser.add_argument('--test-csv', metavar='DIR', default='',
                    help='path to test labels')
    parser.add_argument('--mask-dir', metavar='DIR', default='',
                    help='path to save masks')
    parser.add_argument('--ckpt-dir', metavar='DIR', default='',
                    help='path to load ckpts from')
    # parser.add_argument('--datadir', metavar='DIR', default='../data/cxr14/',
    #                 help='path to dataset')
    # parser.add_argument('--csvdir', metavar='DIR', default='../data_csv',
    #                     help='path to dataset')
    # parser.add_argument('--train-subdir', type=str, default='',
    #                     help='the subdirectory inside the data directory that contains the training data')
    # parser.add_argument('--eval-subdir', type=str, default='',
    #                     help='the subdirectory inside the data directory that contains the evaluation data')
    # parser.add_argument('--labels', default=None, type=str, metavar='FILE',
    #                     help='list of image labels (default: based on directory structure)')
    # parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
    #                     help='exclude unlabeled examples from the training set')
    parser.add_argument('--encoder', default="resnet",
                        help='pretrained model for architecture')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    #########################################################################
    ### training
    parser.add_argument('--seed', default=1, type=int,
                        metavar='N', help='manual seed for GPUs to generate random numbers')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--kfold', default=10, type=int, metavar='N',
                        help="number of folds to make")    
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--lr-decay', default=15, type=int,
                    help='steps for decaying lr to 1/10 (default: 15)')
    # parser.add_argument('--initial-lr', default=0.0, type=float,
    #                     metavar='LR', help='initial learning rate when using linear rampup')
    # parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
    #                     help='length of learning rate rampup in the beginning')
    # parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        # help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pretrained', default=False, type=str2bool,
                        help='use pre-trained model', metavar='BOOL')
    parser.add_argument('--log', default=False, type=str2bool,
                        help='logging into txt file', metavar='BOOL')
    #########################################################################
    ### evaluation
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint for continue training (default: none)')
    parser.add_argument('-e', '--evaluate', default=False, type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--only-mask', default=False, type=str2bool,
                        help='just save masks')
    parser.add_argument('--ckpt', default=' ', type=str,
                    help='use best or final checkpoint (default: none)')
    parser.add_argument('--eval-tag', type=str, default=None,
                    help='suffix for npz result file for eval_mode')
    
    # parser.add_argument('--attention', default=None, type=float, metavar='WEIGHT',
    #                     help='use attention loss with given weight (default: None)')
    # parser.add_argument('--attention-rampup', default=30, type=int, metavar='EPOCHS',
                        # help='length of the attention loss ramp-up')
    # parser.add_argument('--ensemble', default=False, type=str2bool,
    #                     help='use model ensembling for training', metavar='BOOL')
    # parser.add_argument('--exp', default=None, type=str, metavar='FILE',
    #                     help='exp sub-class')

    #########################################################################
    ### experimental setting parameters
    parser.add_argument('--flag', type=str, default='',
                    help='"full" or "balanced" or "unbalanced" training')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--train-portion', type=float, default=0.0,
                    help='portion of training samples for semi-supervised training')
    parser.add_argument('--num-classes', type=int, default=0,
                    help='number of classes for semi-supervised training')
    parser.add_argument('--class_to_idx', type=dict, default=None,
                    help='class name to idx')
    parser.add_argument('--final-model', default=" ", type=str, metavar='TYPE',
                        choices=['primary', 'ema'],
                        help='final model to use and save best.ckpt')

    ### loss function definition parameters
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-type', default=" ", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--logit-distance-cost', default=0.0, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    
    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
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

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
