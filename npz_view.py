from __future__ import division
import sys, os
import subprocess
import numpy as np
from glob import glob

def printwrite(outfile, text):
    outfile.write('%s\n' %text)
    print(text)

def dice_score(pred, target):
    smooth = 1.

    intersection = (pred * target).sum()
    A_sum = (pred * pred).sum()
    B_sum = (target * target).sum()

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    return dice



def combined_dice(bd, filenames, tag):
    outname = bd + '/review_%s.txt' %tag
    outfile = open(outname, 'w')
    print('Saving output to: ', outname)
    print("=> Calulating Dice score for: %s" %tag)
    printwrite(outfile, "=> Loading .npz files from: %s" %bd)

    targets = None
    preds = None
    names = None
    subjs = []
    volumes = {}

    files = [os.path.join(bd, x) for x in filenames] 
    for f in files:
        printwrite(outfile, os.path.basename(f))
        data = np.load(f) #data.files = target, pred, name, dice
        t = data['target']
        p = data['pred']
        n = data['name'][:,0]

        #add to pred/target for subject-wise dice scores
        for i, name in enumerate(n):
            pred = p[i].flatten()
            target = t[i].flatten()

            subjname = name.split('-')[0]
            if subjname not in subjs:
                subjs.append(subjname)
                volumes[subjname] = [pred, target]
            else:
                volumes[subjname] = [np.append(volumes[subjname][0], pred),
                                     np.append(volumes[subjname][1], target)]

        #to calculate overall dice score
        if targets is None:
            targets = t.flatten()
            preds = p.flatten()
            names = n
        else:
            targets = np.append(targets, t.flatten())
            preds = np.append(preds, p.flatten())
            names = np.append(names, n)
    
    dice = dice_score(preds, targets)
    printwrite(outfile, "\nOVERALL DICE: %s\n" %round(dice, 4))


    #ccalculate dice scores by subject
    printwrite(outfile, 'Dice scores by subject:')
    printwrite(outfile, 'Subject\tTarget (mm3)\tPredictd (mm3)\tPercent Diff\tDice')
    printwrite(outfile, '---------------------------------------------------------')
    for subj, vols in sorted(volumes.items()):
        pred = vols[0]
        target = vols[1]
        dice = dice_score(pred, target)

        pred_vol = pred.sum()
        target_vol = target.sum()
        
        #get original scan dimensions
        command = "fslinfo %s/%s.nii.gz" %(niftidir, subj)
        nifti_info = subprocess.check_output(command, shell=True)
        n = str(nifti_info).split('\\n')
        dim1 = float([x for x in n if x.startswith('dim1')][0].split(' ')[-1])
        dim2 = float([x for x in n if x.startswith('dim2')][0].split(' ')[-1])
        pixdim1 = float([x for x in n if x.startswith('pixdim1')][0].split(' ')[-1])
        pixdim2 = float([x for x in n if x.startswith('pixdim2')][0].split(' ')[-1])
        pixdim3 = float([x for x in n if x.startswith('pixdim3')][0].split(' ')[-1])

        #calculate mm3 volume
        pred_calc = (pred_vol / (256*256) ) * (dim1 * dim2 * pixdim1 * pixdim2 * pixdim3)
        target_calc = (target_vol / (256*256) ) * (dim1 * dim2 * pixdim1 * pixdim2 * pixdim3)
        diff = abs(pred_calc - target_calc) / target_calc * 100

        printwrite(outfile, '%s\t%s\t%s\t%s\t%s' %(subj,
					       round(target_calc, 1), 
					       round(pred_calc, 1),
                                               round(diff, 2),
                                               round(dice, 4)))


global niftidir
niftidir = "/data/NormalVentricle/normalHydro/niftis/t2"
bd = "/home/mihan/projects/ventriclesAll/test_pred/UNet_vgg11_cls2/lr0.05_d5_m0.8_b16"

#train: 01-02=normal; 03-05=hydro
#val: 06=hydro; 07=normal
#test: 08=hydro; 09=normal
normals = ["m_test_lr0.05_d5_m0.8_b16_01.npz",
           "m_test_lr0.05_d5_m0.8_b16_02.npz",
           "m_test_lr0.05_d5_m0.8_b16_07.npz",
           "m_test_lr0.05_d5_m0.8_b16_09.npz"]
hydros = ["m_test_lr0.05_d5_m0.8_b16_03.npz",
          "m_test_lr0.05_d5_m0.8_b16_04.npz",
          "m_test_lr0.05_d5_m0.8_b16_05.npz",
          "m_test_lr0.05_d5_m0.8_b16_06.npz",
          "m_test_lr0.05_d5_m0.8_b16_08.npz"]
test_normal = ["m_test_lr0.05_d5_m0.8_b16_09.npz"]
test_hydro = ["m_test_lr0.05_d5_m0.8_b16_08.npz"]


#combined_dice(bd, normals, "normals")
#combined_dice(bd, hydros, "hydros")
#combined_dice(bd, sorted(normals + hydros), "all")
combined_dice(bd, test_normal, "test_normal")
combined_dice(bd, test_hydro, "test_hydro")
combined_dice(bd, sorted(test_normal + test_hydro), "test_all")
