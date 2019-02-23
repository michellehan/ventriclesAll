import os, torch
import numpy as np
import pandas as pd
import nibabel as nib

test_pred_path = '/home/mihan/projects/ventriclesAll/test_pred/UNet_vgg11_cls2/lr0.05_d5_m0.8_b16'
mask_path = '/home/mihan/projects/ventriclesAll/masks/UNet_vgg11_cls2/lr0.05_d5_m0.8_b16'
nifti_dir = '/data/NormalVentricle/normalHydro/niftis/t2' ##############
res_exp = 'lr0.05_d5_m0.8_b16_01'

result = os.path.join(test_pred_path, 'test_{}.npz'.format(res_exp))
print('...loading target_var')
target_var = torch.tensor( np.load(result)['target'] )
print('...loading pred_var')
pred_var = torch.tensor( np.load(result)['pred'] )

test_csv = '/data/NormalVentricle/normalHydro/labels/t2_train_01.csv'
outnames = pd.read_csv(test_csv, header=None)

subjs = [] 
niftis = {} 
total = pred_var.size(0)
for sample in range(total):
    outname = outnames.iloc[sample,0].split(".")[0]
    outpath = mask_path + "/" + outname + ".png"
    outpath_pred = mask_path + "/pred_" + outname + ".png"
    pred = pred_var[sample].float().cpu()
    target = target_var[sample].float().cpu()
    TP = torch.mul(pred, target)
    FP = pred - TP
    FN = target - TP
    subjname = outname.split("-")[0]
    if subjname not in subjs:
        subjs.append(subjname)
        print(subjname)
        raw_ni = nifti_dir + "/" + subjname + ".nii.gz"
        hd = nib.load(raw_ni).header
        pix = hd['pixdim']
        dim = hd['dim']
        hd['pixdim'] = [pix[0], pix[1] * (dim[1]/256), pix[2] * (dim[2]/256), pix[3], pix[4], pix[5], pix[6], pix[7]]
        niftis[subjname] = [ np.flip( pred.data.numpy().T, 1), hd]
    else: 
        niftis[subjname][0] = np.concatenate( (niftis[subjname][0], np.flip( pred.data.numpy().T, 1)), axis=2)

nifti_out = os.path.join(mask_path, 'niftis')
if not os.path.isdir(nifti_out): os.makedirs(nifti_out)
for subj in subjs:
    niftiname = nifti_out + '/' + subj + '_mask.nii.gz'
    refname = nifti_dir + '/' + subj + '.nii.gz'
    ni = niftis[subj]
    img = nib.nifti1.Nifti1Image(ni[0], None, header=ni[1])
    nib.save(img, niftiname)
    cmd = "flirt -in %s -ref %s -applyxfm -out %s" %(niftiname, refname, niftiname)
    print(cmd)
    os.system(cmd)
    cmd = 'fslmaths %s -bin %s' %(niftiname, niftiname)
    print(cmd)
    os.system(cmd)
