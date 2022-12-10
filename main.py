# borrow from https://github.com/EndyWon/Texture-Reformer/blob/main/transfer.py
import os

import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import cv2

import argparse

from texture_reformer_adain_noise import Reformer


parser = argparse.ArgumentParser(description='Texture Reformer Pytorch')

# Specify inputs and outputs
parser.add_argument('-imgf', type=str, default='inputs/image.npy', help="File path to the image")
parser.add_argument('-lblf', type=str, default='inputs/label.npy', help="File path to the label")
parser.add_argument('-outf', type=str, default='outputs', help="Folder to save output images")

# Runtime controls
parser.add_argument('-coarse_alpha', type=float, default=1, help="Hyperparameter to blend transformed feature with content feature in coarse level (level 5)")
parser.add_argument('-fine_alpha', type=float, default=1, help="Hyperparameter to blend transformed feature with content feature in fine level (level 4)")

parser.add_argument('-concat_weight', type=float, default=50, help="Hyperparameter to control the semantic guidance/awareness weight for '-semantic concat' mode and '-semantic concat_ds' mode, range 0-inf")

parser.add_argument('-coarse_psize', type=int, default=0, help="Patch size in coarse level (level 5), 0 means using global view")
parser.add_argument('-fine_psize', type=int, default=3, help="Patch size in fine level (level 4)")

parser.add_argument('-enhance_alpha', type=float, default=1, help="Hyperparameter to control the enhancement degree in level 3, level 2, and level 1")

parser.add_argument('-noise_mu', type=float, default=0.3, help="Hyperparameter to control the noise rate of mean in AdaIN")
parser.add_argument('-noise_sigma', type=float, default=0.3, help="Hyperparameter to control the noise rate of std in AdaIN")

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

args.e5 = './weights/E5.pth'
args.e4 = './weights/E4.pth'
args.e3 = './weights/E3.pth'
args.e2 = './weights/E2.pth'
args.e1 = './weights/E1.pth'

args.d5 = './weights/D5.pth'
args.d4 = './weights/D4.pth'
args.d3 = './weights/D3.pth'
args.d2 = './weights/D2.pth'
args.d1 = './weights/D1.pth'

# View-Specific Texture Reformation (VSTR) operation
@torch.no_grad()
def VSTR(TR, encoder, decoder, content, style, content_sem, style_sem, patch_size, alpha, concat_weight):
    # make the width and height of the temporary content/target image the same as the content/target semantic map
    if content_sem.shape[2] != content.shape[2] or content_sem.shape[3] != content.shape[3]:
        content = content.squeeze(0).cpu().clone()
        content = transforms.ToPILImage()(content)
        content = transforms.Resize([content_sem.shape[2],content_sem.shape[3]])(content)
        if args.cuda:
            content = transforms.ToTensor()(content).unsqueeze(0).cuda()
        else:
            content = transforms.ToTensor()(content).unsqueeze(0)

    sF  = encoder(style)
    cF  = encoder(content)

    # concatenate the features of the semantic maps
    sF_sem  = encoder(style_sem)
    cF_sem  = encoder(content_sem)
    csF = TR.VSTR_concat(cF, sF, cF_sem, sF_sem, patch_size, alpha, concat_weight)
    
    #csF = cF
    Img = decoder(csF)

    return Img

# Statistic-based Enhancement (SE) operation
@torch.no_grad()
def SE(TR, encoder, decoder, content, style, args):
    sF = encoder(style)
    cF = encoder(content)

    # match the first-order statistics, i.e., channel-wise mean and standard deviation
    csF = TR.dadain(cF, sF, args.enhance_alpha, args.noise_mu, args.noise_sigma)

    #csF = cF
    Img = decoder(csF)
    return Img


def gray_nparr_to_rgb_tensor(arr, is_label=False):
    if is_label:
        arr = (arr/14*255).astype('uint8')
    else:
        arr = arr*255
    arr = Image.fromarray(arr).convert('RGB')
    arr = np.array(arr)
    arr = transforms.ToTensor()(arr).unsqueeze(0).cuda()
    return arr

def get_TR_ith_level_output(idx, TR, content, style, content_sem, style_sem, args):
    if idx==5:
        return VSTR(TR, TR.e5, TR.d5, content, style, content_sem, style_sem, args.coarse_psize, args.coarse_alpha, args.concat_weight)
    elif idx==4:
        return VSTR(TR, TR.e4, TR.d4, content, style, content_sem, style_sem, args.fine_psize, args.fine_alpha, args.concat_weight)
    elif idx==3:
        return SE(TR, TR.e3, TR.d3, content, style, args)
    elif idx==2:
        return SE(TR, TR.e2, TR.d2, content, style, args)
    elif idx==1:
        return SE(TR, TR.e1, TR.d1, content, style, args)
    else:
        raise Exception(f'{idx} is invalid idx value, 1 <= i <= 5')

def get_distorted_image(image, label):
    TR = Reformer(args).cuda()
    """
    prepare style and content
    """
    style = gray_nparr_to_rgb_tensor(image)

    lbl_sem = label.copy()
    lbl_sem[label==0] = 14
    lbl_sem[image==0] = 0
    style_sem = gray_nparr_to_rgb_tensor(lbl_sem, is_label=True)

    content = image.copy()
    content = gray_nparr_to_rgb_tensor(content)

    content_sem = gray_nparr_to_rgb_tensor(lbl_sem.copy(), is_label=True)

    """
    run texture reformer
    """
    for idx in [5, 4, 3, 2, 1]:
        content = get_TR_ith_level_output(idx, TR, content, style, content_sem, style_sem, args)

    """
    get dice score, euclidean distance
    """
    content = content.permute(0,2,3,1).squeeze().cpu().detach().numpy()
    content = np.uint8(content/(np.max(content)+1e-5)*255)
    content = np.array(Image.fromarray(content).convert('L')) # 3 channels -> 1 channel

    """
    cv2 convert scale absolute
    """
    cv2beta = -1 * int(content[0,0])
    content_exposure = cv2.convertScaleAbs(content, beta=cv2beta)


    return content_exposure

image = np.load(args.imgf)
label = np.load(args.lblf)
attacked_image = get_distorted_image(image, label)
out_path = os.path.join(args.outf, "output")
np.save(out_path, attacked_image)