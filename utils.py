import numpy as np
import cv2
import random
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score,jaccard_score


# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       


## Temporary
class ISICLoader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(ISICLoader, self).__init__()
        self.train = train
        if train:
          self.data   = np.load(path_Data+'data_train.npy')
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
          
          
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        self.mask   = self.mask /255.
         
    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        if self.train:
            img, seg = self.apply_augmentation(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)

        return {'image': img,
                'mask' : seg}
               
    def apply_augmentation(self, img, seg):
        if random.random() < 0.5:
            img  = np.flip(img,  axis=1)
            seg  = np.flip(seg,  axis=1)
        return img, seg

    def __len__(self):
        return len(self.data)


class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]

        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)

        loss = dice.mean()
        return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss
    
class DiceLossV1(torch.nn.Module):
    def __init__(self, n_classes=1):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
class DiceLossV2(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))