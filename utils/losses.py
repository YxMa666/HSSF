import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
random.seed(42)

def init_loss(args):
    if args.boundary_loss == 'CE':
        boundary_criterion = nn.CrossEntropyLoss()
    if args.boundary_loss == 'MSE':
        boundary_criterion = nn.MSELoss()
    if args.boundary_loss == 'SCE':
        boundary_criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=10)
    if args.boundary_loss == 'FocalLoss':
        boundary_criterion = FocalLoss()
    if args.boundary_loss == 'softmax_mse_loss':
        boundary_criterion = softmax_mse_loss()
    if args.boundary_loss == 'mse_loss':
        boundary_criterion = mse_loss()
    if args.boundary_loss == 'cosine_similarity':
        boundary_criterion = cosine_similarity(softmax_t = args.softmax_t)
    if args.seg_loss == 'structure_loss':
        seg_criterion = structure_loss()
    if args.seg_loss == 'softmax_dice_loss':
        seg_criterion = softmax_dice_loss()
    if args.seg_loss == 'dice_loss':
        seg_criterion = dice_loss()
    if args.kl_loss == 'kl_loss':
        kl_criterion = kl_loss(softmax_t = args.softmax_t)
    if args.kl_loss == 'softmax_mse_loss':
        kl_criterion = softmax_mse_loss()
    if args.kl_loss == 'mse_loss':
        kl_criterion = mse_loss()
    if args.kl_loss == 'kl_mse_loss':
        kl_criterion = kl_mse_loss(softmax_t = args.softmax_t)
    if args.kl_loss == 'cosine_similarity':
        kl_criterion = cosine_similarity(softmax_t = args.softmax_t)
    return boundary_criterion,seg_criterion,kl_criterion


def CEloss(inputs, gt, ignore_index=255):
    if len(gt.size())==4:
        inputs = F.interpolate(
            inputs, size=gt.size()[2:], mode="bilinear", align_corners=True
        )
        return nn.CrossEntropyLoss(ignore_index=ignore_index)((inputs).to(torch.float), gt.argmax(1))
    else:
        return nn.CrossEntropyLoss(ignore_index=ignore_index)((inputs).to(torch.float), gt)

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not (_is_pil_image(img) or isinstance(img, torch.Tensor)):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if _is_pil_image(img):
        return img.crop((j, i, j + w, i + h))

    return img[..., i : (i + h), j : (j + w)]
class RandomCrop4(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        if len(img.shape) == 4:  # b x c x h x w
            h, w = img.shape[2:]
        elif len(img.shape) == 3:  # c x h x w
            h, w = img.shape[1:]
        else:
            raise NotImplementedError
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, i1, i2, i3, i4):
        # Pad if needed
        i, j, h, w = self.get_params(i1, self.size)

        i1 = crop(i1, i, j, h, w)
        i2 = crop(i2, i, j, h, w)
        i3 = crop(i3, i, j, h, w)
        i4 = crop(i4, i, j, h, w)

        return i1, i2, i3, i4

crop_f = RandomCrop4(size=(32, 32))

def batch_pixelwise_distanceloss(inputs, target, final_candidate, final_indices):
    # input : b x c x h x w
    # target : b x c x h x w
    # final_indicies : b x h x w

    assert (
        inputs.size()[2:] == final_candidate.size()[1:]
    ), "input / final_candid : %s, %s" % (inputs.shape, final_candidate.shape)

    final_indices = torch.where(final_candidate == 1, final_indices, 255)

    b_loss = 0
    label = torch.unique(final_indices)
    label = label[label != 255]

    # mask : b x h x w
    inputs = inputs.permute(1, 0, 2, 3)  # c x b x h x w
    target = target.permute(1, 0, 2, 3)  # c x b x h x w

    for idx in label:
        # if idx == 0:
        #     continue
        input_vec = inputs[:, (idx == final_indices)].T  # n x 128
        input_vec = input_vec / input_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)

        pos_vec = target[:, (idx == final_indices)]  # 128 x n
        pos_vec = pos_vec / pos_vec.norm(dim=0, keepdim=True).clamp(min=1e-8)

        neg_v = target[:, (idx != final_indices) & (final_indices != 255)]  # 128 x m
        neg_v = neg_v / neg_v.norm(dim=0, keepdim=True).clamp(min=1e-8)

        pos_pair = torch.mm(input_vec, pos_vec)
        neg_pair = torch.mm(input_vec, neg_v)
        pos_pair = torch.exp(pos_pair / 0.5).to(torch.float64).sum().clamp(min=1e-8)
        neg_pair = torch.exp(neg_pair / 0.5).to(torch.float64).sum().clamp(min=1e-8)

        b_loss += -(torch.log(pos_pair / (neg_pair + pos_pair))) / torch.count_nonzero(
            (idx == final_indices).long()
        )

    if len(label) == 0:
        return 0
    else:
        return b_loss / len(label)

def pixelwisecontrastiveloss(inputs, target, final_candidate=None, final_indicies=None):

    assert final_candidate is not None

    tot_loss = 0
    crop_cnt = 8

    for _ in range(crop_cnt):
        cr_inputs, cr_target, cr_final_candid, cr_final_indicies = crop_f(
            inputs, target, final_candidate, final_indicies
        )

        tot_loss += batch_pixelwise_distanceloss(
            cr_inputs, cr_target, cr_final_candid, cr_final_indicies
        )
    return tot_loss / crop_cnt
class confidence_loss(nn.Module):
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param input_logits: student score map
    :param target_logits: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    def __init__(self):
        super(confidence_loss, self).__init__()
    def forward(self,predict, confidence,mask = None):
        input = predict.argmax(1)
        mask_l = mask.argmax(1)
        ind = (input==mask_l).to(torch.float)
        correction_loss = nn.BCEWithLogitsLoss(reduction="none")(confidence, ind.unsqueeze(1))
        correction_loss.squeeze(1)[ind==0]*=((ind==1).sum()/(ind==0).sum())
        # correction_loss.squeeze(1)[ind==1]*=((ind==0).sum()/(ind==1).sum())
        # r_ind = 1-ind
        # conf = ((1-confidence)*ind + confidence*r_ind).mean()
        return correction_loss.mean()

class kl_loss(nn.Module):
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param input_logits: student score map
    :param target_logits: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    def __init__(self,softmax_t):
        super(kl_loss, self).__init__()
        self.softmax_t = softmax_t
    def forward(self,input_logits, target_logits,mask = None):
        T = self.softmax_t
        assert input_logits.size() == target_logits.size()
        input_softmax = F.log_softmax(input_logits/T, dim=1)
        target_softmax = F.softmax(target_logits/T, dim=1)
        if mask is None:
            klloss = F.kl_div(input_softmax, target_softmax,reduction='batchmean')
            
            return klloss
        else:
            klloss = (F.kl_div(input_softmax, target_softmax,reduction='none')).mean((1,2,3))*mask
            return klloss.mean()
        
class cosine_similarity(nn.Module):
    """
    :param input_logits: student score map
    :param target_logits: teacher score map
    :return: loss value
    """
    def __init__(self,softmax_t):
        super(cosine_similarity, self).__init__()
        self.softmax_t = softmax_t
    def forward(self,input_logits, target_logits):
        T = self.softmax_t
        assert input_logits.size() == target_logits.size()
        input_softmax = F.log_softmax(input_logits/T, dim=1)
        target_softmax = F.softmax(target_logits/T, dim=1)
        input_softmax = input_softmax.view(-1, 2)
        target_softmax = target_softmax.view(-1, 2)
        cosine_similarity_loss = ((1-F.cosine_similarity(input_softmax, target_softmax,dim=0)).mean())* (T ** 2)
        
        
        return cosine_similarity_loss
    

class kl_mse_loss(nn.Module):
    """
    kl and mse
    """
    def __init__(self,softmax_t):
        super(kl_mse_loss, self).__init__()
        self.softmax_t = softmax_t
        self.soft_mse_loss = softmax_mse_loss()
    def forward(self,input_logits, target_logits,theta =1):
        T = self.softmax_t
        assert input_logits.size() == target_logits.size()
        mseloss = self.soft_mse_loss(input_logits, target_logits)
        input_softmax = F.log_softmax(input_logits/T, dim=1)
        target_softmax = F.softmax(target_logits/T, dim=1)
        input_softmax = input_softmax.view(-1, 2)
        target_softmax = target_softmax.view(-1, 2)

        klloss = F.kl_div(input_softmax, target_softmax,reduction = "batchmean") * (T ** 2)
        klloss = klloss + theta*mseloss
        return klloss


class softmax_mse_loss(nn.Module):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    def __init__(self):
        super(softmax_mse_loss, self).__init__()
    def forward(self,input_logits, target_logits, use_weight = True):
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        if use_weight:
            softmseloss =0.0
            allsum = target_logits.shape[-1]*target_logits.shape[-2]
            for i in range(target_softmax.shape[1]):
                softmseloss += ((allsum/target_softmax[:,i].sum())*(input_softmax[:,i] - target_softmax[:,i])**2).mean()
            softmseloss = softmseloss/target_softmax.shape[1]
        else:
            softmseloss = ((input_softmax - target_softmax)**2).sum()
        return softmseloss
    
class mse_loss(nn.Module):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    def __init__(self):
        super(mse_loss, self).__init__()
    def forward(self,input_logits, target_logits, use_weight = True):
        assert input_logits.size() == target_logits.size()
        # input_softmax = F.softmax(input_logits, dim=1)
        # target_softmax = F.softmax(target_logits, dim=1)
        input_softmax = input_logits
        target_softmax = target_logits
        if use_weight:
            mseloss =0.0
            allsum = target_logits.shape[-1]*target_logits.shape[-2]
            for i in range(target_softmax.shape[1]):
                weight = F.softmax(target_softmax,1).sum(-1).sum(-1)[:,i]/allsum
                mseloss += (weight*((input_softmax[:,i] - target_softmax[:,i])**2).sum(-1).sum(-1)/allsum).mean()
            mseloss = mseloss/target_softmax.shape[1]
        else:
            mseloss = ((input_softmax - target_softmax)**2).sum()
        return mseloss
    
    
class dice_loss(nn.Module):
    """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
    def __init__(self):
        super(dice_loss, self).__init__()
    def forward(self,input_logits, target_logits):
        assert input_logits.size() == target_logits.size()
        input_softmax = input_logits
        target_softmax = target_logits
        n = input_logits.shape[1]
        dice = 0
        for i in range(0, n):
            dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
        mean_dice = dice / n

        return mean_dice

class softmax_dice_loss(nn.Module):
    """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
    def __init__(self):
        super(softmax_dice_loss, self).__init__()
    def forward(self,input_logits, target_logits):
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n = input_logits.shape[1]
        dice = 0
        for i in range(0, n):
            dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
        mean_dice = dice / n

        return mean_dice


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class structure_loss(nn.Module):
    
    def __init__(self):
        super(structure_loss, self).__init__()
    def forward(self,pred, mask,weight=None):
        '''
        pred:(b,c,h,w)
        mask:(b,1,h,w) type long
        weight:(b,1,h,w)
        '''
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # if weight is not None:
        #     pred = pred*weight
        # wbce = F.binary_cross_entropy_with_logits(pred, mask)
        
        wbce = nn.CrossEntropyLoss()(pred, mask)
        # wbce = F.binary_cross_entropy_with_logits(pred, mask)
        
        # wbce = F.cross_entropy(pred,(mask.squeeze(1).to(torch.long)))#F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        # wbce = (weit * wbce).sum(dim=(-3,-2,-1)) / weit.sum(dim=(-3,-2,-1))
        
        # wbce = (weit * wbce) / weit
        if weight is not None:
            wbce = wbce*weight
        # wbce = wbce.mean()#.sum(dim=(-2,-1))
        pred = torch.sigmoid(pred)
        
        if weight is not None:
            inter = (((pred * mask) * weit)).sum(dim=(-3,-2,-1))
            union = (((pred + mask) * weit)).sum(dim=(-3,-2,-1))
            wiou = (1 - (inter + 1) / (union - inter + 1))*weight
            # wiou = wiou.mean()
        else:
            inter = ((pred * mask) * weit).sum(dim=(-3,-2,-1))
            union = ((pred + mask) * weit).sum(dim=(-3,-2,-1))
            wiou = (1 - (inter + 1) / (union - inter + 1))

        return (wbce + wiou).mean()
        

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

eps = 1e-7

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

