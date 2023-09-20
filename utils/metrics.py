import numpy as np
import torch 


def precision(y_true, y_pred):
    """
    y_true:(b,c,h,w) label
    y_pred:(b,c,h,w) prediction
    按照每个样本计算
    """
    intersection = (y_true * y_pred).sum((1,2,3))
    return ((intersection + 1e-15) / (y_pred.sum((1,2,3)) + 1e-15)).sum()
    
def recall(y_true, y_pred):
    """
    y_true:(b,c,h,w) label
    y_pred:(b,c,h,w) prediction
    按照每个样本计算
    """
    intersection = (y_true * y_pred).sum((1,2,3))
    return ((intersection + 1e-15) / (y_true.sum((1,2,3)) + 1e-15)).sum()
    
def F2(y_true, y_pred, beta=2):
    
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15) 


def PPV(y_true,y_pred):
    # TP/(TP + FP)
    TP = (y_true * y_pred).sum()
    FP = np.sum(y_true[y_pred>0]==0)
    
    return TP / float(TP+FP+1e-15)

def dice_score(y_true, y_pred):
    """
    y_true:(b,c,h,w) label
    y_pred:(b,c,h,w) prediction
    按照每个样本计算
    """
    return ((2 * (y_true * y_pred).sum((1,2,3)) + 1e-15) / (y_true.sum((1,2,3)) + y_pred.sum((1,2,3)) + 1e-15)).sum()

def iou_score(y_true, y_pred):
    """
    y_true:(b,c,h,w) label
    y_pred:(b,c,h,w) prediction
    按照每个样本计算
    """
    return (((y_true * y_pred).sum((1,2,3)) + 1e-15) / (y_true.sum((1,2,3)) + y_pred.sum((1,2,3)) -(y_true * y_pred).sum((1,2,3))+ 1e-15)).sum()


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def evaluate_network(args,network,dataloader):
    network = network.cuda()
    network.eval()
    with torch.no_grad():
        prec = 0.
        rec = 0.
        dice = 0.
        iou = 0.
        length = 0
        for images, labels,boundary,img_name in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            pred = network(images) #(4,1,384,384)
            mask = pred['mask']
            mask = mask.argmax(1).unsqueeze(1)
            # mask = (torch.sigmoid(mask)>0.5).to(torch.float)
            # mask = mask.argmax(1).unsqueeze(1)
            prec += precision(labels,mask)
            rec += recall(labels, mask)
            dice += dice_score(labels, mask)
            iou += iou_score(labels, mask)
            length += len(labels)
    return dice/length,prec/length,rec/length,iou/length

def evaluate_network_all_data_heterogeneous(args,networks,dataloaders):
    with torch.no_grad():
        prec = 0.
        rec = 0.
        dice = 0.
        iou = 0.
        length = 0
        for i in range(len(dataloaders)):
            dataloader = dataloaders[i]
            network = networks[i]
            network = network.cuda()
            network.eval()
            for images, labels,boundary,img_name in dataloader:
                images = images.cuda()
                labels = labels.cuda()
                pred = network(images) #(4,1,384,384)
                mask = pred['mask']
                mask = mask.argmax(1).unsqueeze(1)
                # mask = (torch.sigmoid(mask)>0.5).to(torch.float)
                # mask = mask.argmax(1).unsqueeze(1)
                prec += precision(labels,mask)
                rec += recall(labels, mask)
                dice += dice_score(labels, mask)
                iou += iou_score(labels, mask)
                length += len(labels)
    return dice/length,prec/length,rec/length,iou/length

def evaluate_network_all_data(args,network,dataloaders):
    network = network.cuda()
    network.eval()
    with torch.no_grad():
        prec = 0.
        rec = 0.
        dice = 0.
        iou = 0.
        length = 0
        for dataloader in dataloaders:
            for images, labels,_,_ in dataloader:
                images = images.cuda()
                labels = labels.cuda()
                pred = network(images) #(4,1,384,384)
                mask = pred['mask']
                
                mask = mask.argmax(1).unsqueeze(1)
                prec += precision(labels,mask)
                rec += recall(labels, mask)
                dice += dice_score(labels, mask)
                iou += iou_score(labels, mask)
                length += len(labels)
    return dice/length,prec/length,rec/length,iou/length

def init_eval_list(args):
    eval_list = []
    for i in range(len(args.clients_model)+1):
        if i == len(args.clients_model):
            eval_dict = {}
            eval_dict['client_id'] = 'global'
            eval_dict['net_name'] = args.global_name
            eval_dict['dice'] = 0.0
            eval_dict['iou'] = 0.0
            eval_dict['precision'] = 0.0
            eval_dict['recall'] = 0.0
            eval_dict['best_epoch'] = 0
            eval_list.append(eval_dict)
        else:
            eval_dict = {}
            eval_dict['client_id'] = i
            eval_dict['net_name'] = args.clients_model[i]
            eval_dict['dice'] = 0.0
            eval_dict['iou'] = 0.0
            eval_dict['precision'] = 0.0
            eval_dict['recall'] = 0.0
            eval_dict['best_epoch'] = 0
            eval_list.append(eval_dict)
    
    return eval_list

def update_eval_list(participant_index,dice,prec,rec,iou,eval_list,epoch_index):
    eval_dict = eval_list[participant_index]
    if dice>eval_dict['dice']:
        eval_dict['dice'] = dice.item()
        eval_dict['iou'] = iou.item()
        eval_dict['best_epoch'] = epoch_index
    # if prec>eval_dict['precision']:
        eval_dict['precision'] = prec.item()
    # if rec>eval_dict['recall']:
        eval_dict['recall'] = rec.item()
    eval_list[participant_index] = eval_dict
    return eval_list

def update_test_list(participant_index,dice,prec,rec,iou,eval_list,epoch_index):
    eval_dict = eval_list[participant_index]
    eval_dict['dice'] = dice.item()
    eval_dict['iou'] = iou.item()
    eval_dict['best_epoch'] = epoch_index
    eval_dict['precision'] = prec.item()
    eval_dict['recall'] = rec.item()
    eval_list[participant_index] = eval_dict
    return eval_list


def get_matrics(args,best_dice_list,dataloaders):
    best_dice = 0.
    best_iou = 0.
    best_precision = 0.
    best_reacll = 0.
    length = 0.
    for participant_index in range(args.num_clients):
        best_dice+=best_dice_list[participant_index]['dice']*len(list(dataloaders[participant_index]))
        best_iou+=best_dice_list[participant_index]['iou']*len(list(dataloaders[participant_index]))
        best_precision+=best_dice_list[participant_index]['precision']*len(list(dataloaders[participant_index]))
        best_reacll+=best_dice_list[participant_index]['recall']*len(list(dataloaders[participant_index]))
        length+=len(list(dataloaders[participant_index]))
    return {"dice":best_dice/length,"iou":best_iou/length,"precision":best_precision/length,"reacll":best_reacll/length}
    