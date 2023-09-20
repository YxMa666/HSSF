import os
import argparse
import time
import numpy as np
import cv2
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import random
from utils.utils import initial_trainer,exp_normalize,visual_img,update_weight #,update_ema_variables
from utils.datasets import get_datasplit
from utils.metrics import evaluate_network_all_data,init_eval_list,update_eval_list,update_test_list,evaluate_network,evaluate_network_all_data_heterogeneous,get_matrics
from utils.losses import init_loss,confidence_loss,mse_loss,CEloss
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
import copy
import datetime
        
def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='HSSF',help="selection from list:[HSSF,LSSL,local]")
    parser.add_argument('--data',type=str,default='polyp',help="selection from list:[polyp,isic]")
    parser.add_argument('--datasets',type=list,default=['CVC-ColonDB', 'CVC-ClinicDB', 'EndoTect-ETIS', 'CVC-300', 'Kvasir'],help="selection from list:[PH2,domain1,domain2,domain3,domain4]")
    parser.add_argument('--CommunicationEpoch', type=int, default=200)
    parser.add_argument('--localEpoch', type=int, default=1)
    parser.add_argument('--condensationEpoch', type=int, default=4)
    parser.add_argument('--fusionEpoch', type=int, default=4)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--global_name', type=str, default='resnet101',help='[resnet152,resnet34,resnet50,resnet101,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--global_optim', type=str, default='adamw',help='[adam,sgd,adamw]')
    parser.add_argument('--client_optim', type=str, default='adamw',help='[adam,sgd,adamw]')
    parser.add_argument('--boundary_loss', type=str, default='mse_loss',help='[MSE,CE,SCE,FocalLoss,softmax_mse_loss,mse_loss,cosine_similarity]')
    parser.add_argument('--seg_loss', type=str, default='structure_loss',help='[structure_loss,softmax_dice_loss,dice_loss,cosine_similarity]')
    parser.add_argument('--kl_loss', type=str, default='kl_loss',help='[kl_loss,softmax_mse_loss,mse_loss,kl_mse_loss]')
    parser.add_argument('--clients_model', type=list, default=['resnet18','resnet18','resnet34','resnet34'],help='[resnet34,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--labeled_ratio', type=list, default=[1,0.,0.,0.,0.],help='Proportion of labeled data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--pretrained_epoch', type=int, default=20)
    parser.add_argument('--need_fp', type=bool, default=True)
    parser.add_argument('--use_unlabel', type=bool, default=True)
    parser.add_argument('--collaborate', type=bool, default=True)
    parser.add_argument('--RPG', type=bool, default=True)
    parser.add_argument('--softmax_t', type=float, default=2)
    parser.add_argument('--DataParallel', type=bool, default=False)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--unlabel_threshold', type=float, default=0.3)
    parser.add_argument('--kl_theta', type=float, default=10)
    parser.add_argument('--unlabel_theta', type=float, default=1)
    parser.add_argument('--unlabel_thres', type=float, default=0.75)
    parser.add_argument('--confide_theta', type=float, default=4, help='confide_theta')
    parser.add_argument('--consis_theta', type=float, default=1)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
    parser.add_argument('--shape', type=tuple, default=384)
    parser.add_argument('--train_val_test', type=tuple, default=(0.8,0.1,0.1))
    parser.add_argument('--device',type=str,default='1',help="device id")
    parser.add_argument('--ver',type=str,default='v1',help="the trainning version")
    parser.add_argument('--log_path', type=str,
                        default='./log',
                        help='path to log')
    parser.add_argument('--img_path', type=str,
                        default='/data5/yxma/HSSF_private/data',
                        help='path to data')
    parser.add_argument('--split_path', type=str,
                        default='./data_split',
                        help='path to log')
    
    args = parser.parse_args()
    return args
        
def local_semi_surprised_learning(args, net_list, optimizer_clients, labeled_train_client_dataloaders, unlabeled_train_client_dataloaders, epoch, local_epoch_index):
    for participant_index in range(args.num_clients):
        netname = args.clients_model[participant_index]
        print('*'*20+netname+'*'*20) 
        logging.info('*'*20+netname+'*'*20)
        network = net_list[participant_index]
        optimizer = optimizer_clients[participant_index]
        summary_writer.add_scalar("local client-{}-{}-optimizer_lr".format(participant_index,netname), optimizer.state_dict()['param_groups'][0]['lr'], epoch*args.localEpoch+local_epoch_index)
        if args.DataParallel:
            network = torch.nn.DataParallel(network).cuda()
        else:
            network = network.cuda()
        network.train()
        if args.model_name in ['LSSL','local']:
            labeled_train_dl_local = labeled_train_client_dataloaders
        else: 
            labeled_train_dl_local = labeled_train_client_dataloaders[participant_index]
        unlabeled_train_dl_local = unlabeled_train_client_dataloaders[participant_index]
        boundary_criterion,seg_criterion,_ = init_loss(args)
        boundary_criterion.cuda()
        seg_criterion.cuda()
        confidence_criterion = confidence_loss().cuda()
        if labeled_train_dl_local:
            for batch_idx, (images, labels, _,_) in enumerate(labeled_train_dl_local):  #########
                images = images.cuda()
                labels = F.one_hot(labels.permute(0,2,3,1).squeeze(-1).to(torch.long),num_classes = args.num_classes).permute(0,3,1,2).cuda().to(torch.float)
                pred = network(images)
                mask = pred['mask']
                confide_loss = args.confide_theta*confidence_criterion(mask.clone().detach(),pred['confidence'],labels)
                confide = torch.sigmoid(pred['confidence'].clone())
                loss = confide_loss + seg_criterion(mask,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 20==0:
                    print('epoch[{}/{}],local epoch[{}/{}],step: {} The local {} model {} on the labeled images: loss: [{}] , confidence loss: [{}] '.format(epoch_index,args.CommunicationEpoch,local_epoch_index,args.localEpoch,batch_idx,participant_index,netname,loss.item(),confide_loss.item()))
                    logging.info('epoch[{}/{}],local epoch[{}/{}],step: {} The local {} model {} on the labeled images: loss: [{}], confidence loss: [{}] '.format(epoch_index,args.CommunicationEpoch,local_epoch_index,args.localEpoch,batch_idx,participant_index,netname,loss.item(),confide_loss.item()))   
        if args.use_unlabel:        
            if unlabeled_train_dl_local:
                for batch_idx, (w_image,s_image, _) in enumerate(unlabeled_train_dl_local):  #########
                    w_image = w_image.cuda()
                    noise = torch.clamp(torch.randn_like(s_image) * 0.1, -0.2, 0.2).cuda()
                    s_image = s_image.cuda()
                    s_image = (s_image+noise).cuda()
                    s_pred = network(s_image)
                    s_mask = s_pred['mask']
                    confidence_s = torch.sigmoid(s_pred['confidence']).squeeze(1).detach()
                    w_pred = network(w_image,args.need_fp)
                    if args.need_fp:
                        w_mask,w_mask_fp = w_pred['mask']
                        confidence_w,_ = torch.sigmoid(w_pred['confidence']).squeeze(1).detach().chunk(2)
                    else:
                        w_mask = w_pred['mask']
                        confidence_w = torch.sigmoid(w_pred['confidence']).squeeze(1).detach()
                    confide = torch.where(confidence_s<confidence_w,confidence_w,confidence_s)
                    pseudo_mask = torch.where(confidence_w<confidence_s,s_mask.clone().detach().argmax(1),w_mask.clone().detach().argmax(1))
                    if args.RPG:
                        pseudo_mask = torch.where(confide<args.unlabel_threshold,1-pseudo_mask,pseudo_mask)
                    pseudo_mask_oh = F.one_hot(pseudo_mask,num_classes=2).permute(0,3,1,2)
                    loss = 0.0 
                    if args.need_fp:
                        loss += boundary_criterion(w_mask,w_mask_fp)
                    if not args.RPG:
                        mask_mask_s_loss = ((F.cross_entropy(w_mask, pseudo_mask,reduction='none') ).mean((-2,-1))).mean()+seg_criterion(w_mask,(pseudo_mask_oh).to(torch.float))
                        mask_mask_w_loss = ((F.cross_entropy(s_mask, pseudo_mask,reduction='none') ).mean((-2,-1))).mean()+seg_criterion(s_mask,(pseudo_mask_oh).to(torch.float))
                    else:
                        mask_mask_s_loss = ((F.cross_entropy(w_mask, pseudo_mask,reduction='none') ).mean((-2,-1))*(confide.mean((-2,-1))>args.unlabel_thres)).mean()+seg_criterion(w_mask,(pseudo_mask_oh).to(torch.float),(confide.mean((-2,-1))>args.unlabel_thres))
                        mask_mask_w_loss = ((F.cross_entropy(s_mask, pseudo_mask,reduction='none') ).mean((-2,-1))*(confide.mean((-2,-1))>args.unlabel_thres)).mean()+seg_criterion(s_mask,(pseudo_mask_oh).to(torch.float),(confide.mean((-2,-1))>args.unlabel_thres))
                    loss = loss + mask_mask_s_loss + mask_mask_w_loss
                    loss = args.unlabel_theta * loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 20==0:
                        print('epoch[{}/{}],local epoch[{}/{}],step: {} The local {} model {} on the unlabeled images: loss: [{}] ,weak loss [{}] , strong loss [{}]'.format( \
                             epoch_index,args.CommunicationEpoch,local_epoch_index,args.localEpoch,batch_idx,participant_index,netname,loss.item(),mask_mask_w_loss.item(),mask_mask_s_loss.item()))
                        logging.info('epoch[{}/{}],local epoch[{}/{}],step: {} The local {} model {} on the unlabeled images: loss: [{}],weak loss [{}] , strong loss [{}]'.format( \
                             epoch_index,args.CommunicationEpoch,local_epoch_index,args.localEpoch,batch_idx,participant_index,netname,loss.item(),mask_mask_w_loss.item(),mask_mask_s_loss.item()))
        optimizer_clients[participant_index] = optimizer
        net_list[participant_index] = network
    return net_list,optimizer_clients

def RegularityCondensation(args,global_net,net_list,global_optimizer,epoch,kl_epoch_index):
    boundary_criterion,seg_criterion,kl_criterion = init_loss(args)
    boundary_criterion.cuda()
    kl_criterion.cuda()
    seg_criterion.cuda()
    confidence_criterion = confidence_loss().cuda()
    summary_writer.add_scalar("global_optimizer lr", global_optimizer.state_dict()['param_groups'][0]['lr'], epoch*args.condensationEpoch+kl_epoch_index)
    for batch_idx, (images, labels, _,_) in enumerate(global_train_dataloader):  #########
        if args.DataParallel:
            global_net = torch.nn.DataParallel(global_net).cuda()
        else:
            global_net = global_net.cuda()
        global_net.train()
        images = images.cuda()
        noise = torch.clamp(torch.randn_like(images) * 0.1, -0.2, 0.2).cuda()
        images =images+noise
        labels = F.one_hot(labels.permute(0,2,3,1).squeeze(-1).to(torch.long),num_classes = args.num_classes).permute(0,3,1,2).cuda().to(torch.float)
        global_pred = global_net(images)
        pre_global_mask = global_pred['mask']
        confide_loss = args.confide_theta*confidence_criterion(pre_global_mask.clone().detach(),global_pred['confidence'],labels)
        confide = torch.sigmoid(global_pred['confidence'])
        pseudo_global_mask = pre_global_mask.clone().argmax(1)
        pseudo_global_mask = torch.where(confide.squeeze(1)<args.unlabel_threshold,1-pseudo_global_mask,pseudo_global_mask)
        glo_acc = (labels.argmax(1)==pseudo_global_mask).to(torch.float).mean((1,2))
        loss = confide_loss
        loss = loss + seg_criterion(pre_global_mask,labels)
        kl_mask_pick = pre_global_mask.clone().detach().cuda()
        picked_met = glo_acc.clone().detach().cuda()
        for participant_index in range(args.num_clients):
            network = net_list[participant_index]
            if args.DataParallel:
                network = torch.nn.DataParallel(network).cuda()
            else:
                network = network.cuda()
            network.eval()
            with torch.no_grad():
                pred = network(images)
                pre_mask = pred['mask']
                confide = torch.sigmoid(pred['confidence'])
                pseudo_mask = pre_mask.clone().argmax(1)
                score = (labels.argmax(1)==pseudo_mask).to(torch.float).mean((1,2))
                weight_sc = (score>picked_met).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                kl_mask_pick = weight_sc*pre_mask+(~weight_sc)*kl_mask_pick
                picked_met = torch.where(score>picked_met,score,picked_met)
                picked_met = torch.where(picked_met>args.unlabel_thres,picked_met,0)
        klloss =  args.kl_theta*kl_criterion(pre_global_mask,kl_mask_pick,(picked_met>glo_acc))
        loss = loss + klloss
        summary_writer.add_scalar("global_model with client kl and mask loss", loss, epoch*args.condensationEpoch*args.batch_size+kl_epoch_index*args.batch_size+batch_idx)
        if batch_idx % 20==0:
            print('epoch[{}/{}],kl epoch[{}/{}],step: {} The global model {} learning with clien on the labeled images loss: [{}], KL loss: [{}]'.format(epoch,args.CommunicationEpoch,kl_epoch_index,args.condensationEpoch,batch_idx,args.global_name,loss,klloss))
            logging.info('epoch[{}/{}],kl epoch[{}/{}],step: {} The global model {} learning with client on the labeled images loss: [{}], KL loss: [{}]'.format(epoch,args.CommunicationEpoch,kl_epoch_index,args.condensationEpoch,batch_idx,args.global_name,loss,klloss))
        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
    return global_net,global_optimizer

def RegularityFusion(args,global_train_dataloader,global_net,net_list,optimizer_clients,epoch,kl_epoch_index):
    boundary_criterion,seg_criterion,kl_criterion = init_loss(args)
    boundary_criterion.cuda()
    confidence_criterion = confidence_loss().cuda()
    kl_criterion.cuda()
    seg_criterion.cuda()
    for batch_idx, (images, labels, _,_) in enumerate(global_train_dataloader):    
        images = images.cuda()
        noise = torch.clamp(torch.randn_like(images) * 0.1, -0.2, 0.2).cuda()
        images =images+noise
        labels = F.one_hot(labels.permute(0,2,3,1).squeeze(-1).to(torch.long),num_classes = args.num_classes).permute(0,3,1,2).cuda().to(torch.float)
        '''
        Calculate boundary Output
        '''
        if args.DataParallel:
            global_net = torch.nn.DataParallel(global_net).cuda()
        else:
            global_net = global_net.cuda()           
        global_net.eval()
        with torch.no_grad():
            global_pred = global_net(images)
            pre_global_mask = global_pred['mask']
            glo_confide = torch.sigmoid(global_pred['confidence'])
            pseudo_global_mask = pre_global_mask.clone().argmax(1)
            pseudo_global_mask = torch.where(glo_confide.squeeze(1)<args.unlabel_threshold,1-pseudo_global_mask,pseudo_global_mask)
            glo_acc = (labels.argmax(1)==pseudo_global_mask).to(torch.float).mean((1,2))
        for participant_index in range(args.num_clients):
            netname = args.clients_model[participant_index]
            network = net_list[participant_index]
            optimizer = optimizer_clients[participant_index]
            if args.DataParallel:
                network = torch.nn.DataParallel(network).cuda()
            else:
                network = network.cuda()
            network.train()
            pred = network(images)
            mask = pred['mask']
            confide_loss = args.confide_theta*confidence_criterion(mask.clone().detach(),pred['confidence'],labels)
            pseudo_mask = mask.clone().argmax(1)
            confide_loss = args.confide_theta*confidence_criterion(mask.clone(),pred['confidence'],labels)
            acc = (labels.argmax(1)==pseudo_mask).to(torch.float).mean((1,2))
            klloss =  args.kl_theta*kl_criterion(mask,pre_global_mask.clone().detach(),(glo_acc>acc))
            loss = confide_loss
            loss = loss + seg_criterion(mask,labels)
            loss = loss +klloss   
            summary_writer.add_scalar("client-{}-{} with global model -kl and seg loss".format(participant_index,netname), loss.item(), epoch*args.fusionEpoch*args.batch_size+kl_epoch_index*args.batch_size+batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20==0:
                print('epoch[{}/{}],kl epoch[{}/{}],step: {} The client {} {} model learning with global {} on the public labeled images: loss: [{}] KLloss: [{}] confide loss: [{}]'.format(epoch,args.CommunicationEpoch,kl_epoch_index,args.fusionEpoch,batch_idx,participant_index,netname,args.global_name,loss,klloss,confide_loss))
                logging.info('epoch[{}/{}],kl epoch[{}/{}],step: {} The client {} {} model learning with global {} on the public labeled images: loss: [{}] KLloss: [{}] confide loss: [{}]'.format(epoch,args.CommunicationEpoch,kl_epoch_index,args.fusionEpoch,batch_idx,participant_index,netname,args.global_name,loss,klloss,confide_loss))
            optimizer_clients[participant_index] = optimizer
            net_list[participant_index] = network
    return net_list,optimizer_clients


def update_global_model_LSSL_with_weight(args,global_net, net_list, client_weight=None):
    if not client_weight:
        client_weight = [1 / len(net_list) for i in range(len(net_list))]
    print('Calculate the model avg----')
    params = dict(global_net.state_dict())
    ave_params = OrderedDict()
    for name, param in params.items():
        for client in range(len(net_list)):
            single_client_weight = client_weight[client]
            if client == 0:
                ave_params[name] = net_list[client].state_dict()[name] * single_client_weight
            else:
                ave_params[name] +=net_list[client].state_dict()[
                                     name]* single_client_weight
    print('Update each client model parameters----')
    global_net.load_state_dict(ave_params)
    for client in range(len(net_list)):
        net_list[client].load_state_dict(ave_params)
    
    return global_net,net_list

if __name__ == "__main__":
    args = get_args()
    # 设置日志文件保存路径
    exp_name = args.model_name+'-'+args.data+'-'+str(args.unlabel_threshold)+'-'+args.ver+'-'+str(args.base_lr)+str(args.global_name)+str(args.clients_model)
    if args.collaborate:
        exp_name = exp_name + '-collaborate'
    if args.use_unlabel:
        exp_name += '-use_unlabel'
    if args.need_fp:
        exp_name += '-use_fp'
    args.log_dir = os.path.join(args.log_path,'ours',args.model_name,exp_name,'logs')
    args.model_dir = os.path.join(args.log_path,'ours',args.model_name,exp_name,'models')
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.model_dir,exist_ok=True)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_writer = SummaryWriter(args.log_dir)
    logging.basicConfig(filename=os.path.join(args.log_dir,'train.log'),format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Hyperparameter setting{}'.format(args))
    print('Hyperparameter setting{}'.format(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # device = torch.device(args.device)
    
    assert args.num_clients==len(args.clients_model)==(len(args.labeled_ratio)-1),"num_clients not equal to clients_model or labeled_ratio"

    logging.info("Load Participants' Models")
    
    if not os.path.exists(os.path.join(args.split_path,'{}-{}-{}.json'.format(args.data,str(args.labeled_ratio),str(args.datasets)))):
        args.split_dict = get_datasplit(args)

    global_net,net_list,_,global_optimizer,optimizer_clients, \
                global_train_dataloader,labeled_train_client_dataloaders, \
                               unlabeled_train_client_dataloaders,val_client_dataloaders, \
                               test_client_dataloaders,dataset_labeled_nums,dataset_unlabeled_nums, \
                                   global_scheduler,scheduler_clients= initial_trainer(args)
    
    eval_list = init_eval_list(args)
    best_dice_list = init_eval_list(args)
    best_test_dice_list = init_eval_list(args)
    kl_theta = args.kl_theta
    unlabel_theta = args.unlabel_theta
    use_unlabel = args.use_unlabel
    # confide_theta = args.confide_theta
    for epoch_index in range(args.CommunicationEpoch):
        logging.info("The "+str(epoch_index)+" th Communication Epoch")
        '''
        Calculate Client Confidence with label quality and model performance
        '''
        if args.model_name not in ['local']:
            if epoch_index<args.pretrained_epoch:
                args.unlabel_theta = 0.
                args.kl_theta = 0.
                args.use_unlabel = False
            else:
                args.unlabel_theta = unlabel_theta
                args.kl_theta = kl_theta
                args.use_unlabel = True
                args.collaborate= True
        else:
            args.use_unlabel = use_unlabel
            args.collaborate= False
               
        if args.collaborate:
            if args.model_name in ['HSSF']:
                for kl_epoch_index in range(args.condensationEpoch):
                
                    global_net,global_optimizer = RegularityCondensation(args,global_net,net_list,global_optimizer,epoch_index,kl_epoch_index)
    
                    global_scheduler.step() 
                for kl_epoch_index in range(args.fusionEpoch):
                    
                    net_list,optimizer_clients = RegularityFusion(args,global_train_dataloader,global_net,net_list,optimizer_clients,epoch_index,kl_epoch_index)
                   
            else:
                global_net,net_list = update_global_model_LSSL_with_weight(args, global_net, net_list, client_weight=None)
                
                
        for local_epoch_index in range(args.localEpoch):
            if args.model_name in ['LSSL','local']:
                
                net_list,optimizer_clients = local_semi_surprised_learning(args, net_list, optimizer_clients, \
                    global_train_dataloader, unlabeled_train_client_dataloaders, epoch_index, local_epoch_index)
                
            else:
                
                net_list,optimizer_clients = local_semi_surprised_learning(args, net_list, optimizer_clients, \
                    labeled_train_client_dataloaders, unlabeled_train_client_dataloaders, epoch_index, local_epoch_index)
            
            for scheduler in scheduler_clients:
                scheduler.step()
    
        print('Evaluate Models')
        print('-'*50) 
        logging.info('Evaluate Models')
        logging.info('-'*50)
        if args.model_name in ['LSSL']:
            dice,prec,rec,iou = evaluate_network_all_data(args,network=global_net, dataloaders=val_client_dataloaders)
        else:
            dice,prec,rec,iou = evaluate_network_all_data_heterogeneous(args,networks=net_list, dataloaders=val_client_dataloaders)
        summary_writer.add_scalar("global {} val_dice: ".format(args.global_name), dice, epoch_index)
        summary_writer.add_scalar("global {} val_iou: ".format(args.global_name), iou, epoch_index)
        summary_writer.add_scalar("global {} val_precision: ".format(args.global_name), prec, epoch_index)
        summary_writer.add_scalar("global {} val_recall: ".format(args.global_name), rec, epoch_index)
        print('The global model {} on the all Validation images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(args.global_name,dice,iou,prec,rec))
        logging.info('The global model {} on the all Validation images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(args.global_name,dice,iou,prec,rec))
        eval_list = update_eval_list(participant_index = -1,dice=dice,prec=prec,rec = rec,iou = iou,eval_list = eval_list,epoch_index = epoch_index)
        if dice > best_dice_list[-1]['dice']:
            best_dice_list[-1]['dice'] = eval_list[-1]['dice']
            best_dice_list[-1]['iou'] = eval_list[-1]['iou']
            best_dice_list[-1]['precision'] = eval_list[-1]['precision']
            best_dice_list[-1]['recall'] = eval_list[-1]['recall']
            best_dice_list[-1]['best_epoch'] = eval_list[-1]['best_epoch']
            torch.save(global_net.state_dict(),os.path.join(args.model_dir,'global_{}_best.pth'.format(args.global_name)))
            if args.model_name in ['LSSL']:
                for participant_index in range(args.num_clients):
                    netname = args.clients_model[participant_index]
                    network = net_list[participant_index]#.cuda()
                    torch.save(network.state_dict(),os.path.join(args.model_dir,'{}_{}_best.pth'.format(netname,participant_index)))
                    
            if args.model_name in ['LSSL']:
                dice_test,prec_test,rec_test,iou_test = evaluate_network_all_data(args,network=global_net, dataloaders=test_client_dataloaders)
            else:
                dice_test,prec_test,rec_test,iou_test = evaluate_network_all_data_heterogeneous(args,networks=net_list, dataloaders=test_client_dataloaders)
            best_test_dice_list = update_test_list(participant_index=-1,dice = dice_test,prec = prec_test,rec = rec_test,iou = iou_test,eval_list = best_test_dice_list,epoch_index = epoch_index)
            print('The global model {} on the all test images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(args.global_name,dice_test,iou_test,prec_test,rec_test))
            logging.info('The global model {} on the all test images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(args.global_name,dice_test,iou_test,prec_test,rec_test))
        for participant_index in range(args.num_clients):
            netname = args.clients_model[participant_index]
            network = net_list[participant_index]#.cuda()
           
            if args.DataParallel:
                network = torch.nn.DataParallel(network).cuda()
            else:
                network = network.cuda()
            
            dice,prec,rec,iou = evaluate_network(args,network=network, dataloader=val_client_dataloaders[participant_index])
            summary_writer.add_scalar("client {} {} val_dice: ".format(participant_index,netname), dice, epoch_index)
            summary_writer.add_scalar("client {} {} val_iou: ".format(participant_index,netname), iou, epoch_index)
            summary_writer.add_scalar("client {} {} val_precision: ".format(participant_index,netname), prec, epoch_index)
            summary_writer.add_scalar("client {} {} val_recall: ".format(participant_index,netname), rec, epoch_index)
            print('The {} model {} on the Validation images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(participant_index,netname,dice,iou,prec,rec))
            logging.info('The {} model {} on the Validation images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(participant_index,netname,dice,iou,prec,rec))
            eval_list = update_eval_list(participant_index = participant_index,dice=dice,prec=prec,rec = rec,iou = iou,eval_list = eval_list,epoch_index = epoch_index)
            
            if eval_list[participant_index]['dice'] > best_dice_list[participant_index]['dice']:
                best_dice_list[participant_index]['dice'] = eval_list[participant_index]['dice']
                best_dice_list[participant_index]['iou'] = eval_list[participant_index]['iou']
                best_dice_list[participant_index]['precision'] = eval_list[participant_index]['precision']
                best_dice_list[participant_index]['recall'] = eval_list[participant_index]['recall']
                best_dice_list[participant_index]['best_epoch'] = eval_list[participant_index]['best_epoch']
                dice_test,prec_test,rec_test,iou_test = evaluate_network(args,network=network, dataloader=test_client_dataloaders[participant_index])
                
                best_test_dice_list = update_test_list(participant_index=participant_index,dice = dice_test,prec = prec_test,rec = rec_test,iou = iou_test,eval_list = best_test_dice_list,epoch_index = epoch_index)
                print('The {} model {} on the test images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(participant_index,netname,dice_test,iou_test,prec_test,rec_test))
                logging.info('The {} model {} on the test images: dice: [{}], iou: [{}], precision: [{}], recall: [{}]'.format(participant_index,netname,dice_test,iou_test,prec_test,rec_test))
                if args.model_name in ['local','HSSF']:
                    torch.save(network.state_dict(),os.path.join(args.model_dir,'{}_{}_best.pth'.format(netname,participant_index)))
        
        if epoch_index == args.CommunicationEpoch-1:
            torch.save(global_net.state_dict(),os.path.join(args.model_dir,'global_{}_last.pth'.format(args.global_name)))
            for participant_index in range(args.num_clients):
                netname = args.clients_model[participant_index]
                network = net_list[participant_index]
                torch.save(network.state_dict(),os.path.join(args.model_dir,'{}_{}_last.pth'.format(netname,participant_index)))
        print('+'*50)
        logging.info('+'*50)
        print('best val score: {}'.format(best_dice_list))  
        logging.info('best val score: {}'.format(best_dice_list))
        print('best test score: {}'.format(best_test_dice_list))  
        logging.info('best test score: {}'.format(best_test_dice_list))
        print('+'*50)  
        logging.info('+'*50)
                
    

    

