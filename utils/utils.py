import os
import torch
from networks.SegmentationModel import SegmentationModel,DeepLabHeadV3Plus #
from utils.datasets import get_datasplit,FEDDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import math
import cv2
from copy import deepcopy

def normal_loss(loss):
    loss = torch.sigmoid(loss)
    loss = (loss-0.5)*2
    return loss

def update_weight(num_labeled,num_unlabeled,loss_labeled,loss_unlabeled,pre_labeled_mean_loss_list,pre_unlabeled_mean_loss_list,theta = 1):
    num_labeled = torch.tensor(num_labeled)
    num_unlabeled = torch.tensor(num_unlabeled)
    weight = num_labeled*loss_labeled+theta*num_unlabeled*loss_unlabeled
    weight = weight/(num_labeled+num_unlabeled)
    weight = weight.sum()/weight
    weight = torch.softmax(weight,-1)
    if torch.isnan(weight).any():
        print(weight)
    
    return weight
    
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        # ema_param.data.mul_(alpha).add_(param.data, 1 - alpha)

def get_net(net_name,num_classes,ema=False):
    from networks.backbone.deeplab import Decoder
    if net_name=='resnet34':
        from networks.backbone.resnet import resnet34
        bkbone = resnet34()
        head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet50':
        from networks.backbone.resnet import resnet50 #
        bkbone = resnet50()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet18':
        from networks.backbone.resnet import resnet18 #
        bkbone = resnet18()
        head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet101':
        from networks.backbone.resnet import resnet101 #
        bkbone = resnet101()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet152':
        from networks.backbone.resnet import resnet152 #
        bkbone = resnet152()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name =='xception':
        from networks.backbone.xception import xception 
        bkbone = xception(pretrained=False)
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name =='mobilenetv2':
        from networks.backbone.mobilenetv2 import mobilenet_v2 #networks.
        bkbone = mobilenet_v2(pretrained=False)
        head = DeepLabHeadV3Plus(in_channels=1280,low_level_channels=24,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name =='vgg':
        from networks.backbone.vgg import VGG16 #networks.
        bkbone = VGG16(pretrained=False)
        head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name =='pvt':
        from networks.backbone.pvt import pvt_v2_b2 #networks.
        bkbone = pvt_v2_b2()
        head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    print("build net with encoder {}.".format(net_name))
    if ema:
            for param in net.parameters():
                param.detach_()
    return net

def build_global_model(args):
    net_name = args.global_name
    num_classes = args.num_classes
    net = get_net(net_name,num_classes)
    return net

def init_client_nets(args):
    nets_list = {net_i: None for net_i in range(args.num_clients)}
    ema_nets_list = {net_i: None for net_i in range(args.num_clients)}
    for net_i in range(args.num_clients):
        net_name = args.clients_model[net_i]
        net = get_net(net_name,args.num_classes)
        
        ema_net = ModelEMA(net, args.ema_decay)  
        nets_list[net_i] = net
        ema_nets_list[net_i] = ema_net
    return nets_list,ema_nets_list

def get_optimizer(optim_name,model,base_lr):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                        betas=(0.9, 0.999), weight_decay=5e-4)
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                    weight_decay=5e-4)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.02)
    
    return optimizer


def initial_trainer(args):
    #定义模型
    global_net = build_global_model(args=args)
    net_list,ema_net_list = init_client_nets(args)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    global_lf = lambda x: ((1 + math.cos(x * math.pi / (args.CommunicationEpoch*args.condensationEpoch))) / 2) * (1 - args.lrf) + args.lrf  # cosine
    client_lf = lambda x: ((1 + math.cos(x * math.pi / (args.CommunicationEpoch*args.condensationEpoch))) / 2) * (1 - args.lrf) + args.lrf  # cosine
    
    
    #定义优化器
    global_optimizer = get_optimizer(optim_name = args.global_optim,model = global_net,base_lr=args.base_lr)
    global_scheduler = lr_scheduler.LambdaLR(global_optimizer, lr_lambda=global_lf)
    print('global_optimizer',global_optimizer )
    # logging.info('global_optimizer',global_optimizer )
    optimizer_clients = []
    scheduler_clients = []
    for ind in range(len(net_list)):
        optimizer = get_optimizer(optim_name = args.client_optim,model = net_list[ind],base_lr=args.base_lr)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=client_lf)
        optimizer_clients.append(optimizer)
        scheduler_clients.append(scheduler)
        
    #定义全局dataset和dataloader # ['labeled_train', 'unlabeled_train', 'val_list', 'test_list']
    global_train_dataset = FEDDataset(args=args, dataset=args.datasets[0], transform = True, split ='labeled_train', noise = args.noise)
    global_train_dataloader = DataLoader(global_train_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True,shuffle=False)
    
    
    #定义局部dataset和dataloader
    labeled_train_client_dataloaders = []
    unlabeled_train_client_dataloaders = []
    val_client_dataloaders = []
    test_client_dataloaders = []
    labeled_train_client_datasets = []
    unlabeled_train_client_datasets = []
    val_client_datasets = []
    test_client_datasets = []
    dataset_labeled_nums = []
    dataset_unlabeled_nums = []
    for ind in range(1,len(args.datasets)):
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = True, split ='labeled_train', noise = args.noise)
        if len(client_dataset)<=1:
            labeled_train_client_dataloaders.append(None)
            labeled_train_client_datasets.append(None)
            # dataset_labeled_nums.append(0)
            dataset_labeled_nums.append(len(global_train_dataset))
        else:
            client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
            labeled_train_client_dataloaders.append(client_dataloader)
            labeled_train_client_datasets.append(client_dataset)
            # dataset_labeled_nums.append(len(client_dataset))
            dataset_labeled_nums.append(len(client_dataset)+len(global_train_dataset))
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = True, split ='unlabeled_train', noise = args.noise)
        if len(client_dataset)<=1:
            unlabeled_train_client_dataloaders.append(None)
            unlabeled_train_client_datasets.append(None)
            dataset_unlabeled_nums.append(0)
        else:
            client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
            unlabeled_train_client_dataloaders.append(client_dataloader)
            unlabeled_train_client_datasets.append(client_dataset)
            dataset_unlabeled_nums.append(len(client_dataset))
            # dataset_length += len(client_dataset)
        
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='val_list', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        val_client_dataloaders.append(client_dataloader)
        val_client_datasets.append(client_dataset)
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='test_list', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        test_client_dataloaders.append(client_dataloader)
        test_client_datasets.append(client_dataset)

    return global_net,net_list,ema_net_list,global_optimizer,optimizer_clients, \
                global_train_dataloader,labeled_train_client_dataloaders, \
                               unlabeled_train_client_dataloaders,val_client_dataloaders, \
                                   test_client_dataloaders,dataset_labeled_nums,dataset_unlabeled_nums, \
                                       global_scheduler,scheduler_clients

def exp_normalize(args,last_mean_loss_list,current_mean_loss_list,epoch_index):
    #EXP标准化处理
    quality_list = []
    amount_with_quality = [1 / (args.num_clients - 1) for i in range(args.num_clients)]
    amount_with_quality_exp = []
    weight_with_quality = []
    if args.Client_Confidence_Reweight:
        beta = 0.5
    else:
        beta = 0
    if epoch_index > 0 :
        for participant_index in range(args.num_clients):
            delta_loss = last_mean_loss_list[participant_index] - current_mean_loss_list[participant_index]
            quality_list.append(delta_loss / current_mean_loss_list[participant_index])
        quality_sum = sum(quality_list)
        for participant_index in range(args.num_clients):
            amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
            amount_with_quality_exp.append(np.exp(amount_with_quality[participant_index]))
        amount_with_quality_sum = sum(amount_with_quality_exp)
        for participant_index in range(args.num_clients):
            weight_with_quality.append(amount_with_quality_exp[participant_index] / amount_with_quality_sum)
    else:
        weight_with_quality = [1 / (args.num_clients - 1) for i in range(args.num_clients)]
    weight_with_quality = torch.tensor(weight_with_quality)
    return weight_with_quality
 
 
def visual_img(images,labels,boundary,ema_images,participant_index,mask,pre_boundary,ema_mask,ema_pre_boundary,flag ='local',save_dir = './plot'):
    os.makedirs(os.path.join(save_dir,flag),exist_ok=True)
    if flag == 'local':
        pred_image = deepcopy(images.detach())
        # pre_ema_image = deepcopy(ema_images.detach())
        # 原始图片及边界
        images[0,0][labels.argmax(1)[0]==1] = 1
        images[0,1][boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'loc_data_client_{}_image.jpg'.format(participant_index)),((images[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # # ema输入图片及边界
        # ema_images[0,0][labels.argmax(1)[0]==1] = 1
        # ema_images[0,1][boundary.argmax(1)[0]==1] = 1
        # cv2.imwrite(os.path.join(save_dir,flag,'loc_data_client_{}_ema_image.jpg'.format(participant_index)),((ema_images[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # 预测图片及边界
        pred_image[0,0][mask.argmax(1)[0]==1] = 1
        pred_image[0,1][pre_boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'predict_data_client_{}_image.jpg'.format(participant_index)),((pred_image[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # # ema预测图片及边界
        # pre_ema_image[0,0][ema_mask.argmax(1)[0]==1] = 1
        # pre_ema_image[0,1][ema_pre_boundary.argmax(1)[0]==1] = 1
        # cv2.imwrite(os.path.join(save_dir,flag,'predict_data_client_{}_ema_image.jpg'.format(participant_index)),((pre_ema_image[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
    if flag == 'unlabel_local':
        pred_image = deepcopy(images.detach())
        pre_ema_image = deepcopy(ema_images.detach())
        # 预测图片及边界
        pred_image[0,0][mask[0]==1] = 1
        pred_image[0,1][pre_boundary[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'predict_data_client_{}_image.jpg'.format(participant_index)),((pred_image[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # ema预测图片及边界
        pre_ema_image[0,0][ema_mask.argmax(1)[0]==1] = 1
        pre_ema_image[0,1][ema_pre_boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'predict_data_client_{}_ema_image.jpg'.format(participant_index)),((pre_ema_image[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
    if flag == 'server_local':
        pred_image = deepcopy(images.detach())
        # 原始图片及边界
        images[0,0][labels.argmax(1)[0]==1] = 1
        images[0,1][boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'loc_data_client_{}_image.jpg'.format(participant_index)),((images[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # 预测图片及边界
        pred_image[0,0][mask.argmax(1)[0]==1] = 1
        pred_image[0,1][pre_boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'predict_data_client_{}_image.jpg'.format(participant_index)),((pred_image[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
    if flag == 'kllocal':
        pred_image = deepcopy(images.detach())
        ema_images = deepcopy(images.detach())
        # 原始图片及边界
        images[0,0][labels.argmax(1)[0]==1] = 1
        images[0,1][boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'loc_data_client_{}_image.jpg'.format(participant_index)),((images[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # server预测图片及边界
        ema_images[0,0][ema_mask.argmax(1)[0]==1] = 1
        ema_images[0,1][ema_pre_boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'predict_server_image.jpg'),((ema_images[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        # 预测图片及边界
        pred_image[0,0][mask.argmax(1)[0]==1] = 1
        pred_image[0,1][pre_boundary.argmax(1)[0]==1] = 1
        cv2.imwrite(os.path.join(save_dir,flag,'predict_data_client_{}_image.jpg'.format(participant_index)),((pred_image[0]+1)/2*255).permute(1,2,0)[:,:,[2,1,0]].cpu().numpy())
        

class ModelEMA(object):
    '''
    from https://github.com/kekmodel/FixMatch-pytorch/blob/master/models/ema.py
    '''
    def __init__(self,  model, decay):
        self.ema = deepcopy(model)
        self.ema.cuda()
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model ,global_step):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        self.decay = min(1 - 1 / (global_step + 1), self.decay)
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])               
                

if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = get_net('resnet50',2).cuda()
    ema_model = ModelEMA(model,0.999)
    imput  = torch.randn((2,3,384,384)).cuda()
    out = model(imput)
    ema_model.update(model)
    print(ema_model)