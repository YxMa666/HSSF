import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from utils.utils import init_client_nets,build_global_model,get_net
from utils.datasets import get_datasplit,FEDDataset
from torch.utils.data import DataLoader
from utils.metrics import precision,recall,dice_score,iou_score
from eln_net.ELN import ELNetwork
from eln_net.deeplab import DeepLab, Decoder
from medpy.metric.binary import hd, hd95, dc, jc, assd
import surface_distance

def recursive_sum(lst):
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += recursive_sum(item)  # 递归调用
        else:
            total += item
    return total


def get_eln_net(args,ema=False):
    enc = DeepLab(resnet_name=args.net_name)
    dec = Decoder(num_cls=args.num_classes)
    return enc,dec

def init_client_eln_nets(args):
    nets_list = {net_i: None for net_i in range(args.num_clients)}
    ema_nets_list = {net_i: None for net_i in range(args.num_clients)}
    for net_i in range(args.num_clients):
        enc,dec = get_eln_net(args)
        nets_list[net_i] = {'enc':enc,'dec':dec}
        enc,dec = get_eln_net(args,ema=True)
        ema_nets_list[net_i] = {'enc':enc,'dec':dec}
    return nets_list,ema_nets_list


def initial_eln_tester(args):
    #定义模型
    global_enc,global_dec = get_eln_net(args)
    net_list,ema_net_list = init_client_eln_nets(args)   
    global_net = {'enc':global_enc,'dec':global_dec}
    
    #定义局部dataset和dataloader
    val_client_dataloaders = []
    test_client_dataloaders = []
    val_client_datasets = []
    test_client_datasets = []
    for ind in range(1,len(args.datasets)):
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='val_list', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        val_client_dataloaders.append(client_dataloader)
        val_client_datasets.append(client_dataset)
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='test_list', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        test_client_dataloaders.append(client_dataloader)
        test_client_datasets.append(client_dataset)

    return global_net,net_list,ema_net_list,val_client_dataloaders,test_client_dataloaders

def initial_tester(args):
    #定义模型
    global_net = build_global_model(args=args)
    net_list,ema_net_list = init_client_nets(args)
    
    #定义局部dataset和dataloader
    
    val_client_dataloaders = []
    test_client_dataloaders = []
    val_client_datasets = []
    test_client_datasets = []
   
    for ind in range(1,len(args.datasets)):
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='val_list', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        val_client_dataloaders.append(client_dataloader)
        val_client_datasets.append(client_dataset)
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='test_list', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        test_client_dataloaders.append(client_dataloader)
        test_client_datasets.append(client_dataset)
        # client_dataset = FEDDataset(args=args, dataset=args.datasets[0], transform = True, split ='labeled_train', noise = False)
        # client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        # test_client_dataloaders.append(client_dataloader)
        # test_client_datasets.append(client_dataset)

    return global_net,net_list,ema_net_list,val_client_dataloaders,test_client_dataloaders 

def metrics_others(args,net_list,val_client_dataloaders,test_client_dataloaders):           
    print('Evaluate Models')
    print('-'*50)
    print('{} all result:'.format(args.model_name))
    evaluate_network_others(args,net_list=net_list, dataloaders=val_client_dataloaders,key='validation')
    evaluate_network_others(args,net_list=net_list, dataloaders=test_client_dataloaders,key='test')
    print('+'*50)
    
def get_f1(prec,recall,length):
    prec = prec/length
    recall = recall/length
    return 2*prec*recall/(prec+recall)

@torch.no_grad()
def evaluate_network_others(args,net_list, dataloaders,key='validation'):
    os.makedirs(os.path.join(args.save_dir,'image'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'mask'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'pred'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'proto'),exist_ok=True)
    dice = []
    iou = []
    length = []
    asd = []
    hd95_s = []
    num = []
    over_dice_format = f"result: & "
    over_hd95_format = f" & "
    for i in range(args.num_clients):
        dice.append([])
        iou.append([])
        length.append([])
        asd.append([])
        hd95_s.append([])
        num.append([])
    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        network = net_list[i]
        network.load_state_dict(torch.load(os.path.join(args.model_dir,f'{args.clients_model[i]}_{i}_best.pth')))
        network = network.cuda()
        network.eval()
        for images, labels,_,img_name in dataloader:
            img_name = img_name[0][:-4]+'_'+str(i)+'_'+args.model_name+img_name[0][-4:]
            images = images.cuda()
            labels = labels.cuda()
            begin = time.time()
            pred = network(images) #(4,1,384,384)
            end = time.time()
            print(f"spend:{(end-begin):.2f}")
            mask = pred['mask']
            mask = mask.argmax(1).unsqueeze(1)
             
            dice[i].append(dice_score(labels, mask).cpu().numpy()*100) 
            iou[i].append(iou_score(labels, mask).cpu().numpy()*100) 
            
            length[i].append(len(labels))
            try:
                asd[i].append(assd(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
                hd95_s[i].append(hd95(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
            except RuntimeError:
                num[i].append(len(labels))

            cv2.imwrite(os.path.join(args.save_dir,'image',img_name),((images[0]+1)/2*255)[[2,1,0]].permute(1,2,0).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name.replace(args.model_name,'label')),(labels[0][0]*255).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name),(mask[0][0]*255).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'proto',img_name),(pred['proto'][0][0]*255).cpu().numpy())
            # np.save(os.path.join(args.save_dir,'pred',img_name).replace('png','npy'),pred['mask'].cpu().numpy())
            # np.save(os.path.join(args.save_dir,'proto',img_name).replace('png','npy'),pred['proto'].cpu().numpy())
        over_dice_format+= f"& {torch.tensor(dice[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(dice[i]).std():.2f} "
        over_hd95_format+= f"& {torch.tensor(hd95_s[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(hd95_s[i]).std():.2f} "
    #     print(f'The global model {args.global_name} on {i} {key} images: dice: [{dice[i]/length[i]:.4f}], iou: [{iou[i]/length[i]:.4f}], ASSD: [{asd[i]/(length[i]-num[i]):.4f}], hd95: [{hd95_s[i]/(length[i]-num[i]):.4f}]')
    # print(f'The global model {args.global_name} on all {key} images: dice: [{dice.sum()/length.sum():.4f}], iou: [{iou.sum()/length.sum():.4f}], ASSD: [{asd.sum()/(length.sum()-num.sum()):.4f}], hd95: [{hd95_s.sum()/(length.sum()-num.sum()):.4f}]')
    all_dice = [item for sublist in dice for item in sublist]
    all_hd = [item for sublist in hd95_s for item in sublist]
    over_dice_format+= f"& {torch.tensor(all_dice).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_dice).std():.2f} "
    over_hd95_format+= f"& {torch.tensor(all_hd).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_hd).std():.2f} "
    print(over_dice_format+over_hd95_format)
    return

def metrics_eln(args,global_net,net_list,val_client_dataloaders,test_client_dataloaders):
    global_net['enc'].load_state_dict(torch.load(os.path.join(args.model_dir,'global_enc_{}_best.pth'.format(args.net_name))))
    global_net['enc'] = global_net['enc'].cuda()
    global_net['dec'].load_state_dict(torch.load(os.path.join(args.model_dir,'global_dec_{}_best.pth'.format(args.net_name))))
    global_net['dec'] = global_net['dec'].cuda()
    print('Evaluate Models')
    print('-'*50)
    print('{} all result:'.format(args.model_name))
    evaluate_network_eln(args,global_net=global_net, dataloaders=val_client_dataloaders,key='validation')
    evaluate_network_eln(args,global_net=global_net, dataloaders=test_client_dataloaders,key='test')
    print('+'*50)
               
    

@torch.no_grad()
def evaluate_network_eln(args,global_net, dataloaders,key='validation'):
    os.makedirs(os.path.join(args.save_dir,'image'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'mask'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'pred'),exist_ok=True)
    global_net['enc'].eval()
    global_net['dec'].eval()
    dice = []
    iou = []
    length = []
    asd = []
    hd95_s = []
    num = []
    over_dice_format = f"result: & "
    over_hd95_format = f" & "
    for i in range(args.num_clients):
        dice.append([])
        iou.append([])
        length.append([])
        asd.append([])
        hd95_s.append([])
        num.append([])
    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        for images, labels,_,img_name in dataloader:
            img_name = img_name[0][:-4]+'_'+str(i)+'_'+args.model_name+img_name[0][-4:]
            images = images.cuda()
            labels = labels.cuda()
            a, b = global_net['enc'](images)
            mask, _ = global_net['dec'](a, b)
            mask = F.interpolate(
            mask, size=labels.size()[2:], mode="bilinear", align_corners=True
        )
            mask = mask.argmax(1).unsqueeze(1)

            dice[i].append(dice_score(labels, mask).cpu().numpy()*100) 
            iou[i].append(iou_score(labels, mask).cpu().numpy()*100) 
            
            length[i].append(len(labels))
            try:
                asd[i].append(assd(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
                hd95_s[i].append(hd95(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
            except RuntimeError:
                num[i].append(len(labels))

            cv2.imwrite(os.path.join(args.save_dir,'image',img_name),((images[0]+1)/2*255)[[2,1,0]].permute(1,2,0).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name.replace(args.model_name,'label')),(labels[0][0]*255).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name),(mask[0][0]*255).cpu().numpy())
        over_dice_format+= f"& {torch.tensor(dice[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(dice[i]).std():.2f} "
        over_hd95_format+= f"& {torch.tensor(hd95_s[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(hd95_s[i]).std():.2f} "
    #     print(f'The global model {args.global_name} on {i} {key} images: dice: [{dice[i]/length[i]:.4f}], iou: [{iou[i]/length[i]:.4f}], ASSD: [{asd[i]/(length[i]-num[i]):.4f}], hd95: [{hd95_s[i]/(length[i]-num[i]):.4f}]')
    # print(f'The global model {args.global_name} on all {key} images: dice: [{dice.sum()/length.sum():.4f}], iou: [{iou.sum()/length.sum():.4f}], ASSD: [{asd.sum()/(length.sum()-num.sum()):.4f}], hd95: [{hd95_s.sum()/(length.sum()-num.sum()):.4f}]')
    all_dice = [item for sublist in dice for item in sublist]
    all_hd = [item for sublist in hd95_s for item in sublist]
    over_dice_format+= f"& {torch.tensor(all_dice).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_dice).std():.2f} "
    over_hd95_format+= f"& {torch.tensor(all_hd).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_hd).std():.2f} "
    print(over_dice_format+over_hd95_format)
    #     print(f'The global model {args.global_name} on {i} {key} images: dice: [{dice[i]/length[i]:.4f}], iou: [{iou[i]/length[i]:.4f}], ASSD: [{asd[i]/(length[i]-num[i]):.4f}], hd95: [{hd95_s[i]/(length[i]-num[i]):.4f}], precision: [{prec[i]/length[i]:.4f}], recall: [{rec[i]/length[i]:.4f}], f1:[{get_f1(prec[i],rec[i],length[i]):.4f}], acc:[{acc[i]/length[i]:.4f}]')
    # print(f'The global model {args.global_name} on all {key} images: dice: [{dice.sum()/length.sum():.4f}], iou: [{iou.sum()/length.sum():.4f}], ASSD: [{asd.sum()/(length.sum()-num.sum()):.4f}], hd95: [{hd95_s.sum()/(length.sum()-num.sum()):.4f}], precision: [{prec.sum()/length.sum():.4f}], recall: [{rec.sum()/length.sum():.4f}], f1:[{get_f1(prec.sum(),rec.sum(),length.sum()):.4f}], acc:[{acc[i]/length[i]:.4f}]')
    return 

def evaluate_network_all_data(network,dataloaders):
    network = network.cuda()
    network.eval()
    with torch.no_grad():
        prec = 0.
        rec = 0.
        dice = 0.
        iou = 0.
        length = 0
        for dataloader in dataloaders:
            for images, labels,_,id in dataloader:
                images = images.cuda()
                labels = labels.cuda()#.squeeze(1)
                pred = network(images) #(4,1,384,384)
                mask = pred['mask']
                # mask = (torch.sigmoid(mask)>0.5).to(torch.float)
                mask = mask.argmax(1).unsqueeze(1)
                prec += precision(labels,mask)
                rec += recall(labels, mask)
                dice += dice_score(labels, mask)
                iou += iou_score(labels, mask)
                length += len(labels)
    return dice/length,prec/length,rec/length,iou/length

@torch.no_grad()
def evaluate_network_fedavg(args,network, dataloaders,key='validation'):
    os.makedirs(os.path.join(args.save_dir,'image'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'mask'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'pred'),exist_ok=True)
    # print(evaluate_network_all_data(network,dataloaders))
    network.eval()
    dice = []
    iou = []
    length = []
    asd = []
    hd95_s = []
    num = []
    over_dice_format = f"result: & "
    over_hd95_format = f" & "
    for i in range(args.num_clients):
        dice.append([])
        iou.append([])
        length.append([])
        asd.append([])
        hd95_s.append([])
        num.append([])
    
    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        for images, labels,_,img_name in dataloader:
            img_name = img_name[0][:-4]+'_'+str(i)+'_'+args.model_name+img_name[0][-4:]
            images = images.cuda()
            labels = labels.cuda()
            begin = time.time()
            pred = network(images) #(4,1,384,384)
            end = time.time()
            print(f"spend:{(end-begin):.2f}")
            mask = pred['mask']
            mask = mask.argmax(1).unsqueeze(1)
            dice[i].append(dice_score(labels, mask).cpu().numpy()*100) 
            iou[i].append(iou_score(labels, mask).cpu().numpy()*100) 
            
            length[i].append(len(labels))
            try:
                asd[i].append(assd(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
                hd95_s[i].append(hd95(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
            except RuntimeError:
                num[i].append(len(labels))
            
            cv2.imwrite(os.path.join(args.save_dir,'image',img_name),((images[0]+1)/2*255)[[2,1,0]].permute(1,2,0).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name.replace(args.model_name,'label')),(labels[0][0]*255).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name),(mask[0][0]*255).cpu().numpy())
        over_dice_format+= f"& {torch.tensor(dice[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(dice[i]).std():.2f} "
        over_hd95_format+= f"& {torch.tensor(hd95_s[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(hd95_s[i]).std():.2f} "
    #     print(f'The global model {args.global_name} on {i} {key} images: dice: [{dice[i]/length[i]:.4f}], iou: [{iou[i]/length[i]:.4f}], ASSD: [{asd[i]/(length[i]-num[i]):.4f}], hd95: [{hd95_s[i]/(length[i]-num[i]):.4f}]')
    # print(f'The global model {args.global_name} on all {key} images: dice: [{dice.sum()/length.sum():.4f}], iou: [{iou.sum()/length.sum():.4f}], ASSD: [{asd.sum()/(length.sum()-num.sum()):.4f}], hd95: [{hd95_s.sum()/(length.sum()-num.sum()):.4f}]')
    all_dice = [item for sublist in dice for item in sublist]
    all_hd = [item for sublist in hd95_s for item in sublist]
    over_dice_format+= f"& {torch.tensor(all_dice).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_dice).std():.2f} "
    over_hd95_format+= f"& {torch.tensor(all_hd).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_hd).std():.2f} "
    print(over_dice_format+over_hd95_format)
    #     print(f'The global model {args.global_name} on {i} {key} images: dice: [{dice[i]/length[i]:.4f}], iou: [{iou[i]/length[i]:.4f}], ASSD: [{asd[i]/(length[i]-num[i]):.4f}], hd95: [{hd95_s[i]/(length[i]-num[i]):.4f}], precision: [{prec[i]/length[i]:.4f}], recall: [{rec[i]/length[i]:.4f}], f1:[{get_f1(prec[i],rec[i],length[i]):.4f}], acc:[{acc[i]/length[i]:.4f}]')
    # print(f'The global model {args.global_name} on all {key} images: dice: [{dice.sum()/length.sum():.4f}], iou: [{iou.sum()/length.sum():.4f}], ASSD: [{asd.sum()/(length.sum()-num.sum()):.4f}], hd95: [{hd95_s.sum()/(length.sum()-num.sum()):.4f}], precision: [{prec.sum()/length.sum():.4f}], recall: [{rec.sum()/length.sum():.4f}], f1:[{get_f1(prec.sum(),rec.sum(),length.sum()):.4f}], acc:[{acc[i]/length[i]:.4f}]')
    return 

def metrics_fedavg(args,global_net,val_client_dataloaders,test_client_dataloaders):
    global_net.load_state_dict(torch.load(os.path.join(args.model_dir,'global_{}_best.pth'.format(args.global_name))))
    global_net = global_net.cuda()           
    print('Evaluate Models')
    print('-'*50)
    print('{} all result:'.format(args.model_name))
    evaluate_network_fedavg(args,network=global_net, dataloaders=val_client_dataloaders,key='validation')
    evaluate_network_fedavg(args,network=global_net, dataloaders=test_client_dataloaders,key='test')
    print('+'*50)

def get_model_dir(model_name):
    if model_name == 'base':
        model_path = "./log/ours/local/local-polyp-0.3-v2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_fp/models"
        # model_path = "./log/ours/local/local-polyp-0.3-v2_unlabel-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_unlabel-use_fp/models"
        # model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
        # model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
        # model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v2_norpg-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'baseSA':
        # model_path = "./log/ours/local/local-polyp-0.3-v2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_fp/models"
        model_path = "./log/ours/local/local-polyp-0.3-v2_unlabel-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_unlabel-use_fp/models"
        # model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
        # model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'baseSARPG':
        model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v2_norpg-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
        # model_path = "./log/ours/local/local-polyp-0.3-v2_unlabel-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_unlabel-use_fp/models"
        # model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'HSSF':
        # model_path = "./log/ours/local/local-polyp-0.3-v2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_fp/models"
        # model_path = "./log/ours/local/local-polyp-0.3-v2_unlabel-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_unlabel-use_fp/models"
        model_path = "./log/ours/semifedseg/semifedseg-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"        
    elif model_name == 'ours':
        model_path = "./log/ours/fedavg/fedavg-polyp-0.3-v1-0.0001resnet34['resnet34', 'resnet34', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'ours_0.0':
        model_path = "./unlabel_thres_log/ours/semifedseg/semifedseg-polyp-0.0-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'ours_0.6':
        model_path = "./unlabel_thres_log/ours/semifedseg/semifedseg-polyp-0.6-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'ours_0.9':
        model_path = "./unlabel_thres_log/ours/semifedseg/semifedseg-polyp-0.9-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'Local_label':
        model_path = "./log/ours/local/local-polyp-0.3-v2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_fp/models"
    elif model_name == 'Local_unlabel':
        model_path = "./log/ours/local/local-polyp-0.3-v2_unlabel-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-use_unlabel-use_fp/models"
    elif model_name == 'MT':
        model_path = "./log/MT/MT-polyp--v1-0.0001resnet34['resnet34', 'resnet34', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-use_unlabel/models"
    elif model_name == 'fixmatch':
        model_path = "./log/fixmatch/fixmatch-polyp-v2-0.0001resnet34['resnet34', 'resnet34', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-use_unlabel/models"
    elif model_name == 'unimatch':
        model_path = "./log/unimatch/unimatch-polyp-v2-0.0001resnet34['resnet34', 'resnet34', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-use_unlabel/models"
    elif model_name == 'RHFL':
        model_path = "./log/RHFL/RHFL-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-10-1-kl_loss-use_unlabel-use_fp/models"
    elif model_name == 'eln':
        model_path = "./log/eln/eln-polyp-v1-resnet34[1, 0.0, 0.0, 0.0, 0.0]-collaborate-use_unlabel/models"
    elif model_name == 'fedmd':
        model_path = "./log/fedmd/fedmd-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-10-1-kl_loss-use_unlabel-use_fp/models"
    elif model_name == 'ktpfl':
        model_path = "./log/kt_pfl/kt_pfl-polyp-0.3-v1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-10-1-kl_loss-use_unlabel-use_fp/models"
    elif model_name == 'feddistill':
        model_path = "./log/feddistill/feddistill-polyp-0.3-v1-1e-05resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'fedproto':
        model_path = "./log/FedProto/FedProto-polyp-0.3-v1-1e-05resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34'][1, 0.0, 0.0, 0.0, 0.0]2-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'semifl':
        model_path = "./test_log/ours/semifl/semifl-polyp-v4-0.0001resnet34['resnet34', 'resnet34', 'resnet34', 'resnet34']-collaborate-use_unlabel/models"
    elif model_name == 'hp_l_0.5':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vl_0.5-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_l_0.25':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vl_0.25-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_l_1':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vl_1-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_lr_0.01':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vlr_0.01-0.01resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_lr_0.001':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vlr_0.001-0.001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_lr_0.00001':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vlr_0.00001-1e-05resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_u_2':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vu_2-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_u_8':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vu_8-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    elif model_name == 'hp_u_16':
        model_path = "./hparam_log/ours/semifedseg/semifedseg-polyp-0.3-vu_16-0.0001resnet101['resnet18', 'resnet18', 'resnet34', 'resnet34']-collaborate-use_unlabel-use_fp/models"
    
    else:
        print("not define!!!")
    return model_path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='ours',help="selection from list:[semifl,semifedseg,ours,eln,MT,fixmatch,unimatch,fedmd,ktpfl,feddistill,fedproto, \
                                                                                            RHFL,Local_label,Local_unlabel]")
    parser.add_argument('--data',type=str,default='polyp',help="selection from list:[polyp,polypv1,prostate,Melanoma,Fundus,isic,isicv1]")
    parser.add_argument('--datasets',type=list,default=['CVC-ColonDB', 'CVC-ClinicDB', 'EndoTect-ETIS', 'CVC-300', 'Kvasir'],help="selection from list:[PH2,domain1,domain2,domain3,domain4]")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--global_name', type=str, default='resnet34',help='[resnet34,resnet50,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--clients_model', type=list, default=['resnet18','resnet18','resnet34','resnet34'],help='[resnet34,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--labeled_ratio', type=list, default=[1,0.,0.,0.,0.],help='Proportion of labeled data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--DataParallel', type=bool, default=False)
    parser.add_argument('--shape', type=tuple, default=384)
    parser.add_argument('--train_val_test', type=tuple, default=(0.8,0.1,0.1))
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
    parser.add_argument('--device',type=str,default='0',help="device id")
    parser.add_argument('--img_path', type=str,
                        default='./data',
                        help='path to data')
    parser.add_argument('--split_path', type=str,
                        default='./data_split',
                        help='path to log')
    parser.add_argument('--save_dir', type=str,
                        default='./cost_test/albation',
                        help='path to log')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    # device = torch.device(args.device)
    
    assert args.num_clients==len(args.clients_model)==(len(args.labeled_ratio)-1),"num_clients not equal to clients_model or labeled_ratio"

    if not os.path.exists(os.path.join(args.split_path,'{}-{}-{}.json'.format(args.data,str(args.labeled_ratio),str(args.datasets)))):
        args.split_dict = get_datasplit(args)
    args.model_dir = get_model_dir(args.model_name)
    if args.model_name in ['eln']:
        args.net_name = args.global_name
        args.save_dir = os.path.join(args.save_dir,'fedavg')
        global_net,net_list,_,val_client_dataloaders,test_client_dataloaders = initial_eln_tester(args)
        metrics_eln(args,global_net,net_list,val_client_dataloaders,test_client_dataloaders)
    elif args.model_name in ['ours','MT','fixmatch','unimatch','semifl']:
        args.save_dir = os.path.join(args.save_dir,'fedavg')
        args.net_name = args.global_name
        # if args.model_name in ['fixmatch','unimatch']:
        #     args.shape = 512
        global_net,_,_,val_client_dataloaders,test_client_dataloaders = initial_tester(args)
        metrics_fedavg(args, global_net, val_client_dataloaders, test_client_dataloaders)
    else:
        args.save_dir = os.path.join(args.save_dir,'others')
        _,net_list,_,val_client_dataloaders,test_client_dataloaders = initial_tester(args)
        metrics_others(args,net_list,val_client_dataloaders,test_client_dataloaders)
    
    
    
    
        
                