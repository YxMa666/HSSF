import os
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import random
import json
from PIL import Image
random.seed(42)
import torch
from torchvision import transforms
from utils.randaugment import RandAugmentMC


class FEDDataset(Dataset):
    """ FED Dataset """
    def __init__(self, args, dataset, transform = True, split='labeled_train', noise=False):
        self.root_dir = os.path.join(args.img_path,args.data,dataset) 
        self.split_dict = json.load(open(os.path.join(args.split_path,'{}-{}-{}.json'.format(args.data,str(args.labeled_ratio),str(args.datasets))),'r'))
        self.transform = transform 
        self.trainsize = args.shape
        self.noise = noise
        self.noise_transform = RandomNoise()
        self.split = split # ['labeled_train', 'unlabeled_train', 'val_list', 'test_list']
        self.image_list = self.split_dict[dataset][split]
        print("total {} slices".format(len(self.image_list)))
        
        if self.transform:
            if split == 'unlabeled_train':
                print('unlabeled data Using RandomRotation, RandomFlip')
                self.weak = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(size=self.trainsize,
                                        padding=int(self.trainsize*0.125),
                                        padding_mode='reflect'),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5])],)
                self.strong = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(size=self.trainsize,
                                        padding=int(self.trainsize*0.125),
                                        padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5])])
                
                self.gt_transform = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()])
                
            else:
                print('Using RandomRotation, RandomFlip')
                self.img_transform = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5])
                    ])
                self.gt_transform = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])
                ])
            
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        idx = idx % len(self.image_list)
        image = self.rgb_loader(os.path.join(self.root_dir,'image',self.image_list[idx]))
        mask = self.binary_loader(os.path.join(self.root_dir,'mask',self.image_list[idx]))
        boundary = self.binary_loader(os.path.join(self.root_dir,'boundary',self.image_list[idx]))
        image,mask,boundary = self.resize(image,mask,boundary)
        if self.split == 'unlabeled_train':
            seed = np.random.randint(42) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            w_image = self.weak(image)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            s_image = self.strong(image)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            mask = self.gt_transform(mask)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            boundary = self.gt_transform(boundary)
        else:
            seed = np.random.randint(42) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            image = self.img_transform(image)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            mask = self.gt_transform(mask)
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            boundary = self.gt_transform(boundary)
            if self.noise:
                image = self.noise_transform(image)
        if self.split=='unlabeled_train':
            return w_image,s_image,self.image_list[idx]
        return image,(mask>0).to(torch.float),(boundary>0).to(torch.float),self.image_list[idx]
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt, boundary):
        assert img.size == gt.size == boundary.size
        w, h = img.size
        h = self.trainsize # max(h, self.trainsize)
        w = self.trainsize # max(w, self.trainsize)
        return img.resize((w, h), Image.Resampling.BILINEAR), gt.resize((w, h), Image.Resampling.NEAREST), boundary.resize((w, h), Image.Resampling.NEAREST)
    
class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, image):
        noise = torch.clamp(
            torch.rand(image.size()) * self.sigma, -2 * self.sigma,
            2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return image
    
    
def get_datasplit(args):
    """
    """
    data_root = os.path.join(args.img_path,args.data) # ../data/polyp
    client_datasets = args.datasets
    labeled_ratio = args.labeled_ratio # [1,0,0,0,0]
    split_path = args.split_path
    os.makedirs(split_path,exist_ok=True)
    assert len(labeled_ratio)==len(client_datasets),"the labeled_ratio is not equal to client_datasets"
    tra,val,test = args.train_val_test # 0.8:0.1:0.1
    split_dict = {}
    dict1 = {}
    ind = 0
    img_list = os.listdir(os.path.join(data_root,client_datasets[ind],'image'))
    random.shuffle(img_list)
    img_length = len(img_list)
    labeled_train = img_list
    dict1['labeled_train'] = labeled_train
    split_dict[client_datasets[ind]] = dict1
    ind += 1
    for dataset in client_datasets[1:]:
        dict1 = {}
        img_list = os.listdir(os.path.join(data_root,dataset,'image'))
        random.shuffle(img_list)
        img_length = len(img_list)
        labeled_train = img_list[:int(img_length*tra*labeled_ratio[ind])]
        unlabeled_train = img_list[int(img_length*tra*labeled_ratio[ind]):int(img_length*tra)]
        val_list = img_list[int(img_length*tra):int(img_length*tra+img_length*val)]
        test_list = img_list[int(img_length*tra+img_length*val):]
        dict1['labeled_train'] = labeled_train
        dict1['unlabeled_train'] = unlabeled_train
        dict1['val_list'] = val_list
        dict1['test_list'] = test_list
        split_dict[dataset] = dict1
        ind+=1
    json.dump(split_dict,open(os.path.join(split_path,'{}-{}-{}.json'.format(args.data,str(args.labeled_ratio),str(args.datasets))),'w'))
    return split_dict
        
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.img_path = './data'
    args.data = 'polyp'
    args.datasets = ['EndoTect-ETIS', 'CVC-300', 'CVC-ColonDB', 'CVC-ClinicDB', 'Kvasir']
    args.shape = 384
    args.train_val_test = (0.7,0.15,0.15)
    assert sum(args.train_val_test)==1,"Training validation test division beyond limits"
    args.labeled_ratio = [1,0.5,0,0,0]
    args.split_path = './split_data'
    get_datasplit(args)
    split_dict = json.load(open('./split_data/polyp.json','r'))
    # print(split_dict)
    for str1 in ['EndoTect-ETIS', 'CVC-300', 'CVC-ColonDB', 'CVC-ClinicDB', 'Kvasir']:
        dataset = FEDDataset(args=args, dataset=str1, transform = True, split = 'val_list', noise = True)
        print(str1,len(dataset))
    for i in dataset:
        import cv2
        cv2.imwrite('./img.jpg',((i[0]+1)/2*255).permute(1,2,0).numpy()[:,:,[2,1,0]])
        cv2.imwrite('./mask.jpg',(i[1]*255).squeeze(0).numpy())
        cv2.imwrite('./boundary.jpg',(i[2]*255).squeeze(0).numpy())
        print(i[3])