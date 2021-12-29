import logging
from PIL import Image
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import *
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    elif args.dataset == 'nabirds':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.RandomCrop((448, 448)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.CenterCrop((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)
    elif args.dataset == 'myBirds':
        train_transforms = transforms.Compose([
                            transforms.Resize((600,600)),
                            transforms.RandomRotation(45),
                            transforms.RandomCrop((448, 448)),
                            AutoAugImageNetPolicy(),
                            # transforms.RandomResizedCrop(224),
                            # transforms.CenterCrop(100) , 
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        test_transforms = transforms.Compose([
                                transforms.Resize((600,600)),
                                transforms.CenterCrop((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        img_labels = pd.read_csv(args.img_labels, header=None, sep =" ")
        img_val = pd.read_csv(args.img_val,header=None,sep = ' ')
        trainset = myBirdsdataset(img_labels =img_labels, img_dir=args.img_dir, transform=train_transforms)
        valstest = myBirdsdataset(img_labels =img_val, img_dir=args.img_dir, transform=test_transforms)
        # testset = myBirdsdataset_test(img_order= args.img_order, img_dir=args.img_testdir, transform = test_transforms)
    # elif args.dataset == 'myFish':
    #     train_transforms = transforms.Compose([
    #                         transforms.Resize((600,600)),
    #                         transforms.RandomRotation(45),
    #                         transforms.RandomCrop((448, 448)),
    #                         AutoAugImageNetPolicy(),
    #                         # transforms.RandomResizedCrop(224),
    #                         # transforms.CenterCrop(100) , 
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                     ])
    #     test_transforms = transforms.Compose([
    #                             transforms.Resize((600,600)),
    #                             transforms.CenterCrop((448, 448)),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                         ])
    #     img_labels = pd.read_csv(args.img_labels, header=None, sep =" ")
    #     img_val = pd.read_csv(args.img_val,header=None,sep = ' ')
    #     trainset = myFishdataset(img_labels =img_labels, transform=train_transforms)
    #     valstest = myFishdataset(img_labels =img_val,transform=test_transforms)
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valstest) if args.local_rank == -1 else DistributedSampler(valstest)
    # test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True, shuffle=False)
    #validation purposes
    test_loader = DataLoader(valstest,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True, shuffle=False) if valstest is not None else None
    # test_loader = DataLoader(testset,
    #                         #  sampler=test_sampler,
    #                          batch_size=args.eval_batch_size,
    #                          num_workers=4,
    #                          pin_memory=True, shuffle=False) if testset is not None else None

    return train_loader, test_loader
def get_fishloader (args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.dataset == 'myFish':
        train_transforms = transforms.Compose([
                            transforms.Resize((448,448)),
                            transforms.RandomRotation(45),
                            # transforms.RandomCrop((448, 448)),
                            AutoAugImageNetPolicy(),
                            # transforms.RandomResizedCrop(224),
                            # transforms.CenterCrop(100) , 
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        test_transforms = transforms.Compose([
                                transforms.Resize((448,448)),
                                # transforms.CenterCrop((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        img_labels = pd.read_csv(args.img_labels, header=None, sep =" ")
        img_val = pd.read_csv(args.img_val,header=None,sep = ' ')
        if args.docrop == True:
            print('here')
            trainset = myFishdataset(img_labels =img_labels,transform=train_transforms)
            valstest = myFishdataset(img_labels =img_val,transform=test_transforms)
            
        else:
            trainset = myFishdataset_withoutcrop(img_labels =img_labels, img_dir=args.img_dir,transform=train_transforms)
            valstest = myFishdataset_withoutcrop(img_labels =img_val,img_dir=args.img_dir,transform=test_transforms)
    if args.local_rank == 0:
        torch.distributed.barrier()
    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    # val_sampler = SequentialSampler(valstest) if args.local_rank == -1 else DistributedSampler(valstest)
    # test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                            #   sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True, shuffle=False)
    #validation purposes
    val_loader = DataLoader(valstest,
                            #  sampler=val_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True, shuffle=False) if valstest is not None else None
    return train_loader,val_loader


    
def get_testloader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.dataset == 'myFish':
        
        test_transforms = transforms.Compose([
                                transforms.Resize((448,448)),
                                # transforms.CenterCrop((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])

        testset = myFishdataset_test(img_order = args.img_order, img_dir=args.img_testdir, transform = test_transforms)
    if args.local_rank == 0:
        torch.distributed.barrier()

    # test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    test_loader = DataLoader(testset,
                            #  sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                            #  num_workers=4,
                            #  pin_memory=True, 
                             shuffle=False) if testset is not None else None

    return test_loader