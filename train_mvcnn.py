import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import time
from tools.my_loss import soft_margin_triplet_centor_loss,improved_soft_Triplet_Center_Loss,Triplet_Center_Loss
from tools.Trainer import ModelNetTrainer,ModelNetTrainer_oneloss
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset,SingleImgDataset_ordered
from models.MVCNN import MVCNN, SVCNN
import torchvision.models as models
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=4)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-lr-center", type=float, help="learning rate for center loss", default=0.1)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=1e-4)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="../mvcnn/modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="../mvcnn/modelnet40_images_new_12x/*/test")
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    localtime = time.strftime('%Y_%m_%d_%H', time.localtime(time.time()))
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()
    num_feature=1024
    # STAGE 1
    log_dir = 'justsoftmax_'+args.name+'_stage_1_'+localtime
    create_folder(log_dir)


    vgg=models.vgg11(pretrained=True)
    cnet = SVCNN(args.name,vgg, num_feature,nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name)
    if(torch.cuda.is_available()):
        cnet=cnet.cuda()
        print('use GPU to train ')
    else:
        print('don.t use gpu')

    #有centor loss 时
    softmax_loss=nn.CrossEntropyLoss()
    optimizer_model = optim.SGD(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)

    # #triplet_loss
    # soft_margin_triplet_loss=soft_margin_triplet(max_dist=2)
    # optimizer_model = optim.SGD(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    n_models_train = args.num_models*args.num_views

    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train,num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)


    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer_model,softmax_loss, 'svcnn', log_dir, num_views=1)
    trainer.train(20)

    # STAGE 2
    log_dir =  'justsoftmax_'+args.name+'_stage_2_'+localtime
    create_folder(log_dir)
    cnet_2 = MVCNN(args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views)
    del cnet

    optimizer = optim.SGD(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer_model,softmax_loss, 'mvcnn', log_dir, num_views=args.num_views)#loss_2,
    trainer.train(20)


