import os
import cv2
import shutil
import random
import datetime
import argparse
import numpy as np
import logging as logger
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import importlib
from PIL import Image
import json
from tqdm import tqdm
import clip
import albumentations as A
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import kornia as K



logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageTextDataset(Dataset):
    def __init__(self, test_file, data_size=448, num_class=2, data_type='train', color_space='RGB'):
        self.data_size = data_size
        self.data_list = []
        self.data_type = data_type
        self.color_space = color_space

        self.albu_pre_train = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type='jpeg', p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)

        
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)


        if self.color_space == 'YCbCr':
            self.RGB_to_YCbCr = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(),
                K.color.rgb_to_ycbcr, 
            ])
            self.imagenet_norm = transforms.Compose([
                transforms.Normalize((0.4702, 0.4766, 0.5128), (0.2524, 0.0616, 0.0613)),
            ])

        elif self.color_space == 'Lab':
            self.RGB_to_LAB = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(), 
                K.color.rgb_to_lab,
            ])
            self.imagenet_norm = transforms.Compose([
                transforms.Normalize((49.6360,  1.1967,  6.7421), (25.8020, 10.6061, 15.7528)),
            ])
        elif self.color_space == 'RGB':
            self.imagenet_norm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])


        test_file_buf = open(test_file)
        line = test_file_buf.readline().strip()


        while line:
            image_path, label, category_and_super = line.split(' ', 2)

            if len(category_and_super.split(' ')) > 2:
                category = category_and_super.split(' ')[0] + " " + category_and_super.split(' ')[1]
            else:
                category = category_and_super.split(' ')[0]

            supercategory = category_and_super.split(' ')[-1]

            label = int(label)
            textual_description = ''

            if num_class == 2:
                if label == 0:
                    textual_description = f'A real photo of a {category}, type of {supercategory}.'
                else:
                    textual_description = f'A synthetic photo of a {category}, type of {supercategory}.'

            elif num_class == 6:

                if label == 0:
                    textual_description = f'A real photo of a {category}, type of {supercategory}.'
                elif label == 1:
                    textual_description = f'A synthetic SD21 photo of a {category}, type of {supercategory}.'
                elif label == 2:
                    textual_description = f'A synthetic SDXL photo of a {category}, type of {supercategory}.'
                elif label == 3:
                    textual_description = f'A synthetic SD3 photo of a {category}, type of {supercategory}.'
                elif label == 4:
                    textual_description = f'A synthetic DALLE3 photo of a {category}, type of {supercategory}.'
                elif label == 5:
                    textual_description = f'A synthetic Midjourney photo of a {category}, type of {supercategory}.'


            self.data_list.append((image_path, int(label), textual_description))

            line = test_file_buf.readline().strip()

        logger.info(f"data list length: {len(self.data_list)}")
        
    def transform(self, x):

        if self.data_type == 'val':
            x = self.albu_pre_val(image=x)['image']
        elif self.data_type == 'train':
            x = self.albu_pre_train(image=x)['image']

        
        if self.color_space == 'YCbCr':
            x = self.RGB_to_YCbCr(x)
        elif self.color_space == 'Lab':
            x = self.RGB_to_LAB(x)

        x = self.imagenet_norm(x)

        return x


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.getitem(index, self.data_list)


    def getitem(self, index, data_list):

        image_path, label, textual_description = data_list[index]

        image = cv2.imread(image_path)

        if image is None:
            logger.info('Error Image: %s' % image_path)

        image = image[..., ::-1]
        image = self.transform(image)

        return image, image_path, label, textual_description



def train_one_epoch(data_loader, model, optimizer, cur_epoch, args, loss_meter):
    loss_meter.reset()
    batch_idx = 0
    running_loss = 0.0

    for (images, image_paths, labels, textual_description) in tqdm(data_loader):

        images = images.cuda()


        textual_description_2 = clip.tokenize([text for text in textual_description])
        textual_description_2 = textual_description_2.unsqueeze(0).expand(8, -1, -1).contiguous() 


        logits_per_image, _ = model(image_input= images, text_input=textual_description_2,  num_class= args.num_class, isTrain=True, isTest=False, text_option=args.text_option)

        ground_truth = torch.arange(len(images),dtype=torch.long).cuda()
        text_features = logits_per_image.t()


        img_loss = args.criterion_ce(logits_per_image,ground_truth)
        text_loss = args.criterion_ce(text_features,ground_truth)

        loss = (img_loss + text_loss)/2


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item() 

        loss_meter.update(loss.item(), images.shape[0])

        if batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f' % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            loss_meter.reset()


        batch_idx += 1



    epoch_loss = running_loss / len(data_loader)


    lr_value = optimizer.param_groups[0]['lr']
    logger.info(f'Epoch [{cur_epoch}/{args.epoches}], Epoch Average Loss: {epoch_loss:.4f}, lr: {lr_value}')

    return epoch_loss
    


def validation(data_loader, model, args):

    with torch.no_grad():

        batch_idx = 0
        running_loss = 0.0

        for (images, image_paths, labels, textual_description) in data_loader:

            images = images.cuda()


            textual_description_2 = clip.tokenize([text for text in textual_description])
            textual_description_2 = textual_description_2.unsqueeze(0).expand(8, -1, -1).contiguous() # torch.Size([8, 48, 77])

            
            logits_per_image, _ = model(image_input= images, text_input=textual_description_2,  num_class= args.num_class, isTrain=True, isTest=False, text_option=args.text_option)
            ground_truth = torch.arange(len(images),dtype=torch.long).cuda()

            text_features = logits_per_image.t()


            img_loss = args.criterion_ce(logits_per_image,ground_truth)
            text_loss = args.criterion_ce(text_features,ground_truth)
            

            loss = (img_loss + text_loss)/2

            running_loss += loss.item()


        val_loss = running_loss / len(data_loader)
        logger.info(f'Val Average Loss: {val_loss:.4f}')
    return val_loss
    



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss(epoch, train_losses, val_losses, plot_name):
    plt.figure(figsize=(10, 5))
    plt.plot(list(range(0,epoch + 1)), train_losses, label='Training Loss', marker='o')
    plt.plot(list(range(0,epoch + 1)), val_losses, label='Validation Loss', marker='o')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(args.weights, f'plot_name{epoch}.jpg'), format='jpg') 
    plt.close() 

    
def main(args):

    set_seed(48)

    if args.train_file == '':
        logger.info("Error: train file is Empty")


    model = getattr(importlib.import_module('model'), args.model)(img_encoder= args.img_encoder, num_class=args.num_class, text_option =args.text_option)
    model = torch.nn.DataParallel(model).cuda()

    train_data_loader = DataLoader(
        ImageTextDataset(args.train_file, data_size=args.data_size, num_class = args.num_class, data_type='train', color_space=args.color_space), args.batch_size, shuffle=True,
        num_workers=4, )

    logger.info(f"train_data_loader: {len(train_data_loader)}")

    val_data_loader = DataLoader(
        ImageTextDataset(args.val_file, data_size=args.data_size, num_class = args.num_class, data_type='val', color_space=args.color_space), args.batch_size, shuffle=False,
        num_workers=4, )

    logger.info(f"val_data_loader: {len(val_data_loader)}")

    args.criterion_ce = torch.nn.CrossEntropyLoss().cuda()


    epoch_start = -1
    if args.resume != '':
        pretrained = torch.load(args.resume)
        model.load_state_dict(pretrained)
        epoch_start = resume_epoch


    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(parameters, lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7)

    if epoch_start ==-1:
        epoch_start = 0

    loss_meter = AverageMeter()

    train_losses = []
    val_losses = []
    for epoch in range(epoch_start, args.epoches):
        model.train()
        train_loss = train_one_epoch(train_data_loader, model, optimizer, epoch, args, loss_meter)
        train_losses.append(train_loss)

        model.eval()
        val_loss = validation(val_data_loader, model, args)
        val_losses.append(val_loss)

        saved_name = 'Ep%03d.pt' % (epoch)

        torch.save(model.state_dict(), os.path.join(args.weights, saved_name))
        lr_schedule.step(val_loss)


        if epoch % 5 == 0 and epoch > 0:

            plot_loss(epoch, train_losses, val_losses, 'loss_plot_at_E')
    
    plot_loss(epoch, train_losses, val_losses, 'loss_plot_all_epochs')

    



    

if __name__ == '__main__':


    conf = argparse.ArgumentParser()
    conf.add_argument("--train_file", type=str, default='txt file that contain the img pathes along with the labels.')
    conf.add_argument("--val_file", type=str, default='txt file that contain the img pathes along with the labels.')
    conf.add_argument("--model", type=str, default='RSID')
    conf.add_argument("--num_class", type=int, default=2, help='The class number of training dataset.')
    conf.add_argument('--batch_size', type=int, default=48, help='The training batch size.')
    conf.add_argument('--data_size', type=int, default=448, help='The image size for training.')
    conf.add_argument("--img_encoder", type=str, default='RN50x64')

    conf.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    conf.add_argument("--weights", type=str, default='checkpoints', help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=100, help='The training epoches.')
    conf.add_argument("--resume", type=str, default='')
    conf.add_argument("--resume_epoch", type=int, default=0)
    conf.add_argument('--gpu', type=str, default='0,1,2,3', help='The gpu')
    conf.add_argument('--text_option', type=int, default=1)
    conf.add_argument('--color_space', type=str, default='RGB', help='The type of color spaces. RGB or Lab or YCbCr')

    args = conf.parse_args()



    os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))
    logger.info(str(min(os.cpu_count(), os.cpu_count())))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    date_now = datetime.datetime.now()
    date_now = '/Log_v%02d%02d%02d%02d' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
    args.time = date_now
    args.weights = args.weights + args.time
    if os.path.exists(args.weights):
        shutil.rmtree(args.weights)
    os.makedirs(args.weights, exist_ok=True)


    logger.info(args)
    main(args)


