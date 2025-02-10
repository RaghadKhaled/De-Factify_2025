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
import kornia as K
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')


class ImageTextDataset(Dataset):
    def __init__(self, test_file, data_size=448, model_type = 'RGB'):
        self.data_size = data_size
        self.data_list = []
        self.model_type = model_type



        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)

        if self.model_type == 'YCbCr':
            self.RGB_to_YCbCr = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(), 
                K.color.rgb_to_ycbcr,
            ])
            self.imagenet_norm = transforms.Compose([
                transforms.Normalize((0.4702, 0.4766, 0.5128), (0.2524, 0.0616, 0.0613)),
            ])

        elif self.model_type == 'Lab':
            self.RGB_to_LAB = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(), 
                K.color.rgb_to_lab,
            ])
            self.imagenet_norm = transforms.Compose([
                transforms.Normalize((49.6360,  1.1967,  6.7421), (25.8020, 10.6061, 15.7528)),
            ])
        elif self.model_type == 'RGB' or self.model_type == 'baseline':
            self.imagenet_norm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        test_file_buf = open(test_file)
        line = test_file_buf.readline().strip()

        while line:
            image_path, label = line.split(' ')
            # image_path, label, caption = line.split(' ', 2)


            self.data_list.append((image_path, int(label)))

            line = test_file_buf.readline().strip()


        logger.info(f"data list length: {len(self.data_list)}")

        

    def transform(self, x):

        x = self.albu_pre_val(image=x)['image']

        if self.model_type == 'YCbCr':
            x = self.RGB_to_YCbCr(x)
        elif self.model_type == 'Lab':
            x = self.RGB_to_LAB(x)

        x = self.imagenet_norm(x) 
        return x


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.getitem(index, self.data_list)


    def getitem(self, index, data_list):

        image_path, label = data_list[index]

        image = cv2.imread(image_path)

        if image is None:
            logger.info('Error Image: %s' % image_path)

        image = image[..., ::-1]
        image = self.transform(image)


        return image, image_path, label



def compare(feature, features_list, method, hard_topk, num_classes, args):

    feature = feature.unsqueeze(0)

    if method == 'cosine':
        if num_classes == 2:
            real_scores = 1 - (feature * features_list[0]).sum(dim=1)
            fake_scores = 1 - (feature * features_list[1]).sum(dim=1)

        elif num_classes == 6:
            real_scores = 1 - (feature * features_list[0]).sum(dim=1)
            SD21_scores = 1 - (feature * features_list[1]).sum(dim=1)
            SDXL_scores = 1 - (feature * features_list[2]).sum(dim=1)
            SD3_scores = 1 - (feature * features_list[3]).sum(dim=1)
            DALLE_scores = 1 - (feature * features_list[4]).sum(dim=1)
            Mid_scores = 1 - (feature * features_list[5]).sum(dim=1)

    elif method == 'l2':
        if num_classes == 2:
            real_scores = torch.norm( feature-features_list[0], p=2, dim=1 ) 
            fake_scores = torch.norm( feature-features_list[2], p=2, dim=1 ) 
        elif num_classes == 6:
            real_scores = torch.norm( feature-features_list[0], p=2, dim=1 ) 
            SD21_scores = torch.norm( feature-features_list[1], p=2, dim=1 ) 
            SDXL_scores = torch.norm( feature-features_list[2], p=2, dim=1 ) 
            SD3_scores = torch.norm( feature-features_list[3], p=2, dim=1 ) 
            Dalle_scores = torch.norm( feature-features_list[4], p=2, dim=1 ) 
            Mid_scores = torch.norm( feature-features_list[5], p=2, dim=1 ) 

    elif method == 'l1':
        if num_classes == 2:
            real_scores = torch.norm( feature-features_list[0], p=1, dim=1 ) 
            fake_scores = torch.norm( feature-features_list[1], p=1, dim=1 ) 
        elif num_classes == 6:
            real_scores = torch.norm( feature-features_list[0], p=1, dim=1 ) 
            SD21_scores = torch.norm( feature-features_list[1], p=1, dim=1 ) 
            SDXL_scores = torch.norm( feature-features_list[2], p=1, dim=1 ) 
            SD3_scores = torch.norm( feature-features_list[3], p=1, dim=1 ) 
            Dalle_scores = torch.norm( feature-features_list[4], p=1, dim=1 ) 
            Mid_scores = torch.norm( feature-features_list[5], p=1, dim=1 ) 
    else:
        assert False


    if num_classes == 2:
        scores = torch.cat([real_scores, fake_scores], dim=0)
        indicators = torch.cat([torch.zeros_like(real_scores), torch.ones_like(fake_scores)], dim=0)
    elif num_classes == 6:
        scores = torch.cat([real_scores, SD21_scores, SDXL_scores, SD3_scores, DALLE_scores, Mid_scores], dim=0)
        indicators = torch.cat([torch.zeros(len(real_scores)), torch.ones(len(SD21_scores)), torch.full(SDXL_scores.size(), 2), torch.full(SD3_scores.size(), 3), torch.full(DALLE_scores.size(), 4), torch.full(Mid_scores.size(), 5)], dim=0)
        


    sorted, indices = torch.sort(scores, descending=False)
    indicators=indicators[indices]


    pred_hard_list = [  indicators[0:topk].mode()[0] for topk in hard_topk      ]

    if args.task == 'B':
        pred_hard_list = pred_hard_list
    elif args.task == 'A':
        top1 = int(pred_hard_list[0].item())
        if top1 != 0:
            pred_hard_list = [torch.tensor(1.)]

    return pred_hard_list[0]




def test_NN(data_loader, model, args,features_list, method ,topk, num_classes, output_path, file_name):


    model.eval()


    true_labels = []
    predicted_labels = []
    image_features = []
    image_paths_list = []



    for (images, image_paths, labels) in tqdm(data_loader):
        images = images.cuda()
        image_paths = np.array(image_paths)

        image_paths_list.extend(image_paths)
        true_labels.extend(np.array(list(labels)))


        with torch.no_grad():
            img_features = model(image_input=  images)

            if method=='cosine':
                img_features = img_features/ torch.norm(img_features, dim=1, keepdim=True)
                image_features.append(img_features.cpu().detach().numpy())

            prediction = []
            for feat, img_path in zip(img_features, image_paths):
                pred_hard_list = compare(feat, features_list, method, topk, num_classes, args)
                top1 = int(pred_hard_list.item())
                prediction.append(top1)

        predicted_labels.extend(np.array(prediction))

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)


    concatenated_features = np.concatenate(image_features)
    concatenated_pathes = np.array(image_paths_list)

    predictions_list = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    os.makedirs(os.path.join(output_path, file_name), exist_ok=True)


    np.save(os.path.join(output_path, file_name, 'img_features'+'.npy'), concatenated_features)  
    np.save(os.path.join(output_path, file_name, 'img_pathes'+'.npy'), concatenated_pathes)  


    np.save(os.path.join(output_path, file_name, 'predictions'+'.npy'), predictions_list)  
    np.save(os.path.join(output_path,file_name, 'ground_truth'+'.npy'), true_labels)  



    accuracy = accuracy_score(true_labels, predictions_list)


    if args.task == 'A':

        r_acc = accuracy_score(true_labels[true_labels == 0], predictions_list[true_labels == 0])
        f_acc = accuracy_score(true_labels[true_labels == 1], predictions_list[true_labels == 1])


        f1 = f1_score(true_labels, predictions_list)
        return accuracy, r_acc, f_acc, f1
        
    elif args.task == 'B':

        r_acc = accuracy_score(true_labels[true_labels == 0], predictions_list[true_labels == 0])
        f_acc_sd21 = accuracy_score(true_labels[true_labels == 1], predictions_list[true_labels == 1])
        f_acc_sdxl = accuracy_score(true_labels[true_labels == 2], predictions_list[true_labels == 2])
        f_acc_sd3 = accuracy_score(true_labels[true_labels == 3], predictions_list[true_labels == 3])
        f_acc_dalle = accuracy_score(true_labels[true_labels == 4], predictions_list[true_labels == 4])
        f_acc_midjourny = accuracy_score(true_labels[true_labels == 5], predictions_list[true_labels == 5])


        f1 = f1_score(true_labels, predictions_list, average='weighted')
        return accuracy, f1,r_acc, f_acc_sd21, f_acc_sdxl, f_acc_sd3, f_acc_dalle, f_acc_midjourny





if __name__ == '__main__':

    conf = argparse.ArgumentParser()
    conf.add_argument("--test_files_pathes", type=str, default='txt file that contain the img pathes')
    conf.add_argument("--model", type=str, default='RSID')
    conf.add_argument("--num_class", type=int, default=2, help='The class number of testing dataset')
    conf.add_argument('--batch_size', type=int, default=48, help='The testing batch size.')
    conf.add_argument('--data_size', type=int, default=448, help='The image size for testing.')
    conf.add_argument("--ckps", type=str, default='/checkpoints/Ep.pt', help='The path for the pre-trained model')
    conf.add_argument("--out_dir", type=str, default='The directory path to store the results.')
    conf.add_argument("--out_file", type=str, default='The file name to store the results.')
    conf.add_argument("--img_encoder", type=str, default='RN50x64')
    conf.add_argument("--text_option", type=int, default=1)
    conf.add_argument("--task", type=str, default='A')
    conf.add_argument("--transformation", type=str, default='without')
    conf.add_argument("--method", type=str, default='RGB', help='The type of model. baseline, RGB, Lab or YCbCr')


    conf.add_argument("--topk", type=int, default=10, help='The K value on NN.')
    conf.add_argument('--template_size', type=int, default=None, help='')
    conf.add_argument('--method', type=str, default="cosine", help='The method use to measure the similarities between the query feature and the bank features.')
    conf.add_argument("--real_template", type=str, default="Features_bank/real_features.npy")
    conf.add_argument("--fake_template", type=str, default="Features_bank/fake_features.npy")
    conf.add_argument("--SD21_template", type=str, default="Features_bank/SD21_features.npy")
    conf.add_argument("--SDXL_template", type=str, default="Features_bank/SDXL_features.npy")
    conf.add_argument("--SD3_template", type=str, default="Features_bank/SD3_features.npy")
    conf.add_argument("--Dalle_template", type=str, default="Features_bank/Dalle_features.npy")
    conf.add_argument("--Mid_template", type=str, default="Features_bank/Mid_features.npy")


    args = conf.parse_args()
    os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))
    logger.info(str(min(os.cpu_count(), os.cpu_count())))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(torch.cuda.current_device())

    logger.info(args)


    test_files = os.listdir(args.test_files_pathes) 


    Acc_list = {}
    F1_list = {}
    r_acc_list = {}
    f_acc_list = {}


    model = getattr(importlib.import_module('model_NN'), args.model)(img_encoder= args.img_encoder, num_class=args.num_class, text_option=args.text_option)
    model = torch.nn.DataParallel(model).cuda()



    if args.model != 'baseline':
        pretrained = torch.load(args.ckps) 
        model.load_state_dict(pretrained, strict=False)
        logger.info(f'ckps: {args.ckps}')

    output_path = os.path.join(args.out_dir, args.out_file)

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)


    if args.num_class == 2:

        real_features = torch.from_numpy(np.load(args.real_template)).cuda()
        fake_features =  torch.from_numpy(np.load(args.fake_template)).cuda()

    elif args.num_class == 6:
        
        real_features = torch.from_numpy(np.load(args.real_template)).cuda()
        SD21_features = torch.from_numpy(np.load(args.SD21_template)).cuda()
        SDXL_features = torch.from_numpy(np.load(args.SDXL_template)).cuda()
        SD3_features = torch.from_numpy(np.load(args.SD3_template)).cuda()
        DALLE_features = torch.from_numpy(np.load(args.Dalle_template)).cuda()
        Mid_features = torch.from_numpy(np.load(args.Mid_template)).cuda()



    topk = [args.topk] 



    if args.template_size is not None:
        if args.num_class == 2:
            real_features = real_features[0:args.template_size]
            fake_features = fake_features[0:args.template_size]
        elif args.num_class == 6:
            real_features = real_features[0:args.template_size]
            SD21_features = SD21_features[0:args.template_size]
            SDXL_features = SDXL_features[0:args.template_size]
            SD3_features = SD3_features[0:args.template_size]
            DALLE_features = DALLE_features[0:args.template_size]
            Mid_features = Mid_features[0:args.template_size]

    features_list = []

    if args.num_class == 2:
        features_list.append(real_features)
        features_list.append(fake_features)
    elif args.num_class == 6:
        features_list.append(real_features)
        features_list.append(SD21_features)
        features_list.append(SDXL_features)
        features_list.append(SD3_features)
        features_list.append(DALLE_features)
        features_list.append(Mid_features)

    output_path = os.path.join(args.out_dir, args.out_file)

    os.makedirs(output_path, exist_ok=True)



    for idx, test_file in enumerate(test_files):

        file_name = str(test_file)


        if args.transformation == 'without':

            if args.task == 'A':
                if test_file != 'TaskB.txt':
                    continue
            elif args.task == 'B':
                if test_file != 'TaskB.txt':
                    continue
            test_file_path = os.path.join(args.test_files_pathes, test_file)


        elif args.transformation == 'with':
            if args.task == 'A':
                test_file_path = os.path.join(args.test_files_pathes, test_file, 'TaskA.txt')
            elif args.task == 'B':
                test_file_path = os.path.join(args.test_files_pathes, test_file, 'TaskB.txt')




        test_data_loader = DataLoader(
        ImageTextDataset(test_file_path, data_size=args.data_size,  model_type = args.method_model), args.batch_size, shuffle=False,
        num_workers=4, )

        if args.task == 'A':
            accuracy, r_acc, f_acc, f1 = test_NN(test_data_loader, model, args, features_list, args.method ,topk, args.num_class,  output_path, file_name)
        elif args.task == 'B':
            accuracy, f1,r_acc, f_acc_sd21, f_acc_sdxl, f_acc_sd3, f_acc_dalle, f_acc_midjourny = test_NN(test_data_loader, model, args, features_list, args.method ,topk, args.num_class,  output_path, file_name) 
            f_acc = [f_acc_sd21, f_acc_sdxl, f_acc_sd3, f_acc_dalle, f_acc_midjourny]

        Acc_list[file_name] = accuracy
        F1_list[file_name] = f1 
        r_acc_list[file_name] = r_acc
        f_acc_list[file_name] = f_acc




    with open(f'{output_path}/Acc.json', 'w') as f:
        json.dump(Acc_list,  f, indent=4)

    with open(f'{output_path}/F1_score.json', 'w') as f:
        json.dump(F1_list,  f, indent=4)
    
    with open(f'{output_path}/r_acc.json', 'w') as f:
        json.dump(r_acc_list,  f, indent=4)


    with open(f'{output_path}/f_acc.json', 'w') as f:
        json.dump(f_acc_list,  f, indent=4)


