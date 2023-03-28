#######################################################################################################################
# This is the demo code for the submitted TGRS paper "A Scene Graph Encoding and Matching Network for UAV Visual Localization"
# Author: Dr. Ran Duan, LSGI, PolyU, HK
# Contct: rduan@polyu.edu.hk
#######################################################################################################################
from __future__ import division

import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv

def model_eval(model, device, ref_satellite_path, que_drone_path):
    test_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    ref_satellite_datasets = datasets.ImageFolder(ref_satellite_path, test_transform)
    que_drone_datasets = datasets.ImageFolder(que_drone_path, test_transform)
    ref_satellite_dataloaders = DataLoader(dataset=ref_satellite_datasets, batch_size=1, shuffle=False)
    que_drone_dataloaders = DataLoader(dataset=que_drone_datasets, batch_size=55, shuffle=False)
    # # 各类别名称
    ref_class_names = ref_satellite_dataloaders.dataset.classes

    model.eval()
    print('loading satellite images')
    ref_satellite_feats = None
    ref_satellite_labels = []
    with torch.no_grad():
        for ref_img, ref_label in tqdm(ref_satellite_dataloaders):
            input = Variable(ref_img.to(device).detach())
            output = model(input)
            d_flattened = output['descriptor_flatten']
            if ref_satellite_feats is not None:
                ref_satellite_feats = torch.cat((ref_satellite_feats, d_flattened.cpu()), 0)
            else:
                ref_satellite_feats = d_flattened.detach().clone().cpu()
            ref_satellite_labels.append(ref_class_names[ref_label])
    print('matching drone images with satellite images')
    top_1_counter = 0
    top_5_counter = 0
    total_counter = 0
    que_class_names = que_drone_dataloaders.dataset.classes
    with torch.no_grad():
        for que_img, labels in tqdm(que_drone_dataloaders):
            label_gt_list = []
            for label in labels:
                label_gt_list.append(que_class_names[label])
            input = Variable(que_img.to(device).detach())
            output = model(input)
            d_flattened = output['descriptor_flatten']
            que_feat = d_flattened.detach().clone().cpu()
            sim_scores = que_feat @ ref_satellite_feats.t()
            _, top_idx = torch.topk(sim_scores, 5)
            batch_id = 0
            for top_idx_by_batch in top_idx:
                total_counter += 1
                label_gt = label_gt_list[batch_id]
                recall_names = []
                for idx in top_idx_by_batch:
                    recall_names.append(ref_satellite_labels[idx])
                if label_gt == recall_names[0]:
                    top_1_counter += 1
                if label_gt in recall_names:
                    top_5_counter += 1
                batch_id += 1

    accuracy_score = {
        'R1': top_1_counter / total_counter * 100,
        'R5': top_5_counter / total_counter * 100
    }

    return accuracy_score

def load_ref_img(model, device, ref_satellite_path):
    print('encoding reference images...')
    test_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    ref_satellite_datasets = datasets.ImageFolder(ref_satellite_path, test_transform)
    ref_satellite_dataloaders = DataLoader(dataset=ref_satellite_datasets, batch_size=1, shuffle=False)
    # # 各类别名称
    ref_class_names = ref_satellite_dataloaders.dataset.classes
    model.eval()
    ref_satellite_feats = None
    ref_satellite_labels = []
    with torch.no_grad():
        for ref_img, ref_label in tqdm(ref_satellite_dataloaders):
            input = Variable(ref_img.to(device).detach())
            output = model(input)
            d_flattened = output['descriptor_flatten']
            if ref_satellite_feats is not None:
                ref_satellite_feats = torch.cat((ref_satellite_feats, d_flattened.cpu()), 0)
            else:
                ref_satellite_feats = d_flattened.detach().clone().cpu()
            ref_satellite_labels.append(ref_class_names[ref_label])

    ref_imgs = {
        'descriptors': ref_satellite_feats,
        'labels': ref_satellite_labels,
        'size': len(ref_satellite_labels)
    }
    return ref_imgs

def recall(model, device, ref_imgs, que_drone_path, visualize=False):
    print('retrieving...')
    test_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    que_drone_datasets = datasets.ImageFolder(que_drone_path, test_transform)
    que_drone_dataloaders = DataLoader(dataset=que_drone_datasets, batch_size=55, shuffle=False)
    ref_satellite_feats = ref_imgs['descriptors']
    ref_satellite_labels = ref_imgs['labels']
    top_1_counter = 0
    top_5_counter = 0
    total_counter = 0
    recall_results = []
    que_class_names = que_drone_dataloaders.dataset.classes
    with torch.no_grad():
        for que_img, labels in tqdm(que_drone_dataloaders):
            label_gt_list = []
            for label in labels:
                label_gt_list.append(que_class_names[label])
            input = Variable(que_img.to(device).detach())
            output = model(input)
            d_flattened = output['descriptor_flatten']
            que_feat = d_flattened.detach().clone().cpu()
            sim_scores = que_feat @ ref_satellite_feats.t()
            _, top_idx = torch.topk(sim_scores, 5)
            batch_id = 0
            for top_idx_by_batch in top_idx:
                total_counter += 1
                label_gt = label_gt_list[batch_id]
                recall_names = []
                for idx in top_idx_by_batch:
                    recall_names.append(ref_satellite_labels[idx])
                if label_gt == recall_names[0]:
                    top_1_counter += 1
                if label_gt in recall_names:
                    top_5_counter += 1
                recall_result = {'Query image label': label_gt, 'R@5': recall_names}
                recall_results.append(recall_result)
                batch_id += 1

    accuracy_score = {
        'R1': top_1_counter / total_counter * 100,
        'R5': top_5_counter / total_counter * 100
    }

    return recall_results, accuracy_score
