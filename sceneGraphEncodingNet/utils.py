#######################################################################################################################
# This is the demo code for the submitted TGRS paper "A Scene Graph Encoding and Matching Network for UAV Visual Localization"
# Author: Dr. Ran Duan, LSGI, PolyU, HK
# Contct: rduan@polyu.edu.hk
#######################################################################################################################
from __future__ import division

import torch
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# ======================================================================================================================
# cnn feature recommender utils
# ======================================================================================================================
def tensor_norm(tensor, max_value=1):
    contrast = tensor.max() - tensor.min()
    if contrast > 0:
        tn = (tensor - tensor.min()) / contrast * max_value
        return tn
    else:
        return tensor

def image_norm(img):
    contrast = img.max() - img.min()
    if contrast > 0:
        img_normed = (img - img.min()) / contrast * 255
        img_normed = img_normed.astype(np.uint8)
    else:
        img_normed = img.astype(np.uint8)
    return img_normed

def array_norm(array, max_value=1):
    contrast = array.max() - array.min()
    if contrast > 0:
        array_normed = (array - array.min()) / contrast * max_value
    return array_normed

def scores_norm(scores):
    weights = [float(i) / sum(scores) for i in scores]
    return weights

def sort_list_descending(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs, reverse=True)]
    return z

def draw_local_global_matches(ref_img, query_img, mkpts0, mkpts1, margin=10):
    H0, W0, _ = ref_img.shape
    H1, W1, _ = query_img.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = ref_img
    out[:H1, W0 + margin:, :] = query_img
    c = [0, 255, 0]
    if mkpts0 is not None:
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            x0 = int(x0)
            y0 = int(y0)
            x1 = int(x1)
            y1 = int(y1)
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                     color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                       lineType=cv2.LINE_AA)
    return out

def draw_ref_query_image(ref_img, query_img, margin=10):
    H0, W0, _ = ref_img.shape
    H1, W1, _ = query_img.shape
    H, W = max(H0, H1), W0 + W1 + margin
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = ref_img
    out[:H1, W0 + margin:, :] = query_img
    return out

def draw_matches_within_bbox(img_A, img_B, keypoints0, keypoints1, bbox):
    p1s = []
    p2s = []
    dmatches = []
    if bbox[2] == 0 or bbox[3] == 0:
        for i, (x1, y1) in enumerate(keypoints0):
            if x1 > bbox[0] and x1 < bbox[0] + bbox[2] and y1 > bbox[1] and y1 < bbox[1] + bbox[3]:
                p1s.append(cv2.KeyPoint(np.float32(x1), np.float32(y1), 1))
                p2s.append(cv2.KeyPoint(np.float32(keypoints1[i][0]), np.float32(keypoints1[i][1]), 1))
                j = len(p1s) - 1
                dmatches.append(cv2.DMatch(j, j, 1))

    matched_images = cv2.drawMatches(img_A, p1s, img_B, p2s, dmatches, None)
    return matched_images

def draw_matches(img_A, img_B, keypoints0, keypoints1):
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
        p1s.append(cv2.KeyPoint(np.float32(x1), np.float32(y1), 1))
        p2s.append(cv2.KeyPoint(np.float32(keypoints1[i][0]), np.float32(keypoints1[i][1]), 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))

    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s,
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)

    return matched_images



