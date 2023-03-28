#######################################################################################################################
# This is the demo code for the submitted TGRS paper "A Scene Graph Encoding and Matching Network for UAV Visual Localization"
# Author: Dr. Ran Duan, LSGI, PolyU, HK
# Contct: rduan@polyu.edu.hk
#######################################################################################################################
import torch.nn as nn
from torchvision import models
from sceneGraphEncodingNet.model_eval import *
from sceneGraphEncodingNet.nets import CSMG, JointNet
import warnings
import csv
warnings.filterwarnings("ignore")
device = torch.device("cuda")

# set dataset path
dataset_dir = './data/University-Release'
test_ref_satellite_path = dataset_dir + '/test/gallery_satellite'
test_que_drone_path = dataset_dir + '/test/query_drone'

# Init scene graph
NUM_CLUSTERS = 4
our_net = CSMG(512, NUM_CLUSTERS)
sceneGraphEncoder = JointNet(None, our_net)

PRE_TRAINED_PATH = './weights/best_result.pth'
model_weights = torch.load(PRE_TRAINED_PATH)
sceneGraphEncoder.load_state_dict(model_weights['model_state_dict'])

# Init backbone
backbone = models.vgg16(pretrained=True)
layers = list(backbone.features.children())[:-8]  # output c, h, w = 512, 28, 28
for l in layers:
    for p in l.parameters():
        p.requires_grad = False
backbone = nn.Sequential(*layers)
# Load backbone
sceneGraphEncoder.backbone = backbone
sceneGraphEncoder = sceneGraphEncoder.to(device)

if torch.cuda.device_count() > 1:
    sceneGraphEncoder.module = nn.DataParallel(sceneGraphEncoder.module)

# Encode reference images
ref_imgs = load_ref_img(sceneGraphEncoder, device, test_ref_satellite_path)

# Recall test
visualize = True
recall_results, accuracy_score = recall(sceneGraphEncoder, device, ref_imgs, test_que_drone_path, visualize)

# Check results
print('Overall recall accuracy')
print(accuracy_score)

print('Sampled recall result')
print(recall_results[0])
print(recall_results[55])
print(recall_results[110])

# save results
with open('recall_results.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
    write.writerow(['Query image label', 'R@1', 'R@2', 'R@3', 'R@4', 'R@5'])
    for result in recall_results:
        write.writerow([result['Query image label'], result['R@5'][0], result['R@5'][1], result['R@5'][2], result['R@5'][3], result['R@5'][4]])