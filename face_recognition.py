import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

high_dataset = datasets.ImageFolder('./YouTubeFaces/frame_images_DB_224x224_highest')
low_dataset = datasets.ImageFolder('./YouTubeFaces/frame_images_DB_224x224_lowest')
whole_dataset = datasets.ImageFolder('./YouTubeFaces/frame_images_DB_224x224')
high_dataset.idx_to_class = {i:c for c, i in high_dataset.class_to_idx.items()}
low_dataset.idx_to_class = {i:c for c, i in low_dataset.class_to_idx.items()}
whole_dataset.idx_to_class = {i:c for c, i in whole_dataset.class_to_idx.items()}
# whole_dataset_size = len(whole_dataset)
# test_dataset_size = int(0.5 * whole_dataset_size)
# [test_dataset, _] = torch.utils.data.random_split(whole_dataset, [test_dataset_size, whole_dataset_size - test_dataset_size])
test_dataset = whole_dataset

high_loader = DataLoader(high_dataset, collate_fn=collate_fn, num_workers=0)
low_loader = DataLoader(low_dataset, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_dataset, collate_fn=collate_fn, num_workers=0)


high_aligned = []
high_names = []
for x, y in high_loader:
    print(x)
    x_aligned = mtcnn(x)
    if x_aligned is not None:
        x_aligned = x_aligned[0]
        high_aligned.append(x_aligned)
        high_names.append(high_dataset.idx_to_class[y])
high_aligned = torch.stack(high_aligned).to(device)
high_embeddings = resnet(high_aligned).detach().cpu()


low_aligned = []
low_names = []
for x, y in low_loader:
    print(x)
    x_aligned = mtcnn(x)
    if x_aligned is not None:
        x_aligned = x_aligned[0]
        low_aligned.append(x_aligned)
        low_names.append(low_dataset.idx_to_class[y])
low_aligned = torch.stack(low_aligned).to(device)
low_embeddings = resnet(low_aligned).detach().cpu()

test_aligned = []
test_names = []
for x, y in test_loader:
    print(x)
    x_aligned = mtcnn(x)
    if x_aligned is not None:
        x_aligned = x_aligned[0]
        test_aligned.append(x_aligned)
        test_names.append(test_dataset.idx_to_class[y])
test_aligned = torch.stack(test_aligned).to(device)
test_embeddings = resnet(test_aligned).detach().cpu()

high_dists = [[(e1 - e2).norm() for e2 in high_embeddings] for e1 in test_embeddings]
high_cnt = 0
for i in range(len(test_embeddings)):
    min_diff = 1e9
    for j in range(len(high_embeddings)):
        diff = (test_embeddings[i] - high_embeddings[j]).norm()
        if (diff < min_diff):
            min_diff = diff
            recog = j
    if test_names[i] == high_names[recog]:
        high_cnt += 1;

low_dists = [[(e1 - e2).norm() for e2 in low_embeddings] for e1 in test_embeddings]
low_cnt = 0
for i in range(len(test_embeddings)):
    min_diff = 1e9
    for j in range(len(low_embeddings)):
        diff = (test_embeddings[i] - low_embeddings[j]).norm()
        if (diff < min_diff):
            min_diff = diff
            recog = j
    if test_names[i] == low_names[recog]:
        low_cnt += 1;
        
print(high_cnt/len(test_embeddings))
print(low_cnt/len(test_embeddings))