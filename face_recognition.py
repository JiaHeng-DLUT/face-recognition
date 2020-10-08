import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1

start = time.time()
device = torch.device('cuda:7,8' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device, select_largest=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
num_workers = 0

def collate_fn(x):
    # print(x)
    return x

# template
template_dataset = datasets.ImageFolder('./YouTubeFaces/frame_images_DB_224x224_highest')
# print(type(template_dataset))
# print(type(template_dataset.class_to_idx))
# print(template_dataset.class_to_idx)
# print(type(template_dataset.class_to_idx.items()))
template_dataset.idx_to_class = { i: c for c, i in template_dataset.class_to_idx.items() }
template_loader = DataLoader(template_dataset, collate_fn=collate_fn, num_workers=num_workers)
template_aligned = []
template_names = []
for _ in template_loader:
    for (x, y) in _:
        x_aligned = mtcnn(x)
        if x_aligned is not None:
            template_aligned.append(x_aligned)
            template_names.append(template_dataset.idx_to_class[y])
template_aligned = torch.stack(template_aligned).to(device)
template_embeddings = resnet(template_aligned).detach().cpu()
# test
test_dataset = datasets.ImageFolder('./YouTubeFaces/frame_images_DB_224x224')
test_dataset.idx_to_class = { i: c for c, i in test_dataset.class_to_idx.items() }
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
test_aligned = []
test_names = []
for _ in test_loader:
    for (x, y) in _:
        x_aligned = mtcnn(x)
        if x_aligned is not None:
            x_aligned = x_aligned
            test_aligned.append(x_aligned)
            test_names.append(test_dataset.idx_to_class[y])
test_aligned = torch.stack(test_aligned).to(device)
test_embeddings = resnet(test_aligned).detach().cpu()
# calculate similarity
(num_template, num_feature) = template_embeddings.shape
(num_test, num_feature) = test_embeddings.shape
dists = [[(e1 - e2).norm() for e2 in template_embeddings] for e1 in test_embeddings]
test_names_predict = [template_names[i] for i in np.argmin(np.array(dists), axis=1)]
accuracy = np.sum(np.char.equal(test_names, test_names_predict)) / len(test_names)
print(accuracy)
print(time.time() - start)
