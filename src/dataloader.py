# data.py

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image


# from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self):
        meta_file = open(osp.join(self.root_dir, 'train.json'), 'r')
        meta_json = json.load(meta_file)

        ans = {}
        val_t = []
        DatAns = []

        for i in range(len(meta_json)):
            set_to_item = {}
            items = meta_json[i]["items"]
            set_id1 = meta_json[i]["set_id"]
            item_id = [sub["item_id"] for sub in items]
            # mapping set id to item-ids
            set_to_item[set_id1] = item_id
            ans.update(set_to_item)

        f = open(osp.join(self.root_dir, "compatibility_train.txt"), "r")
        f = [line.split(' ') for line in f.readlines()]
        g = f
        for i in range(len(g)):
            (g[i][len(g[i]) - 1]) = (g[i][len(g[i]) - 1]).rstrip('\n')
            if g[i][0] == '0':
                g[i].pop(1)

        for i in range(len(g)):
            lst = []
            for j in range(len(g[i])):
                if j == 0:
                    binary_label = int(g[i][j])
                    lst.append(binary_label)

                else:

                    elem = g[i][j]
                    if elem[-2] == '_':
                        set_id = elem[0:-2]
                        idx = int(elem[-1])

                    elif elem[-3] == '_':
                        set_id = elem[0:-3]
                        idx = int(elem[-2:])

                    itemid = ans[set_id][idx - 1]
                    lst.append(itemid + '.jpg')
            val_t.append(lst)

        for i in range(len(val_t)):
            for j in range(1, len(val_t[i]) - 1):
                for k in range(j + 1, len(val_t[i])):
                    flst = []
                    flst.append(val_t[i][0])
                    flst.append(val_t[i][j])
                    flst.append(val_t[i][k])
                    DatAns.append(flst)

        files = os.listdir(self.image_dir)
        Xf = []
        y = []
        for i in range(len(DatAns)):
            Xf.append(DatAns[i][1:3])
            y.append(DatAns[i][0])
        X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1


# For category classification
class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_train[item][0])
        file_path2 = osp.join(self.image_dir, self.X_train[item][1])
        X = self.transform(Image.open(file_path))
        X2 = self.transform(Image.open(file_path2))
        return X-X2, self.y_train[item]


class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item][0])
        file_path2 = osp.join(self.image_dir, self.X_test[item][1])
        X = self.transform(Image.open(file_path))
        X2 = self.transform(Image.open(file_path2))
        return X-X2, self.y_test[item]


def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug == True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x == 'train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                   for x in ['train', 'test']}
    return dataloaders, classes, dataset_size


def text_gen():
    m = 0
    le = LabelEncoder()
    #g = le.fit_transform(y)
    B = []
    f = open(Config['root_path'] + "test_pairwise.txt", "r")
    a = [line.split() for line in f.readlines()]
    for i in range(len(a)):
        x = [ str(a[i][0]) + '.jpg', str(a[i][1])+'.jpg']
        #print(x)
        B.append(x)

    a = open(Config['root_path'] + "test_pairwise.txt", "r")
    b = open('Compatibility_test_category_hw.txt', 'w')

    f = [lines.split() for lines in a.readlines()]
    J = []
    for i in range(len(f)):
        J.append(f[i][0]+ ' ' + f[i][1])

    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()['test']

    size = int(np.floor(len(B) / Config['batch_size']))

    # model_copy_tensor = torch.load('./Results/ResNet.pth')
    check_point = torch.load('model.pth')
    model_copy = check_point['model']
    # model_copy = load_model('Build_model.hdf5')
    # model_copy.eval()

    for i in range(0, size * Config['batch_size'], Config['batch_size']):
        X = []
        #Y = []
        ans = []
        for j in range(Config['batch_size']):
            file_path = osp.join(osp.join(Config['root_path'], 'images'), B[i + j][0])
            file_path2 = osp.join(osp.join(Config['root_path'], 'images'), B[i + j][1])
            l = transforms(Image.open(file_path))
            l2 = transforms(Image.open(file_path2))
            X.append(l-l2)
            #Y.append(id_to_category[J[i + j]])


        with torch.no_grad():
            for inputs in X:
                # print(inputs.shape)
                inputs = inputs.to(device)
                outputs = model_copy(inputs[None, ...])
                _, pred = torch.max(outputs, 1)
                for p in pred:
                    ans.append(p)

        #             acc, loss1, ans = eval_model(model_copy_tensor, Y, criterion, device)
        #             # ans = (model_copy.predict(Y))

        #             for k in range(len(ans)):
        #                 ans1.append(np.argmax(ans[k]))
        #print(type(ans))
        #preds = ans.numpy()
        for p in range(Config['batch_size']):
            b.write(J[p + m] + '\t' + str(ans[p].numpy()) + '\n')
        m = m + Config['batch_size']
        if m == size * Config['batch_size']:
            break
    b.close()


########################################################################
# For Pairwise Compatibility Classification

# train_category.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt


# from utils import Config
# from model import model
# from data import get_dataloader


def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs.shape)
                    #inputs = inputs.view(-1,28*28)
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)

            if phase == 'test':
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model)

        # torch.save({'model':best_model_wts}, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        # print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))
        torch.save({'model': best_model_wts}, 'model.pth')
        print('Model saved at: {}'.format('model.pth'))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    plt.figure()
    plt.plot(np.arange(num_epochs), train_loss_list, label='Train')
    plt.plot(np.arange(num_epochs), val_loss_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('./Compatibility_new_loss.png', dpi=256)
    # plt.show()

    plt.figure()
    plt.plot(np.arange(num_epochs), train_acc_list, label='Train')
    plt.plot(np.arange(num_epochs), val_acc_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('./Compatibility_new_acc.png', dpi=256)
    # plt.show()


if __name__ == '__main__':



    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'],
                                                        num_workers=Config['num_workers'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'], weight_decay=0.0001)
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)
    text_gen()
