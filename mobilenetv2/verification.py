#! /usr/bin/env python3

# activate_this_file = "/ihome/xjiang/abj40/.virtualenvs/pytorch/bin/activate_this.py"
# exec(open(activate_this_file).read())
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
from collections import OrderedDict
from collections import defaultdict
import pickle


def save_model(model, epoch, task='Classification'):
        path = task+'/'+str(epoch)+'.pt'
        torch.save(model.state_dict(), path)


class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label

class imageFolderPath(torchvision.datasets.ImageFolder):
    #Inspired from https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(imageFolderPath, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#Parse the given directory to accumulate all the images
def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n



# img_list, label_list, class_n = parse_data('medium')
# train_data_item, train_data_label = trainset.__getitem__(0)
# dataloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=False)
# 
# imageFolder_dataset = torchvision.datasets.ImageFolder(root='medium/', transform=torchvision.transforms.ToTensor())
# imageFolder_dataloader = DataLoader(imageFolder_dataset, batch_size=10, shuffle=True, num_workers=1)

class BottleneckResBlock(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, expansion=False, stride=1):
        super(BottleneckResBlock, self).__init__()
        self.stride = stride
        channel_size = self.in_channel_size = in_channel_size
        self.out_channel_size = out_channel_size

        if expansion:
            self.block = nn.Sequential(
                #expansion layer
                nn.Conv2d(in_channels=channel_size, out_channels=expansion*channel_size, 
                                             kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=expansion*channel_size),
                nn.ReLU6(inplace=True),

                #"Depthwise" Convolution
                nn.Conv2d(in_channels=expansion*channel_size, out_channels=expansion*channel_size, 
                                             kernel_size=3, stride=stride, padding=1, groups=expansion*channel_size, bias=False),
                nn.BatchNorm2d(num_features=expansion*channel_size),
                nn.ReLU6(inplace=True),

                #"Projection" Layer
                nn.Conv2d(in_channels=expansion*channel_size, out_channels=out_channel_size, 
                                             kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channel_size),
                )
        else:
            self.block = nn.Sequential(

                #"Depthwise" Convolution
                nn.Conv2d(in_channels=channel_size, out_channels=channel_size, 
                                             kernel_size=3, stride=stride, padding=1, groups=channel_size, bias=False),
                nn.BatchNorm2d(num_features=channel_size),
                nn.ReLU6(inplace=True),

                #"Projection" Layer
                nn.Conv2d(in_channels=channel_size, out_channels=out_channel_size, 
                                             kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channel_size),
                )
        # self.logit_non_linear = nn.ReLU(inplace=True)

    def forward(self, x):
        if(self.stride == 1 and self.in_channel_size == self.out_channel_size):
            output = x + self.block(x)
        else:
            output = self.block(x)
        # output = x
        # output = self.block(output)
        # output = self.logit_non_linear(output + x)
        return output


class Network_MNV2(nn.Module):
    """docstring for Network_MNV2"""
    def __init__(self, num_classes):
        super(Network_MNV2, self).__init__()
        # self.arg = arg 
        self.num_classes = num_classes
        br_block = BottleneckResBlock
        in_channels = 3
        self.out_channels = 1280
        self.layers = []
        BottleneckResBlock_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        #first layer

        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
            ))
        in_channel_size = 32

        #adding middle layers with Bottleneck Residual Block
        for t, c, n, s in BottleneckResBlock_config:
            for i in range(n):
                if i!=0:
                    stride=1
                    self.layers.append(BottleneckResBlock(in_channel_size=in_channel_size, out_channel_size=c, stride=stride, expansion=t))
                    in_channel_size = c
                else:
                    stride=s
                    self.layers.append(BottleneckResBlock(in_channel_size=in_channel_size, out_channel_size=c, stride=stride, expansion=t))
                    in_channel_size = c

                

        #adding last layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channel_size, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU6(inplace=True)
            ))

        self.layers = nn.Sequential(*self.layers)

        #add classifier network
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.AvgPool2d(kernel_size=8),
            nn.Conv2d(self.out_channels,  self.num_classes, 1, bias=True),
            # nn.Linear(self.out_channels, self.num_classes),
        )

    def forward(self, x):
        output = x
        output = self.layers(output)
        # output = output.view(-1,1280)
        # output.mean(3).mean(2)
        output = self.classifier(output)
        output = output.view(-1, self.num_classes)

         # Create the feature embedding for the Center Loss
        # closs_output = self.linear_closs(output)
        # closs_output = self.relu_closs(closs_output)

        # output = self.classifier(output)

        # return closs_output, label_output


        return output




def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)



def train(model, data_loader, val_loader, test_loader, train_classes, task='Classification'):
    model.train()
    model.to(device)
    # print('class 2',test_classes[1])

    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                # val_loss, val_acc = val_classify(model, val_loader)
                # train_loss, train_acc = val_classify(model, data_loader)
                # test_classify(model, test_loader, epoch+1, train_classes)
                # exit()
                # print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  # format(train_loss, train_acc, val_loss, val_acc))
                # path = 'asd.pt'
                # torch.save(model.state_dict(), path)
                # save_model(model, epoch+1, task)
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if task == 'Classification':
            val_loss, val_acc = val_classify(model, val_loader)
            train_loss, train_acc = val_classify(model, data_loader)
            test_classify(model, test_loader, epoch+1, train_classes)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
            save_model(model, epoch+1, task)

        else:
            test_verify(model, test_loader)

# get txt data
def import_data(filename):
    with open(filename, 'rt') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)

        lineData = list()
        cols = next(reader)

        for col in cols:
            # create a list in lineData for each column of data
            lineData.append(list())

        for line in reader:
            for i in range(0, len(lineData)):
                lineData[i].append(line[i])
                # Copy the data from the line into the correct columns.
                # lineData[i].append(line[i])

        data = dict()

        for i in range(0, len(cols)):
            # Create each key in the dict with the data in its column.
            data[cols[i]] = lineData[i]

        return data


def val_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total

def test_classify(model, test_loader, epoch, train_classes):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    final=np.empty(shape=(0,0))
    filename = str(epoch)+"test_results.csv"

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        final_labels = [train_classes[label_id] for label_id in pred_labels.cpu().numpy()]
        for i in final_labels:
            final = np.append(final,  i)
        del feats
        del labels
    pd.DataFrame(data=final).to_csv(filename,header=False,  index=True)
    model.train()
    return 


def test_verify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    filename = 'image_embeding.csv'
    final=np.empty(shape=(0,0))
    embed_dict = dict()
    # test_verify_input = import_data('./data/test_trials_verification_student.txt')
    # test_verify_input = import_data('./data/mini_verify.txt')


    for batch_num, (feats, labels, paths) in enumerate(test_loader):
        feats = feats.to(device)#, labels.to(device)#, paths.to(device)
        outputs = model(feats)
        # print(outputs.cpu().detach().numpy())
        for output, path in zip(outputs.detach(), paths):
            # print(path.split('/'))
            img_id = path.split('/')[2]
            img_id = img_id.replace("VID_", "v_")
            # embed_dict[img] = output
            if img_id in embed_dict:
                embed_dict[img_id].append(output)
            else:
                embed_dict[img_id] = [output]
            # final = np.append(final, [output.cpu().detach().numpy(), path.split('/')[3]])
        # final = np.append(final, [outputs.cpu().detach().numpy(), [path.split('/')[3] for path in paths]])
        # paths = [path.split('/')[3] for path in paths]

        del feats
        del labels
        del paths

        # if batch_num % 5 == 4:
            # break

    # print(embed_dict)
    # for i in range(len(test_verify_input['1'])):
        # print(test_verify_input['1'][i])
        # print(test_verify_input['2'][i])
        # x1 = embed_dict[test_verify_input['1'][i]].reshape(1, 2300)
        # x2 = embed_dict[test_verify_input['2'][i]].reshape(1, 2300)
        # similarity = F.cosine_similarity(x1, x2)
        # final = np.append(final, similarity.cpu().numpy())
        # similarity = F.cosine_similarity(embed_dict[test_verify_input['1'][i]][0], embed_dict[test_verify_input['2'][i]][0])
        # print(similarity)


    # with open('./data/test_trials_verification_student.txt', 'r') as istr:
    #   with open('./data/test_trials_verification_student_output.txt', 'w') as ostr:
    #       for i, line in enumerate(istr):
    #           if i!=0:
    #               line = line.rstrip('\n') +" "+ str(final[i-1])
    #               print(line, file=ostr)
    for k in embed_dict.keys():
        embed_dict[k] = np.vstack(list(map(lambda x: x.cpu().detach().numpy(), embed_dict[k])))
        print(embed_dict[k].shape)
    # print(embed_dict)
    f = open("/afs/cs/projects/kovashka/maher/vol2/dense/activitynet/activitynet_mnet_fc7_stride_64_frame.pkl","wb")
    pickle.dump(embed_dict,f,protocol=2)
    f.close()
    return embed_dict

# train_dataset = torchvision.datasets.ImageFolder(root='/ihome/xjiang/abj40/dl/data/train_data/medium/', 
                                                 # transform=torchvision.transforms.ToTensor())
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, 
                                               # shuffle=True, num_workers=4)
# print('class 1',train_dataset.classes[1])
# exit()
# train_large_dataset = torchvision.datasets.ImageFolder(root='data/train_data/large/', 
                                                 # transform=torchvision.transforms.ToTensor())
# train_large_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, 
                                               # shuffle=True, num_workers=8)

# val_dataset = torchvision.datasets.ImageFolder(root='/ihome/xjiang/abj40/dl/data/validation_classification/medium/', 
                                               # transform=torchvision.transforms.ToTensor())
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, 
                                             # shuffle=False, num_workers=4)

# test_dataset = torchvision.datasets.ImageFolder(root='/ihome/xjiang/abj40/dl/data/test_classification/', 
                                               # transform=torchvision.transforms.ToTensor())
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, 
                                             # shuffle=False, num_workers=4)

# test_verify_input = import_data('./data/test_trials_verification_student.txt')
# print(test_verify_input['1'][0])
# print(test_verify_input['2'][0])
# exit()


test_verification = imageFolderPath(root='./yt_resized/',
                                                transform=torchvision.transforms.ToTensor())
test_verify_dataloader = torch.utils.data.DataLoader(test_verification, batch_size=128,
                                                shuffle=False, num_workers=4)

# numEpochs = 20
# num_feats = 3

# learningRate = 1e-3
# weightDecay = 1e-4

# hidden_sizes = [32, 64]
num_classes = 101

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:2')

mnetv2 = Network_MNV2(num_classes)
state_dict = torch.load('./Classification/20.pt')
mnetv2.load_state_dict(state_dict)
mnetv2 = mnetv2.to(device)

test_verify(mnetv2, test_verify_dataloader)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(network.parameters(), lr=learningRate, weight_decay=weightDecay)
# print(summary(network))