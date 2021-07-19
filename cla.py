import torch
import pandas as pd
from model import ClaModel
from random import shuffle
import os
import pandas as pd
import torch
from tqdm import tqdm
from model import HCL
import dgl
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import time


def get_node_batch(dataset_path, data_split, begin, BATCH_SIZE):
    inputs, srcs, dsts, labels,node_num = [], [], [], [],[]
    for i in range(begin, begin + BATCH_SIZE):
        data = pd.read_pickle(os.path.join(dataset_path, data_split[i]))
        for x, y in data.iterrows():
            node_num.append(len(y["ast"]))
            for src in y['src']:
                srcs.append((src - y['root'][0] + len(inputs)))
            for dst in y['dst']:
                dsts.append(dst - y['root'][0] + len(inputs))
            for emb in y['ast']:
                inputs.append(emb)
            labels.append(y['file_label']-1)
    return inputs, srcs, dsts, labels,node_num

def get_path_batch(pathdataset_path, data_split, begin, BATCH_SIZE):
    path_emb,path_num=[],[]
    path_label=[]
    num=[]
    for i in range(begin, begin + BATCH_SIZE):
        num.append(i)
        data = pd.read_pickle(os.path.join(pathdataset_path, data_split[i]))
        for x, y in data.iterrows():
            path_num.append(len(y['path']))
            for path in y['path']:
                path_emb.append(path)
            path_label.append(y['file_label']-1)
    return path_emb,path_num

def get_dataset(dataset_path, train_ratio, val_ratio, test_ratio):
    dataset_file_list = []
    for lists in os.listdir(dataset_path):
        dataset_file_list.append(lists)
    shuffle(dataset_file_list)
    train_split = int(len(dataset_file_list) * 0.1 * train_ratio)
    val_split = int(len(dataset_file_list) * 0.1 * val_ratio) + train_split
    train_split_data = dataset_file_list[0:train_split]
    val_split_data = dataset_file_list[train_split:val_split]
    test_split_data = dataset_file_list[val_split:]
    return train_split_data, val_split_data, test_split_data

def CrossEntropyLoss_label_smooth(outputs, targets,device, num_classes=104, epsilon=0.1):
    N = targets.size(0)
    if device=='cuda':
        smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1)).cuda()
    else:
        smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    targets = targets.data
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    log_prob = F.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--nodedataset_path', default='data/oj/node_emb',
                        help='node emb path')
    parser.add_argument('--pathdataset_path', default='data/oj/path_emb',
                        help='path emb path')
    parser.add_argument('--pre_model', default='model.pkl',
                        help='pre_model')
    args = parser.parse_args()
    nodedataset_path = args.nodedataset_path
    pathdataset_path = args.pathdataset_path
    pre_model_dict = torch.load(args.pre_model)

    train_ratio, val_ratio, test_ratio = 6, 2, 2
    train_split_data, val_split_data, test_split_data = get_dataset(nodedataset_path, train_ratio, val_ratio,
                                                                    test_ratio)

    begin = 0


    lr, BATCH_SIZE, EPOCH = 0.00001,32, 150
    USE_GPU = False
    in_feats,n_layer,n_head,drop_out,n_class=768,2,1,0.5,104




    # pre_model_dict = pre_model.state_dict()
    if USE_GPU:
        model = ClaModel(in_feats,n_layer,n_head,drop_out,n_class, device='cuda').cuda()
    else:
        model = ClaModel(in_feats, n_layer, n_head, drop_out, n_class, device='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = CrossEntropyLoss_label_smooth

    pbar = tqdm(range(EPOCH))

    best_model = model
    best_acc = 0.0
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []

    for epoch in pbar:
        pbar.set_description('epoch:%d  processing' % (epoch))
        i = 0
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        model.train()

        while (i + BATCH_SIZE) <= len(train_split_data):
            model.train()

            node_emb, srcs, dsts, labels,node_num = get_node_batch(nodedataset_path, train_split_data, i, BATCH_SIZE)
            path_emb,path_num=get_path_batch(pathdataset_path, train_split_data, i, BATCH_SIZE)
            i = i + BATCH_SIZE
            g = dgl.graph((srcs, dsts))
            g = dgl.add_self_loop(g)


            node_emb = torch.FloatTensor(node_emb)
            path_emb = torch.FloatTensor(path_emb)
            labels = torch.tensor(labels)


            if USE_GPU:
                g = g.to('cuda:0')
                node_emb = torch.FloatTensor(node_emb).cuda()
                path_emb = torch.FloatTensor(path_emb).cuda()
                labels = labels.cuda()

            model.zero_grad()
            logits=model(g,node_emb,path_emb,node_num,path_num)
            if  USE_GPU:
                loss = loss_function(logits, labels, 'cuda')
            else:
                loss = loss_function(logits, labels, 'cpu')

            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(logits.data, 1)
            total_acc += (predicted == labels).sum()

            total += len(labels)
            total_loss += loss.item() * len(labels)
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)

        i = 0
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        while (i + BATCH_SIZE) <= len(val_split_data):

            model.eval()
            node_emb, srcs, dsts, labels,node_num = get_node_batch(nodedataset_path, val_split_data, i, BATCH_SIZE)
            path_emb,path_num=get_path_batch(pathdataset_path,  val_split_data, i, BATCH_SIZE)
            i = i + BATCH_SIZE
            g = dgl.graph((srcs, dsts))
            g = dgl.add_self_loop(g)

            node_emb = torch.FloatTensor(node_emb)
            path_emb = torch.FloatTensor(path_emb)
            labels = torch.tensor(labels)


            if USE_GPU:
                g = g.to('cuda:0')
                node_emb = torch.FloatTensor(node_emb).cuda()
                path_emb = torch.FloatTensor(path_emb).cuda()
                labels = labels.cuda()

            logits=model(g,node_emb,path_emb,node_num,path_num)


            _, predicted = torch.max(logits.data, 1)
            total_acc += (predicted == labels).sum()

            total += len(labels)
            total_loss += loss.item() * len(labels)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc / total > best_acc:
            best_model = model
            torch.save(best_model, 'bestmodel.pkl')
            best_acc=total_acc / total
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCH, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while (i + BATCH_SIZE) <= len(test_split_data):

        model.eval()
        node_emb, srcs, dsts, labels, node_num = get_node_batch(nodedataset_path, test_split_data, i, BATCH_SIZE)
        path_emb, path_num = get_path_batch(pathdataset_path, test_split_data, i, BATCH_SIZE)
        i = i + BATCH_SIZE
        g = dgl.graph((srcs, dsts))
        g = dgl.add_self_loop(g)

        node_emb = torch.FloatTensor(node_emb)
        path_emb = torch.FloatTensor(path_emb)
        labels = torch.tensor(labels)

        if USE_GPU:
            g = g.to('cuda:0')
            node_emb = torch.FloatTensor(node_emb).cuda()
            path_emb = torch.FloatTensor(path_emb).cuda()
            labels = labels.cuda()

        logits = model(g, node_emb, path_emb, node_num, path_num)

        # calc training acc
        _, predicted = torch.max(logits.data, 1)
        total_acc += (predicted == labels).sum()
        total += len(labels)
        total_loss += loss.item() * len(labels)
    print("Testing results(Acc):", total_acc / total)
