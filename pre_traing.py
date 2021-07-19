from random import  shuffle
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

def prepar_tri_loss(deep_vex,embed,nodeNum):
    tri_p = []
    tri_n = []
    tri_a = []
    node = []
    label = []
    nodeIndex = nodeNum

    for k in range(len(embed)):
        get_tri_p = False
        get_tri_n = False
        node.append(k)
        if k>=len(embed):
            print(deep_vex)
            print(k)
        label.append(deep_vex[k])
        tri_a.append(embed[k])
        pre = (k - 1 + nodeIndex) % nodeIndex

        while not get_tri_p or not get_tri_n:
            if k == 0:
                tri_p.append(embed[k])
                get_tri_p = True
                if not get_tri_n:
                    tri_n.append(embed[pre])
                    get_tri_n = True
            else:
                if deep_vex[pre] == deep_vex[k] and not get_tri_p:
                    tri_p.append(embed[pre])
                    get_tri_p = True
                elif not get_tri_n:
                    tri_n.append(embed[pre])
                    get_tri_n = True
            pre = (pre - 1 + nodeIndex) % nodeIndex
    return tri_p,tri_n,tri_a,node,label

def draw_tsne(input,label,img_name):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(input.detach().cpu().numpy())
    np.save("X_tsne.npy", X_tsne)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)

    if x_max.all() == 0:
        print(img_name)
        print(input)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 8))
    label_tmp=[]
    for i in range(X_norm.shape[0]):
        label_tmp.append(label[i])
    plt.scatter(X_norm[:, 0], X_norm[:, 1], marker = 'o', c = label_tmp, cmap = 'coolwarm')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./'+img_name)
    plt.close('all')

def get_max_deep(dataset_path):
    max_deep=0
    for lists in tqdm(os.listdir(dataset_path)):
        sub_path = os.path.join(dataset_path, lists)
        if os.path.isfile(sub_path):
            data = pd.read_pickle(sub_path)
            for index,item in data.iterrows():
                deep_list=item['deep_label']
                if max_deep<max(deep_list):
                    max_deep=max(deep_list)
    return max_deep


def get_batch(dataset_path,data_split,begin,BATCH_SIZE):
    inputs,srcs,dsts,labels=[],[],[],[]
    for i in range(begin,begin+BATCH_SIZE):
        data=pd.read_pickle(os.path.join(dataset_path,data_split[i]))
        for x,y in data.iterrows():
            for src in y['src']:
                srcs.append((src-y['root'][0]+len(inputs)))
            for dst in y['dst']:
                dsts.append(dst-y['root'][0]+len(inputs))
            for emb in y['ast']:
                inputs.append(emb)
            for label in y['deep_label']:
                labels.append(label)

    return inputs,srcs,dsts,labels


def get_dataset(dataset_path,train_ratio,val_ratio,test_ratio):
    dataset_file_list=[]
    for lists in os.listdir(dataset_path):
        dataset_file_list.append(lists)
    shuffle(dataset_file_list)
    train_split=int(len(dataset_file_list)*0.1*train_ratio)
    val_split = int(len(dataset_file_list) * 0.1 * val_ratio)+train_split
    train_split_data=dataset_file_list[0:train_split]
    val_split_data=dataset_file_list[train_split:val_split]
    test_split_data=dataset_file_list[val_split:]
    return train_split_data,val_split_data,test_split_data

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--dataset_nodeemb',default='data/oj/node_emb',
                        help='The path of the data set that has been encoded')
    args = parser.parse_args()

    dataset_path=args.dataset_nodeemb
    train_ratio,val_ratio,test_ratio=6,2,2
    train_split_data,val_split_data,test_split_data=get_dataset(dataset_path,train_ratio,val_ratio,test_ratio)

    begin=0

    lw,lr,BATCH_SIZE,EPOCH=0.5,0.00001,32,5
    USE_GPU=False
    n_layers,in_features=4,768
    d_k,d_v,hidden_size=128,128,1024
    dropout=0.5
    num_class=76
    if USE_GPU:
        model=HCL(n_layers,in_features,d_k,d_v,hidden_size,dropout,num_class,'cuda').cuda()
    else:
        model = HCL(n_layers, in_features, d_k, d_v, hidden_size, dropout, num_class, 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(EPOCH))

    best_model = model
    best_acc=0.0
    for epoch in pbar:
        pbar.set_description('epoch:%d  processing' % (epoch))
        i = 0
        model.train()
        while (i + BATCH_SIZE) <= len(train_split_data):
            inputs,srcs,dsts,labels=get_batch(dataset_path,train_split_data,i,BATCH_SIZE)
            # print(len(embed), len(deep_label), len(src), len(dst))
            i = i + BATCH_SIZE
            g = dgl.graph((srcs, dsts))
            g = dgl.add_self_loop(g)
            if USE_GPU:
                g = g.to('cuda:0')
            # visual(g)
            tri_p, tri_n, tri_a, node, label = prepar_tri_loss(labels, inputs, len(labels))

            '''
            数据格式转换
            '''
            inputs = torch.FloatTensor(inputs)
            tri_a = torch.FloatTensor(tri_a )
            tri_p = torch.FloatTensor(tri_p)
            tri_n = torch.FloatTensor(tri_n )
            label = torch.tensor(label)
            if USE_GPU:
                inputs = torch.FloatTensor(inputs).cuda()
                tri_a = torch.FloatTensor(tri_a).cuda()
                tri_p = torch.FloatTensor(tri_p).cuda()
                tri_n = torch.FloatTensor(tri_n).cuda()
                label=label.cuda()

            model.zero_grad()

            a_logits,conv= model(g, tri_a)
            p_logits,conv= model(g, tri_p)
            n_logits,conv = model(g, tri_n)
            logits,conv= model(g, inputs)


            loss1 = F.cross_entropy(logits, label)
            loss2 = F.triplet_margin_loss(a_logits, p_logits, n_logits, reduction='mean', margin=0.5)
            print('null_loss:{}\ntriplet_loss:{}'.format(loss1, loss2))
            loss = (1-lw)*loss1 + lw * loss2

            loss.backward()
            optimizer.step()
            print('Epoch %d | Loss: %.4f\n' % (epoch, loss.item()))
            if epoch == EPOCH-1:
                draw_tsne(conv, label, 'fe.jpg')
                torch.save(model.state_dict(), 'model_train_par.pkl')

        total_acc = 0.0
        model.eval()
        while (i + BATCH_SIZE) <= len(val_split_data):
            inputs,srcs,dsts,label=get_batch(dataset_path,val_split_data,i,BATCH_SIZE)
            total=len(label)
            # print(len(embed), len(deep_label), len(src), len(dst))
            i = i + BATCH_SIZE
            g = dgl.graph((srcs, dsts))
            g = dgl.add_self_loop(g)
            if USE_GPU:
                g = g.to('cuda:0')

            inputs = torch.FloatTensor(inputs)
            label = torch.tensor(label)
            if USE_GPU:
                inputs = torch.FloatTensor(inputs).cuda()
                label=label.cuda()

            logits,conv= model(g, inputs)
            _, predicted = torch.max(logits.data, 1)
            total_acc += (predicted == label).sum()
            if total_acc>best_acc:
                best_model=model
                torch.save(model.state_dict(), 'model.pkl')


    model.eval()
    i=0
    while (i + BATCH_SIZE) <= len(test_split_data):
        inputs, srcs, dsts, label = get_batch(dataset_path, test_split_data, i, BATCH_SIZE)
        total = len(label)
        # print(len(embed), len(deep_label), len(src), len(dst))
        i = i + BATCH_SIZE
        g = dgl.graph((srcs, dsts))
        g = dgl.add_self_loop(g)
        if USE_GPU:
            g = g.to('cuda:0')

        inputs = torch.FloatTensor(inputs)
        label = torch.tensor(label)
        if USE_GPU:
            inputs = torch.FloatTensor(inputs).cuda()
            label = label.cuda()

        logits, conv = model(g, inputs)
        draw_tsne(conv, label, str(i)+'fe.jpg')



