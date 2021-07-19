
import pandas as pd
from model import CloModel
import os
import pandas as pd
import torch
from tqdm import tqdm
from model import HCL
import dgl
import torch
import time
from sklearn.metrics import precision_recall_fscore_support


def get_node_batch(dataset_path, data_split, begin, BATCH_SIZE):
    inputs1, srcs1, dsts1,node_num1 = [], [], [], []
    inputs2, srcs2, dsts2, node_num2 = [], [], [], []
    labels=[]
    for i in range(begin, begin + BATCH_SIZE):
        data1 = pd.read_pickle(os.path.join(dataset_path, str(data_split[i][0])+'.pkl'))
        data2 = pd.read_pickle(os.path.join(dataset_path, str(data_split[i][1]) + '.pkl'))
        labels.append(data_split[i][2])
        for x, y in data1.iterrows():
            node_num1.append(len(y["ast"]))
            for src in y['src']:
                srcs1.append(src - y['root'][0] + len(inputs1))
            for dst in y['dst']:
                dsts1.append(dst - y['root'][0] + len(inputs1))
            for emb in y['ast']:
                inputs1.append(emb)
        
        for x, y in data2.iterrows():
            node_num2.append(len(y["ast"]))
            for src in y['src']:
                srcs2.append(src - y['root'][0] + len(inputs2))
            for dst in y['dst']:
                dsts2.append(dst - y['root'][0] + len(inputs2))
            for emb in y['ast']:
                inputs2.append(emb)
    return  inputs1, srcs1, dsts1,node_num1,inputs2, srcs2, dsts2, node_num2,labels



def get_path_batch(pathdataset_path, data_split, begin, BATCH_SIZE):
    path_emb1, path_num1 = [], []
    path_emb2, path_num2 = [], []
    for i in range(begin, begin + BATCH_SIZE):
        data1 = pd.read_pickle(os.path.join(pathdataset_path, str(data_split[i][0]) + '.pkl'))
        data2 = pd.read_pickle(os.path.join(pathdataset_path, str(data_split[i][1]) + '.pkl'))
        for x, y in data1.iterrows():
            path_num1.append(len(y['path']))
            for path in y['path']:
                path_emb1.append(path)

        for x, y in data2.iterrows():
            path_num2.append(len(y['path']))
            for path in y['path']:
                path_emb2.append(path)
    return path_emb1,path_num1,path_emb2,path_num2

def get_dataset(pair_file_path, train_ratio, val_ratio):
    train_data,val_data,test_data=[],[],[]
    data=pd.read_pickle(pair_file_path)
    data=data.loc[1:10]
    train_split=int(len(data)*0.1*train_ratio)
    val_split = int(len(data) * 0.1 * val_ratio)+train_split
    for i ,j in data.iterrows():
        if i < train_split:
            train_data.append([j['id1'],j['id2'],j['label']])
        elif i < val_split:
            val_data.append([j['id1'],j['id2'],j['label']])
        else:
            test_data.append([j['id1'],j['id2'],j['label']])
    return train_data,val_data,test_data

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code Clone Detection")
    parser.add_argument('--dataset', default='bcb',
                        help='The name of the dataset')
    parser.add_argument('--pair_file', default='',
                        help='The path of the clone pairs')
    parser.add_argument('--nodedataset_path', default='data/oj/node_emb',
                        help='node emb path')
    parser.add_argument('--pathdataset_path', default='data/oj/path_emb',
                        help='path emb path')
    parser.add_argument('--pre_model', default='model.pkl',
                        help='pre_model')
    args = parser.parse_args()

    dataset=args.dataset #The name of the dataset
    pair_file=args.pair_file
    nodedataset_path = args.nodedataset_path
    pathdataset_path = args.pathdataset_path
    pre_model = torch.load(args.pre_model)

    train_ratio, val_ratio, test_ratio = 6, 2, 2
    train_split_data,val_split_data,test_split_data=get_dataset(pair_file,train_ratio, val_ratio, test_ratio)



    begin = 0


    lr, BATCH_SIZE, EPOCH = 0.00001, 2, 5
    USE_GPU = False
    in_feats,n_layer,n_head,drop_out,n_class=768,4,4,0.5,1



    pre_model_dict = pre_model.state_dict()
    if USE_GPU:
        model = CloModel(in_feats,n_layer,n_head,drop_out,n_class, device='cuda').cuda()
    else:
        model = CloModel(in_feats, n_layer, n_head, drop_out, n_class, device='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.BCELoss()
    precision, recall, f1 = 0, 0, 0


    pbar = tqdm(range(EPOCH))

    best_model = model
    best_acc = 0.0
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []

    for t in range(1, n_class+ 1):
        train_data_t,test_data_t=[],[]
        if dataset == 'bcb':
            for i in range(len(train_split_data)):
                if train_split_data[i][2] == t:
                    train_data_t.append([train_split_data[i][0],train_split_data[i][1]],1)

            for i in range(len(test_split_data)):
                if test_split_data[i][2] == t:
                    test_data_t.append([test_split_data[i][0], test_split_data[i][1]], 1)

        else:
            train_data_t, test_data_t = train_split_data,test_split_data


        for epoch in pbar:
            pbar.set_description('epoch:%d  processing' % (epoch))
            i = 0
            start_time = time.time()
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            model.train()

            while (i + BATCH_SIZE) <= len(train_data_t):
                model.train()

                node_emb1, srcs1, dsts1,node_num1,node_emb2, srcs2, dsts2, node_num2,labels = get_node_batch(nodedataset_path, train_split_data, i, BATCH_SIZE)
                path_emb1,path_num1,path_emb2,path_num2=get_path_batch(pathdataset_path, train_split_data, i, BATCH_SIZE)
                i = i + BATCH_SIZE
                g1 = dgl.graph((srcs1, dsts1))
                g1 = dgl.add_self_loop(g1)
                g2 = dgl.graph((srcs2, dsts2))
                g2 = dgl.add_self_loop(g2)

                node_emb1 = torch.FloatTensor(node_emb1)
                path_emb1 = torch.FloatTensor(path_emb1)
                node_emb2 = torch.FloatTensor(node_emb2)
                path_emb2 = torch.FloatTensor(path_emb2)
                labels = torch.FloatTensor(labels)


                if USE_GPU:
                    g1 = g1.to('cuda:0')
                    node_emb1 = torch.FloatTensor(node_emb1).cuda()
                    path_emb1 = torch.FloatTensor(path_emb1).cuda()

                    g2 = g2.to('cuda:0')
                    node_emb2 = torch.FloatTensor(node_emb2).cuda()
                    path_emb2 = torch.FloatTensor(path_emb2).cuda()
                    labels = labels.cuda()

                model.zero_grad()
                logits=model(g1,node_emb1,path_emb1,node_num1,path_num1,g2,node_emb2,path_emb2,node_num2,path_num2)

                labels = labels.view(-1, 1)
                loss = loss_function(logits, labels)

                loss.backward()
                optimizer.step()

        print("Testing-%d...")
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0


        while (i + BATCH_SIZE) < len(test_data_t):
            node_emb1, srcs1, dsts1, node_num1, node_emb2, srcs2, dsts2, node_num2, labels = get_node_batch(
                nodedataset_path, train_split_data, i, BATCH_SIZE)
            path_emb1, path_num1, path_emb2, path_num2 = get_path_batch(pathdataset_path, train_split_data, i,
                                                                        BATCH_SIZE)
            i = i + BATCH_SIZE
            g1 = dgl.graph((srcs1, dsts1))
            g1 = dgl.add_self_loop(g1)
            g2 = dgl.graph((srcs2, dsts2))
            g2 = dgl.add_self_loop(g2)

            node_emb1 = torch.FloatTensor(node_emb1)
            path_emb1 = torch.FloatTensor(path_emb1)
            node_emb2 = torch.FloatTensor(node_emb2)
            path_emb2 = torch.FloatTensor(path_emb2)
            labels = torch.FloatTensor(labels)

            if USE_GPU:
                g1 = g1.to('cuda:0')
                node_emb1 = torch.FloatTensor(node_emb1).cuda()
                path_emb1 = torch.FloatTensor(path_emb1).cuda()

                g2 = g2.to('cuda:0')
                node_emb2 = torch.FloatTensor(node_emb2).cuda()
                path_emb2 = torch.FloatTensor(path_emb2).cuda()
                labels = labels.cuda()

            model.zero_grad()
            logits = model(g1, node_emb1, path_emb1, node_num1, path_num1, g2, node_emb2, path_emb2, node_num2,
                           path_num2)

            labels = labels.view(-1, 1)
            loss = loss_function(logits, labels)

            # calc testing acc
            predicted = (logits.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(labels.cpu().numpy())
            total += len(labels)
            total_loss += loss.item() * len(labels)
        if dataset == 'bcb':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
