
from dgl.nn.pytorch import GraphConv
import numpy as np
import torch
import torch.nn as nn

class Self_Attention(nn.Module):
    def __init__(self,in_feats,d_k,d_v,device):
        super(Self_Attention,self).__init__()
        self.W_Q = GraphConv(in_feats, d_k)
        self.W_K = GraphConv(in_feats, d_k)
        self.W_V = GraphConv(in_feats, d_v)
        self.W_O = GraphConv(d_v,in_feats)
        self.d_k=d_k
        self.device=device
    def forward(self,g,inputs,h_attn=None):
        Q = self.W_Q(g, inputs)
        K = self.W_K(g, inputs)
        V = self.W_V(g, inputs)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.FloatTensor([self.d_k])).to(self.device)
        if h_attn == None:
            attn = nn.Softmax(dim=-1)(scores)
        else:
            attn = nn.Softmax(dim=-1)(scores+h_attn)
        attn_out = torch.matmul(attn, V)
        attn_out=self.W_O(g,attn_out)
        return attn_out, attn
        pass

class GCNFeedforwardLayer(nn.Module):
    def __init__(self, in_feats, hidden_size,dropout):
        super(GCNFeedforwardLayer, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, in_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        out = self.dropout(torch.relu(self.conv1(g,inputs)))
        out=self.conv2(g,out)
        return out

class HCLLayer(nn.Module):
    def __init__(self,in_feats,d_k,d_v,hideen_size,dropout,device):
        super(HCLLayer,self).__init__()
        self.in_feats=in_feats
        self.self_attention=Self_Attention(in_feats,d_k,d_v,device)
        self.ln=nn.LayerNorm(in_feats)
        self.feedforward=GCNFeedforwardLayer(in_feats,hideen_size,dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self,g,inputs,attn=None):
        attn_out,attn=self.self_attention(g,inputs,attn)
        attn_out=self.ln(attn_out)
        out=self.feedforward(g,attn_out+inputs)
        out=self.ln(out)
        return out,attn


class HCL(nn.Module):
    def __init__(self,n_layers,in_feats,d_k,d_v,hidden_size,dropout,num_class,device):
        super(HCL,self).__init__()
        self.device=device
        self.layers=nn.ModuleList([HCLLayer(in_feats,d_k,d_v,hidden_size,dropout,device) for _ in range(n_layers)])
        self.cla1 = nn.Linear(in_feats,128)
        self.cla2 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(dropout)
    def forward(self,g,node_emb,attn=None):
        for layer in self.layers:
            node_emb,attn =layer(g,node_emb,attn)
        fe = self.dropout(torch.relu(self.cla1(node_emb)))
        out=self.cla2(fe)
        return out,fe
'''
直接拼接加池化求和
'''
def fefusion_pooling(node_num,path_num,node_emb,path_emb):
    import copy
    cat_polling=False
    gating=False
    if len(node_num)!=len(path_num):
        print('特征數量不匹配，無法進行特征融合')
        return []
    else:
        node_begin,path_begin=0,0
        ast=[]
        for i in range(len(node_num)):
            node_slice=copy.deepcopy(node_emb[node_begin:node_begin+node_num[i]])
            path_slice=copy.deepcopy(path_emb[path_begin:path_begin+path_num[i]])
            node_begin=node_begin+node_num[i]
            path_begin=path_begin+path_num[i]
            ast_temp=torch.cat((node_slice,path_slice),0)
            ast.append(torch.sum(ast_temp,0).numpy())
        tensor_data=torch.Tensor(ast)
        return tensor_data


def pooling(node_num,path_num,node_emb,path_emb):
    import copy
    if len(node_num)!=len(path_num):
        print('特征數量不匹配，無法進行特征融合')
        return []
    else:
        node_begin,path_begin=0,0
        node,path=[],[]
        for i in range(len(node_num)):
            # node_slice=copy.deepcopy(node_emb[node_begin:node_begin+node_num[i]])
            # path_slice=copy.deepcopy(path_emb[path_begin:path_begin+path_num[i]])

            node_slice = node_emb[node_begin:node_begin + node_num[i]]
            path_slice = path_emb[path_begin:path_begin + path_num[i]]
            node_begin=node_begin+node_num[i]
            path_begin=path_begin+path_num[i]
            node.append(torch.sum(node_slice,0))
            path.append(torch.sum(path_slice, 0))
        ast_node= torch.stack(node)
        ast_path =torch.stack(path)
        return ast_node,ast_path

class Gating(nn.Module):
    def __init__(self,heads,in_feature,device):
        super(Gating,self).__init__()
        self.heads=heads
        self.lq=nn.Linear(in_feature, in_feature // heads)
        self.lk1 = nn.Linear(in_feature, in_feature // heads)
        self.lk2 = nn.Linear(in_feature, in_feature // heads)
        self.lv1 = nn.Linear(in_feature, in_feature // heads)
        self.lv2 = nn.Linear(in_feature, in_feature // heads)
        self.lh = nn.Linear( in_feature, in_feature)
        self.scale = torch.sqrt(torch.FloatTensor([in_feature // heads])).to(device)
    def forward(self,Q,K,V):
        list_concat = []
        for i in range(self.heads):
            q=self.lq(Q)
            k1=self.lk1(K)
            k2=self.lk2(V)
            v1=self.lv1(K)
            v2=self.lv2(V)

            kv1=torch.sum(torch.mul(q,k1), dim=(1,), keepdim=True)
            kv2 = torch.sum(torch.mul(q, k2), dim=(1,), keepdim=True)

            kv1_1 = kv1 - torch.max(kv1, kv2)
            kv2_1 = kv2 - torch.max(kv1, kv2)

            kv1 = torch.exp(kv1_1)
            kv2= torch.exp(kv2_1)
            kv1_S = kv1 / (kv1 + kv2)
            kv2_S = kv2 / (kv1 + kv2)


            kv1_S =torch.mul(kv1_S ,v1)
            kv2_S =torch.mul(kv2_S ,v2)
            list_concat.append(kv1_S+ kv2_S)  # self.headattention_qkv(W_q, W_k, W_v, mask, flag, antimask))

        concat_head = torch.cat(list_concat, -1)
        W_o = self.lh(concat_head)
        return W_o

from tf import tfEncoder

class ClaModel(nn.Module):
    def __init__(self,
                 in_feature,
                 n_layers,
                 n_heads,
                 dropout,
                 n_class,
                 d_k=128, d_v=128, hidden_size=1024,device='cuda'):
        super(ClaModel,self).__init__()

        self.layers = nn.ModuleList(
            [HCLLayer(in_feature, d_k, d_v, hidden_size, dropout, device) for _ in range(n_layers)])
        # self.pre_model = pre_model
        self.tf=tfEncoder(in_feature,
                 n_layers,
                 n_heads,
                 dropout,
                 device)
        self.pooling=pooling
        self.gating=Gating(n_heads,in_feature,device)
        self.cla=nn.Sequential(
            nn.Linear(in_feature,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,n_class),
        )
    def forward(self,g,node_emb,path_emb,node_num,path_num,attn=None):
        for layer in self.layers:
            node_emb, attn = layer(g, node_emb, attn)
        path_emb=self.tf(path_emb)
        ast_node,ast_path=self.pooling(node_num,path_num,node_emb,path_emb)
        ast_out=self.gating(ast_node,ast_node,ast_path)
        cla_out=self.cla(ast_out)
        return cla_out


class CloModel(nn.Module):
    def __init__(self,
                 in_feature,
                 n_layers,
                 n_heads,
                 dropout,
                 n_class,
                 d_k=128, d_v=128, hidden_size=128,device='cuda'):
        super(CloModel,self).__init__()

        self.layers = nn.ModuleList(
            [HCLLayer(in_feature, d_k, d_v, hidden_size, dropout, device) for _ in range(n_layers)])
        # self.pre_model = pre_model
        self.tf=tfEncoder(in_feature,
                 n_layers,
                 n_heads,
                 dropout,
                 device)
        self.pooling=pooling
        self.gating=Gating(n_heads,in_feature,device)
        self.cla=nn.Sequential(
            nn.Linear(in_feature,128),
            nn.ReLU(),
            nn.Linear(128,n_class),
        )
    def forward(self,g1,node_emb1,path_emb1,node_num1,path_num1,g2,node_emb2,path_emb2,node_num2,path_num2,attn=None):
        for layer in self.layers:
            node_emb1, attn = layer(g1, node_emb1, attn)
        path_emb1=self.tf(path_emb1)
        ast_node1,ast_path1=self.pooling(node_num1,path_num1,node_emb1,path_emb1)
        ast_out1=self.gating(ast_node1,ast_node1,ast_path1)
        
        attn=None
        for layer in self.layers:
            node_emb2, attn = layer(g2, node_emb2, attn)
        path_emb2=self.tf(path_emb2)
        ast_node2,ast_path2=self.pooling(node_num2,path_num2,node_emb2,path_emb2)
        ast_out2=self.gating(ast_node2,ast_node2,ast_path2)

        ast_out= torch.abs(torch.add(ast_out1, -ast_out2))
        cla_out=self.cla(ast_out)
        cla_out = torch.sigmoid(cla_out)
        return cla_out


# import dgl
# srcs=[0,1,2,3,4]
# dsts=[1,0,1,4,5]
# g = dgl.graph((srcs, dsts))
# g = dgl.add_self_loop(g)
# g = g.to('cuda:0')
# #
# node_emb=torch.randn(6,768).cuda()
# path_emb=torch.randn(3,768).cuda()
# node_num=[4,2]
# path_num=[1,2]
# pre_model=torch.load('model.pkl')
# pre_model_dict=pre_model.state_dict()
# model=ClaModel(768,4,4,0.5,'cuda').cuda()
# model_dict=model.state_dict()
# pretrained_dict = {k:v for k, v in pre_model_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
#
# ast_out=model(g,node_emb,path_emb,node_num,path_num)
# print(ast_out)
#








