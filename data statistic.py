import pandas as pd
import copy
from Node import ASTNode,SingleNode
import javalang
from javalang.ast import Node


ID=-1
def java_get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'#node.pop()
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token



def java_get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def java_get_node(node,node_list,id,src,dst,deep_label,deep_label_list):
    token, children = java_get_token(node), java_get_children(node)
    node_content = ''

    if isinstance(node, Node):
        node_content = node_content + str(node.position)

    if len(children) == 0:
        node_content = node_content + str(token)
    else:
        node_content = node_content + str(token) + str(children)
    global ID
    id = ID
    node_list.append(node_content)
    deep_label_list.append(deep_label)
    for child in children:
        if java_get_token(child) == '':
            continue
        ID += 1
        src.append(id)
        dst.append(ID)
        java_get_node(child, node_list, ID, src, dst, deep_label + 1, deep_label_list)


def get_gcj_ast(dir,file):
    if dir=='train':
        data=pd.read_pickle('data/gcj/val.pkl')
    else:
        data = pd.read_pickle('data/gcj/test.pkl')
    code = data.loc[data['id'] == int(file), 'code'].values[0]
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree=parser.parse()
    return tree

def java_get_node_list(ast):
    node_list,src,dst,deep_label_list,root=[],[],[],[],[]
    deep_label=0
    global ID
    ID+=1
    root.append(ID)
    java_get_node(ast,node_list,ID,src,dst,deep_label,deep_label_list)
    return node_list,deep_label_list,src,dst,root



data1=pd.read_pickle('clo/gcj_clone_ids2.pkl')
deep_avg,deep_max,node_max,node_avg=0.0,0,0,0.0
node,deep=[],[]
from tqdm import tqdm
for i ,j in tqdm(data1.iterrows()):
    id1=j['id1']
    id2=j['id2']
    id1_path=id1.split("/")
    id2_path=id2.split("/")
    tree1 = get_gcj_ast(id1_path[0], id1_path[1])
    tree2 = get_gcj_ast(id2_path[0], id2_path[1])
    node_list1,deep_label_list1,src1,dst1,root1=java_get_node_list(tree1)
    node_list2,deep_label_list2,src2,dst2,root2=java_get_node_list(tree2)
    node.append(len(node_list1))
    node.append(len(node_list2))
    deep.append(max(deep_label_list1))
    deep.append(max(deep_label_list2))

    node_single=len(node_list1) if len(node_list1)>len (node_list2) else len(node_list2)
    deep_single=max(deep_label_list1) if max(deep_label_list1)>=max(deep_label_list2) else max(deep_label_list2)

    deep_max=deep_max if deep_max >= deep_single else deep_single
    node_max=node_max if node_max >= node_single else node_single

from numpy import *
node_avg= mean(node)
deep_avg= mean(deep)
print('node_avg',node_avg,'deep_avg',deep_avg,'node_max',node_max,'deep_max',deep_max)