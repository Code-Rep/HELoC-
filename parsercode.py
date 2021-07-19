import javalang
from javalang.ast import Node
import pandas as pd
from pycparser import c_parser
import copy
from tqdm import tqdm
import numpy as np
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
import torch
from Node import ASTNode,SingleNode
from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

# init embedding
embedding = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')


ID=-1
isDemo=True

def delete_blank(str):
    strRes=str.replace(' ','' ).replace('\n', '').replace('\r', '')
    return strRes

def c_get_node(node,node_List,id,src,dst,deep_label,deep_label_list):
    global ID
    id=ID
    node_List.append(str(node.coord)+delete_blank(str(node)))
    deep_label_list.append(deep_label)
    if node.children() is not None:
        for x, y in node.children():
            ID+=1
            src.append(id)
            dst.append(ID)
            c_get_node(y,node_List,ID,src,dst,deep_label+1,deep_label_list)
    else:
        pass

def c_get_ast(code):
    node_list=[]
    src=[]
    dst=[]
    deep_label_list=[]
    deep_label=0
    root=[]
    parser = c_parser.CParser()
    ast = parser.parse(code)
    global ID
    ID+=1
    root.append(ID)
    c_get_node(ast,node_list,ID,src,dst,deep_label,deep_label_list)
    return node_list,deep_label_list,src,dst,root


def parserC(c_path,save_path):
    code=pd.read_pickle(c_path)
    code.columns=['id','code','label']
    id,node_list, deep_label_list, src, dst, root,file_label=[],[],[],[],[],[],[]

    for index,item in code.iterrows():
        node_list_single, deep_label_list_single, src_single, dst_single, root_single=c_get_ast(item['code'])
        id.append(item['id'])
        node_list.append(copy.deepcopy(node_list_single))
        deep_label_list.append(copy.deepcopy(deep_label_list_single))
        src.append(copy.deepcopy(src_single))
        dst.append(copy.deepcopy(dst_single))
        root.append(copy.deepcopy(root_single))
        file_label.append(item['label'])
    parserC_dict={
        'id':id,
        'ast':node_list,
        'deep_label':deep_label_list,
        'src':src,
        'dst':dst,
        'root':root,
        'file_label':file_label
    }

    data=pd.DataFrame.from_dict(parserC_dict)
    data.to_pickle(save_path)
    return save_path


def get_node_emb(code):
    embed_list=[]
    for i in range(len(code)):
        sen=delete_blank(str(code[i]))
        sentence = Sentence(sen)

        with torch.no_grad():
            torch.cuda.empty_cache()
            embedding.embed(sentence)
        list = sentence.embedding.detach().cpu().numpy().tolist()
        embed_list.append(list)
    return embed_list


def embCode(ast_file_path,save_nodeemb_path,line_name='ast'):
    data=pd.read_pickle(ast_file_path)
    print(data)
    BATCH_SIZE=128
    i,len_data=0,len(data)
    end_flag=False
    while i<len_data:
        print(i)
        if i+BATCH_SIZE >= len_data:
            end_flag=True
            end_tmp=len_data
        else:
            end_tmp=i+BATCH_SIZE
        data_slice=data[i:end_tmp].copy()
        data_slice[line_name]= data_slice[line_name].apply(get_node_emb)
        print('data_slice',data_slice)
        data.to_pickle(save_nodeemb_path)
        data[i:end_tmp]=data_slice
        if end_flag:
            data.to_pickle(save_nodeemb_path)
            print(data)
        i += BATCH_SIZE
    pass


def c_trans_path_list(ast):
    ast_path_single=[]
    ast_path=[]
    c_trans_path(ast,ast_path_single,ast_path)
    return ast_path


def c_trans_path(node,ast_path_single,ast_path):
    current = SingleNode(node)
    ast_path_single.append(current.get_token())
    if current.is_leaf():
        ast_path.append(copy.deepcopy(ast_path_single))

    for path, child in node.children():
        c_trans_path(child,ast_path_single,ast_path)
        ast_path_single.pop()


def get_c_path(oj_path,save_ojpath_path):
    parser = c_parser.CParser()
    source = pd.read_pickle(oj_path)
    source.columns = ['id', 'code', 'file_label']
    source['code'] = source['code'].apply(parser.parse)
    source['code'] = source['code'].apply(c_trans_path_list())
    source.columns = ['id', 'path', 'file_label']
    source.to_pickle(save_ojpath_path)

'''

java

'''

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

def java_trans_path(node,ast_path_single, ast_path):
    token, children = java_get_token(node), java_get_children(node)
    ast_path_single.append(token)
    if len(children)==0:
        ast_path.append(copy.deepcopy(ast_path_single))


    for child in children:
        java_trans_path(child, ast_path_single,ast_path)
        ast_path_single.pop()

def get_java_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

def get_gcj_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree=parser.parse()
    return tree


def gcj_trans_path_list(code):

    ast=get_gcj_ast(code)
    ast_path_single=[]
    ast_path=[]
    java_trans_path(ast, ast_path_single,ast_path)
    return ast_path

def java_trans_path_list(code):
    ast=get_java_ast(code)
    ast_path_single=[]
    ast_path=[]
    java_trans_path(ast, ast_path_single,ast_path)
    return ast_path

ID=-1

def java_get_node_list(ast):
    node_list,src,dst,deep_label_list,root=[],[],[],[],[]
    deep_label=0
    global ID
    ID+=1
    root.append(ID)
    java_get_node(ast,node_list,ID,src,dst,deep_label,deep_label_list)

    return node_list,deep_label_list,src,dst,root

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



def java_get_node_list(ast):
    node_list,src,dst,deep_label_list,root=[],[],[],[],[]
    deep_label=0
    global ID
    ID+=1
    root.append(ID)
    java_get_node(ast,node_list,ID,src,dst,deep_label,deep_label_list)

    return node_list,deep_label_list,src,dst,root


def parserJava(java_path,save_path,dataset='gcj'):
    code=pd.read_pickle(java_path)
    if dataset=='gcj':
        code.columns = ['id', 'code','file_label']
    else:
        code.columns=['id','code']
    # print(code)
    id,node_list, deep_label_list, src, dst, root,file_label=[],[],[],[],[],[],[]


    for index,item in code.iterrows():

        ast=get_gcj_ast(item['code'])
        node_list_single, deep_label_list_single, src_single, dst_single, root_single=java_get_node_list(ast)
        id.append(item['id'])
        node_list.append(copy.deepcopy(node_list_single))
        deep_label_list.append(copy.deepcopy(deep_label_list_single))
        src.append(copy.deepcopy(src_single))
        dst.append(copy.deepcopy(dst_single))
        root.append(copy.deepcopy(root_single))
        file_label.append(item['label'])

    parserJava_dict={
        'id':id,
        'ast':node_list,
        'deep_label':deep_label_list,
        'src':src,
        'dst':dst,
        'root':root,
        'file_label':file_label
    }
    data=pd.DataFrame.from_dict(parserJava_dict)
    data.to_pickle(save_path)
    return save_path
def creat_dir(code_dir,node_dir,node_emb_dir,path_dir,path_emb_dir):
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
    if not os.path.exists(node_dir):
        os.makedirs(node_dir)
    if not os.path.exists(node_emb_dir):
        os.makedirs(node_emb_dir)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if not os.path.exists(path_emb_dir):
        os.makedirs(path_emb_dir)


if __name__ == '__main__':
    import argparse
    import os
    parser=argparse.ArgumentParser(description='Select a data set that you want to parse')
    parser.add_argument('--lang',default='gcj')
    args=parser.parse_args()
    if not args.lang:
        print('Please select a data set!')
        exit(1)
    else:
        if args.lang=='oj':
            root='data/oj/'
            oj_path=root+'programs.pkl'
            code_dir = root+ 'code'
            node_dir=root+'node'
            node_emb_dir=root+'node_emb'
            path_dir=root+'path'
            path_emb_dir = root + 'path_emb'

            source=pd.read_pickle(oj_path)
            i=0

            pbar= tqdm(range(len(source)))
            for i in pbar:
                pbar.set_description(('file:%d')%i)
                data_slice=source[i:i+1]
                data_slice.columns = ['id','code','file_label']
                data_slice=data_slice.reset_index(drop=True)
                data_slice.to_pickle(code_dir+'/'+str(i)+'.pkl')


            print('get ast...')
            for lists in tqdm(os.listdir(code_dir)):
                sub_path = os.path.join(code_dir, lists)
                sub_new_path=os.path.join(node_dir, lists)
                if os.path.isfile(sub_path):
                    save_path = parserC(sub_path, sub_new_path)


            node_emb_list=[]

            for lists in tqdm(os.listdir(node_emb_dir)):
                sub_path = os.path.join(node_emb_dir, lists)
                if os.path.isfile(sub_path):
                    node_emb_list.append(lists)

            print('get node emb...')
            for lists in tqdm(os.listdir(node_dir)):
                sub_path = os.path.join(node_dir, lists)
                sub_new_path=os.path.join(node_emb_dir, lists)
                if os.path.isfile(sub_path):
                    if lists in node_emb_list:
                        pass
                    else:
                        data = pd.read_pickle(sub_path)
                        data['ast'] = data['ast'].apply(get_node_emb)
                        data.to_pickle(sub_new_path)

            print('get path...')
            for lists in tqdm(os.listdir(code_dir)):
                sub_path = os.path.join(code_dir, lists)
                sub_new_path=os.path.join(path_dir, lists)
                if os.path.isfile(sub_path):
                    get_c_path(sub_path,sub_new_path)

            print('get path emb...')
            for lists in tqdm(os.listdir(path_dir)):
                sub_path = os.path.join(path_dir, lists)
                sub_new_path=os.path.join(path_emb_dir, lists)
                if os.path.isfile(sub_path):
                    data = pd.read_pickle(sub_path)
                    data['path'] = data['path'].apply(get_node_emb)
                    data.to_pickle(sub_new_path)

            # print('get path emb...')
            # embCode(save_ojpath_path,save_ojpathemb_path,'path')
        else:
            if args.lang=='bcb':
                root = 'data/bcb/'
                java_path = root + 'bcb_funcs.tsv'
                code_dir = root + 'code'
                node_dir = root + 'node'
                node_emb_dir = root + 'node_emb'
                path_dir = root + 'path'
                path_emb_dir = root + 'path_emb'
                source = pd.read_csv(java_path, sep='\t', header=0, encoding='utf-8')
            else:
                print('gcj...')
                root = 'data/gcj/'
                sec_root='test/'
                java_path = root + 'train.pkl'
                code_dir = root +sec_root + 'code'
                node_dir = root+sec_root + 'node'
                node_emb_dir = root+sec_root + 'node_emb'
                path_dir = root+sec_root + 'path'
                path_emb_dir = root +sec_root + 'path_emb'
                source = pd.read_pickle(java_path)
                print(source)

            creat_dir(code_dir,node_dir,node_emb_dir,path_dir,path_emb_dir)


            i=0
            print('get code...')
            parserWrong_list=[1223]
            pbar= tqdm(range(len(source)))
            for i in pbar:

                pbar.set_description(('file:%d')%i)
                data_slice=source[i:i+1]
                if args.lang == 'gcj':
                    source.columns = ['id', 'code', 'file_label']
                else:
                    source.columns = ['id', 'code']
                data_slice=data_slice.reset_index(drop=True)
                data_slice.to_pickle(code_dir+'/'+str(i)+'.pkl')

            print('get path...')
            for lists in tqdm(os.listdir(code_dir)):
                try:
                    sub_path = os.path.join(code_dir, lists)
                    sub_new_path = os.path.join(path_dir, lists)
                    if os.path.isfile(sub_path):
                        source = pd.read_pickle(sub_path)
                        source['code'] = source['code'].apply(gcj_trans_path_list)
                        if args.lang == 'gcj':
                            source.columns = ['id', 'path', 'file_label']
                        else:
                            source.columns = ['id', 'path']
                        source.to_pickle(sub_new_path)
                except Exception as e:
                    pass
                continue


            print('get path emb...')
            for lists in tqdm(os.listdir(path_dir)):
                sub_path = os.path.join(path_dir, lists)
                sub_new_path = os.path.join(path_emb_dir, lists)
                if os.path.isfile(sub_path):
                    data = pd.read_pickle(sub_path)
                    data['path'] = data['path'].apply(get_node_emb)
                    if args.lang=='gcj':
                        source.columns = ['id', 'path','file_label']
                    else:
                        source.columns = ['id', 'path']
                    data.to_pickle(sub_new_path)

            print('get node......')
            for lists in tqdm(os.listdir(code_dir)):
                try:
                    sub_path = os.path.join(code_dir, lists)
                    sub_new_path = os.path.join(node_dir, lists)
                    if os.path.isfile(sub_path):
                        save_path = parserJava(sub_path, sub_new_path)
                except Exception as e:
                    pass
                continue



            print('get node emb......')
            for lists in tqdm(os.listdir(node_dir)):
                sub_path = os.path.join(node_dir, lists)
                sub_new_path = os.path.join(node_emb_dir, lists)
                if os.path.isfile(sub_path):
                    data = pd.read_pickle(sub_path)
                    for i ,j in data.iterrows():
                        data['ast'] = data['ast'].apply(get_node_emb)
                        data.to_pickle(sub_new_path)
            data=pd.read_pickle(os.path.join(node_emb_dir, '0.pkl'))
