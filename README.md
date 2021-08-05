# HCLoC: Hierarchical Contrastive Learning of Code Representation
## Map Any Code Snippet into Vector Embedding with HCLoC
HCLoC is a self-supervised hierarchical contrastive learning model of code representation. Its key idea is to take the prediction node hierarchy as a pretext task for contrastive learning. Then, the learned weights are transferred to other downstream tasks or used only as embedding features. By applying a self-supervised framework, the representation features of the source code can be implemented to learn from unlabeled data.
# Requirements <br />
pytorch 1.7.0 <br />
python 3.7.8 <br />
dgl 0.5.3 <br />
flair 0.7 <br />
pycparser 2.20 <br />
javalang 0.13.0 <br />
gensim 3.8.3 <br />
# Usage
We extract the code AST node embedding and path embedding in the following two steps:
1. run ```python parsercode.py --lang oj```/ ```python parsercode.py --lang gcj```/ ```python parsercode.py --lang bcb``` to generate initial encoding.
2. run ```python pre_training.py --dataset_nodeemb [The path to the dataset in which the nodes have been encoded]```
# Application of HCLoC in downstream tasks
We evaluate HCLoC model on two tasks, source code classification and code clone detection. It is also expected to be helpful in more downstream tasks.
In the source code classification task, we evaluate HCLoC on two datasets: GCJ and OJ. In the code clone detection task, we further evaluate HCLoC on three datasets: BCB, GCJ and OJ. 
## Source Code Classification <br /> 
run ```python cla.py --nodedataset_path [node emb path] --pathdataset_path [path emb path] --pre_model [pre_model]```
## Code Clone Detection <br />
run ```python clo.py --dataset [The name of the dataset] --pair_file [The path of the clone pairs] --nodedataset_path [node emb path] --pathdataset_path [path emb path] --pre_model [pre_model]```
