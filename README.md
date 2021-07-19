# HCLoC
# Requirements <br />
pytorch 1.7.0 <br />
python 3.7.8 <br />
dgl 0.5.3 <br />
flair 0.7 <br />
pycparser 2.20 <br />
javalang 0.13.0 <br />
gensim 3.8.3 <br />
# Run <br />
## pre-training 
1. run ```python parsercode.py --lang oj```/ ```python parsercode.py --lang gcj```/ ```python parsercode.py --lang bcb``` to generate initial encoding.
2. run ```python pre_training.py --dataset_nodeemb [The path to the dataset in which the nodes have been encoded]```
## Source Code Classification <br />
run ```python cla.py --nodedataset_path [node emb path] --pathdataset_path [path emb path] --pre_model [pre_model]```
## Source Code Detection <br />
run ```python clo.py --dataset [The name of the dataset] --pair_file [The path of the clone pairs] --nodedataset_path [node emb path] --pathdataset_path [path emb path] --pre_model [pre_model]```
