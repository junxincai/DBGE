# DBGE: Employee Turnover Prediction based on Dynamic Bipartite Graph Embedding

This repository provides a reference implementation of the paper: 

> X. Cai et al., "DBGE: Employee Turnover Prediction Based on Dynamic Bipartite Graph Embedding," in IEEE Access, vol. 8, pp. 10390-10402, 2020.

which was published in the journal of IEEE Access.



## Environment settings

- python==3.7.2
- numpy==1.16.4
- pandas==0.24.2
- sklearn==0.19.1
- networkx==2.3
- tqdm==4.32.2
- gensim==3.7.1
- joblib==0.13.2




## Basic Usage



We provide two processed dataset for link prediction:

- Amazon. It contains:
  - A graph file           ./data/Amazon/amazon_bg.txt 
  - User vertices file     ./data/Amazon/amazon_user.txt 
  - Item vertices file     ./data/Amazon/amazon_item.txt 
  - A training dataset     ./data/Amazon/amazon_train.csv 
  - A testing dataset      ./data/Amazon/amazon_test.csv

- Taobao. It contains:
  - A graph file           ./data/Taobao/tb_bg.txt 
  - User vertices file     ./data/Taobao/tb_user.txt 
  - Item vertices file     ./data/Taobao/tb_item.txt 
  - A training dataset     ./data/Taobao/tb_train.csv 
  - A testing dataset      ./data/Taobao/tb_test.csv


### graph file sample

```
u0 i5 1350518400
u0 i2 1352246400
u0 i3 1370908800
u0 i6 1350518400
...
```

> noted: The userID and the itemID should be marked with different symbols.<br>
> noted: The third column in the file is a timestamp.

### Run

Please run the './model/train.py' 

```
cd model
python train.py
```

The embedding vectors of nodes are saved in file '/output/user_embeddings.csv' and '/out_put/item_embeddings.csv', respectively.


