## Galaxy Network Embedding: A Hierarchical Community Structure Preserving Approach
#### Authors: Lun Du, Zhicong Lu, Yun Wang, Guojie Song, Yiming Wang, Wei Chen
Implementation:  https://github.com/lundu28/GNE.git
## How to use

- We upload a sample case travel network dataset and you can run our algorithm on this dataset by the following command:
```shell
python main.py
```
- After running the command above, you can find the node embedding and classification results in `./res/new_train_res`.
- Then, by running the `gne_place_emb2dict.ipynb` notebook, the `python dict` of the place representation can be obtained and stored at `represent_learning/poi_embeddings/gne_poi_embedding.pickle`
