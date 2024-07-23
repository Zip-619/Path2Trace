# SpaceNE

This is a TensorFlow implementation of Hierarchical Community Structure Preserving Network Embedding (https://github.com/YimiAChack/SpaceNE.git), as described in our paper:
Hierarchical Community Structure Preserving Network Embedding: A Subspace Approach (CIKM 2019) 
https://www.gjsong-pku.cn/files/SpaceNE_CIKM.pdf



#### Run
`python main.py`

For parameter settings, please see `conf/case_travel.json`, the place embedding dict is stored at `represent_learning/poi_embeddings/spacene_poi_embedding.pickle`




#### Cite

```
@inproceedings{long2019hierarchical,
  title={Hierarchical Community Structure Preserving Network Embedding: A Subspace Approach},
  author={Long, Qingqing and Wang, Yiming and Du, Lun and Song, Guojie and Jin, Yilun and Lin, Wei},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={409--418},
  year={2019}
}
```

