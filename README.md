# Path2Trace: Case Association Tracing

### Description
There are two stages with three modules of this project. The two stages are the place representation learning and the case association tracing stage. The three modules are the city attraction representation learning module, the hierarchical place representation learning module and the case tracing prediction module.
Here we introduce the three modules as following:
1. City Attraction Representation Learning(CARL), corresponding to the `Competition_Aanlysis` folder:
   - Construct city migration dataset `cd Competiton_Analysis` and run `bash construct_dataset.sh`
   - Implementing the city attraction representation learning(CARL) process: `cd Competition_Analysis/DNE`, `bash get_all_embeddings.sh`
2. Hierarchical Place Representation Learning(HPRL), CARL and HPRL embedding fusion:
   - Hierarchical place representation learning: `cd representation_learning` and run the jupyter notebook `Representation learning`
   - There are three part in this notebook: HPRL, embedding fusion, and basic flat network representation learning baselines, details are shown in the notebook.
   - All pretrained place embeddings have saved in the `representation_learning/poi_embeddings/` folder. 
3. Case Association Prediction, corresponding to the `link_prediction` folder.
   - you can simply run `python attention_LP.py` to get the association prediction results by the attention neural network
   - run `python min_loc.py` to mask places according to the place importance or randomly and get the association prediction results.
   - By setting the `run_model` parameter in `attention_LP`, you can decide run `ablation` experiments, `baselines` experiments, `parameter sensitivity` experiments or simply training and testing process.
   
### Requirements

* Pytorch 
* python 3.x
* networkx
* scikit-learn
* scipy
* numpy
* pandas
* karateclub
