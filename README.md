Source code for the NeurIPS 2022 paper [STNDT: Modeling Neural Population Activity with Spatiotemporal Transformers](https://papers.nips.cc/paper_files/paper/2022/file/72163d1c3c1726f1c29157d06e9e93c1-Paper-Conference.pdf)
# Spatio-Temporal Neural Data Transformers

This repo contains Python scripts to train and evaluate Spatio-Temporal Neural Data Transformers (STNDT). STNDT is developed based upon [Neural Data Transformers](https://github.com/snel-repo/neural-data-transformers) and allows the transformer to learn both the spatial coordination between neurons and the temporal progression of the population activity. STNDT is enhanced with a contrastive loss to encourage consistent firing rate predictions under different noise corruption. Evaluation of STNDT performance and other state-of-the-art methods can be found on [Neural Latents Benchmark Challenge 2021](https://neurallatents.github.io/).

## Environment Setup
Assuming you have Python 3.8+ and Miniconda installed, run the following to set up the environment with necessary dependencies:
```
conda env create -f environment.yml
```

## Data Setup
Download NLB data by running the following script:
```
./scripts/download_nlb_data.sh
````
Refer to instructions in the script to specify the URL of the dataset you want to download.

Next, run `src/nlb_data_setup.py` with the name of the dataset of interest to generate train, val and test data in HDF5 format:
```
python -u src/nlb_data_setup.py <dataset_name>
```
For example, to generate data for `MC_Maze`, use:
```
python -u src/nlb_data_setup.py mc_maze
```

## Training + Evaluation
The best performance of STNDT is achieved by ensembling multiple models obtained from Bayesian hyperparameter optimization. To get the candidates for ensembling, run the following script:
```
./scripts/train_ensemble.sh <variant_name>
```
For example, to train on `MC_Maze` dataset:
```
./scripts/train_ensemble.sh mc_maze
```
This script includes commands to launch a Bayesian hyperparameter optimization and ensemble the top N models to give the firing rate prediction on the test split. The hyperparameter sweep is defined in the corresponding `<dataset_name>.json`.

## Citation
If you find our code helpful, please cite our paper:
```
@inproceedings{NEURIPS2022_72163d1c,
 author = {Le, Trung and Shlizerman, Eli},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {17926--17939},
 publisher = {Curran Associates, Inc.},
 title = {STNDT: Modeling Neural Population Activity with Spatiotemporal Transformers},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/72163d1c3c1726f1c29157d06e9e93c1-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
This repo uses the following repositories and works. Please include their citations when using STNDT:

[Representation learning for neural population activity with Neural Data Transformers](https://github.com/snel-repo/neural-data-transformers):
```
@article{ye2021representation,
  title={Representation learning for neural population activity with Neural Data Transformers},
  author={Ye, Joel and Pandarinath, Chethan},
  journal={Neurons, Behavior, Data analysis, and Theory},
  volume={5},
  number={3},
  pages={1--18},
  year={2021},
  publisher={The neurons, behavior, data analysis and theory collective}
}
```
[Neural Latents Benchmark'21: Evaluating latent variable models of neural population activity](https://github.com/neurallatents/nlb_tools.git):
```
@article{pei2021neural,
  title={Neural Latents Benchmark’21: Evaluating latent variable models of neural population activity},
  author={Pei, F and Ye, J and Zoltowski, D and Wu, A and Chowdhury, RH and Sohn, H and O’Doherty, JE and Shenoy, KV and Kaufman, MT and Churchland, MM and others},
  journal={Advances in Neural Information Processing Systems (NeurIPS), Track on Datasets and Benchmarks},
  volume={34},
  year={2021}
}
```
[AEStudio's NLB Challenge Phase 1 Winning Model](https://github.com/agencyenterprise/ae-nlb-2021.git):
```
@misc{aestudio,
  author = {Sleiter, Darin and Schoenfield, Joshua and Vaiana, Mike},
  title = {ae-nlb-2021},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/agencyenterprise/ae-nlb-2021.git}},
  commit = {39d0de79aef2b997dcb419a9d3f9cd81180ee57b}
}
```
[Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/sthalles/SimCLR.git):
```
@article{
  silva2020exploringsimclr,
  title={Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations},
  author={Silva, Thalles Santos},
  journal={https://sthalles.github.io},
  year={2020}
  url={https://sthalles.github.io/simple-self-supervised-learning/}
}
```
[Transformers without Tears: Improving the Normalization of Self-Attention](https://github.com/tnq177/transformers_without_tears):
```
@inproceedings{nguyen2019transformers,
  title={Transformers without Tears: Improving the Normalization of Self-Attention},
  author={Nguyen, Toan Q and Salazar, Julian},
  booktitle={Proceedings of the 16th International Conference on Spoken Language Translation},
  year={2019}
}
```
[Improving transformer optimization through better initialization](https://github.com/layer6ai-labs/T-Fixup.git):
```
@inproceedings{huang2020improving,
  title={Improving transformer optimization through better initialization},
  author={Huang, Xiao Shi and Perez, Felipe and Ba, Jimmy and Volkovs, Maksims},
  booktitle={International Conference on Machine Learning},
  pages={4475--4483},
  year={2020},
  organization={PMLR}
}
```