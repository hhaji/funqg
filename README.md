# FunQG: Molecular Representation Learning Via Quotient Graphs

**FunQG** is a novel graph coarsening framework specific to molecular data, utilizing functional groups based on a graph-theoretic concept called quotient graph. FunQG can accurately complete various molecular property prediction tasks with a significant parameters reduction. By experiments, this method significantly outperforms previous baselines on various datasets, besides its low computational complexity.

<p align="center">
   <img  src=https://github.com/zahta/funqg/blob/main/data/funqg.png?raw=true width="1000"/>
</p>

 <br>


## Requirements 
The resulting graphs of the FunQG are much smaller than the molecular graphs (Supplementary Section D). Therefore, a GNN model architecture requires much less depth in working with resulting graphs compared to working with molecular graphs. Thus, using FunQG reduces the computational complexity compared to working with molecular graphs. We utilize 1 *Intel (R) Xeon (R) E5-2699 v4 @ 2.20GHz* CPU for training, testing, and hyperparameter tuning of the FunQG-DMPNN on each dataset in a relatively short time.

- [PyTorch](https://pytorch.org/) >= 1.9.0

- [DGL](https://www.dgl.ai/) >= 0.6.0

- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html#) >= 1.9.0

- [RDKit](https://www.rdkit.org/) >= '2019.03.4'




