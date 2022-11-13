
""" Utils for Graph Generator """

import numpy as np
import pandas as pd
import torch
import dgllife, dgl
import networkx as nx
from torch.utils.data import Subset
from rdkit import Chem
from dgllife.utils import featurizers as fs
from collections import namedtuple
from descriptastorus.descriptors import rdNormalizedDescriptors
from deepchem.data import Dataset
from deepchem.splits import Splitter
from random import Random
from typing import List, Optional, Tuple


""" Node and Edge Features """
class CanonicalAtomFeaturizer(fs.BaseAtomFeaturizer):
    def __init__(self, atom_data_field='h'):
        super().__init__(
            featurizer_funcs={atom_data_field: fs.ConcatFeaturizer(
                [
                    fs.atomic_number_one_hot,
                    fs.atom_total_degree_one_hot,
                    fs.atom_formal_charge_one_hot,
                    fs.atom_chiral_tag_one_hot,
                    fs.atom_total_num_H_one_hot,
                    fs.atom_hybridization_one_hot,
                    fs.atom_is_aromatic,
                    fs.atom_mass,
                ])})

class CanonicalBondFeaturizer(fs.BaseBondFeaturizer):
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: fs.ConcatFeaturizer(
                [
                    fs.bond_type_one_hot,
                    fs.bond_is_conjugated,
                    fs.bond_is_in_ring,
                    fs.bond_stereo_one_hot,
                ])}, self_loop=self_loop)


""" Functional Group by Ertl's algorithm """
"""
Original authors: Richard Hall and Guillaume Godin. This file is part of the RDKit.
The contents are covered by the terms of the BSD license which is included in the file license.txt, 
found at the root of the RDKit source tree.
Richard hall 2017 # IFG main code # Guillaume Godin 2017.
refine output function: identify functional groups, Ertl, J. Cheminform (2017).
"""
def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)

PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')
PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)

def identify_functional_groups(mol):
    marked = set()
    #mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6,1): # would we ever have hydrogen?
            marked.add(atom.GetIdx())

    #mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

    #merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)

    #extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
    ifgs = []
    ifgs_types = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        all_type=g.union(uca)
        list_type_g=list(all_type)
        ifgs_types.append(list_type_g)
        ifgs.append(ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True), type=Chem.MolFragmentToSmiles(mol, g.union(uca),canonical=True)))
    
    return ifgs, ifgs_types


"""" Global Features """
"""
Reference: https://github.com/bp-kelley/descriptastorus/blob/master/descriptastorus/descriptors/DescriptorGenerator.py
def create_descriptors(df: pd.DataFrame,
                       mols_column_name: str,
                       generator_names: list):
"""
def create_descriptors(df: pd.DataFrame,
                       mols_column_name: str
                       ):
    """pyjanitor style function for using the descriptor generator
    Convert a column of smiles strings or RDKIT Mol objects into Descriptors.
    Returns a new dataframe without any of the original data. This is
    intentional, as Descriptors are usually high-dimensional
    features.
    This method does not mutate the original DataFrame.
    .. code-block:: python
        import pandas as pd
        import descriptastorus.descriptors
        df = pd.DataFrame(...)
        # For "counts" kind
        descriptors = descriptastorus.descriptors.create_descriptors(
            mols_column_name='smiles', generator_names=["Morgan3Count"])
    """
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    mols = df[mols_column_name]
    if len(mols):
        if type(mols[0]) == str:
            _, results = generator.processSmiles(mols)
        else:
            results = generator.processMols(mols, [Chem.MolToSmiles(m) for m in mols])

    else:
        results = []
    fpdf = pd.DataFrame(results, columns=generator.GetColumns())
    fpdf.index = df.index
    return fpdf


""" Splitters """
# RandomScaffoldSplitter is a modified version of https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py 
# to obtain randomly seeded scaffold splittings similar to that of " Wu, Z., Ramsundar, B., Feinberg, E.N., Gomes, J., Geniesse, C., Pappu,
#  A.S., Leswing, K. and Pande, V., 2018. MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), pp.513–530"
# Reference for other splitters: 
# https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py

# Scaffold Splitter    
class RandomScaffoldSplitter(Splitter):            
  def split(self,
            dataset, #dataset: Dataset, # imagining a pandas df
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = 1000
            ) -> Tuple[List[int], List[int], List[int]]:
    
          """
                Splits internal compounds into train/validation/test by scaffold.
                Parameters
                ----------
                dataset: Dataset
                Dataset to be split.
                frac_train: float, optional (default 0.8)
                The fraction of data to be used for the training split.
                frac_valid: float, optional (default 0.1)
                The fraction of data to be used for the validation split.
                frac_test: float, optional (default 0.1)
                The fraction of data to be used for the test split.
                seed: int, optional (default None)
                Random seed to use.
                log_every_n: int, optional (default 1000)
                Controls the logger by dictating how often logger outputs
                will be produced.
                Returns
                -------
                Tuple[List[int], List[int], List[int]]
                A tuple of train indices, valid indices, and test indices.
                Each indices is a list of integers.

          """          
          np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

          train_size = frac_train * len(dataset)
          valid_size = frac_valid * len(dataset)
          test_size = frac_test * len(dataset)
          train_inds: List[int] = []
          valid_inds: List[int] = []
          test_inds: List[int] = []

          scaffold_sets = self.generate_scaffolds(dataset)

          # Seed randomness
          random = Random(seed)

        #   logger.info("About to sort in scaffold sets")

          # Put stuff that's bigger than half the val/test size into train, rest just order randomly
          big_index_sets = []
          small_index_sets = []
          for index_set in scaffold_sets:
              if len(index_set) > valid_size / 2 or len(index_set) > test_size / 2:
                  big_index_sets.append(index_set)
              else:
                  small_index_sets.append(index_set)
          random.seed(seed)
          random.shuffle(big_index_sets)
          random.shuffle(small_index_sets)
          scaffold_sets = big_index_sets + small_index_sets

          for index_set in scaffold_sets:
              if len(train_inds) + len(index_set) <= train_size:
                  train_inds += index_set
                  # list_train_inds.append(index_set)
                  # train_scaffold_count += 1
              elif len(valid_inds) + len(index_set) <= valid_size:
                  valid_inds += index_set
                  # list_valid_inds.append(index_set)
                  # val_scaffold_count += 1
              else:
                  test_inds += index_set
                  # list_test_inds.append(index_set)
                  # test_scaffold_count += 1

          return train_inds, valid_inds, test_inds
  
  def generate_scaffolds(self, #dataset: Dataset,
                          dataset, log_every_n: int = 1000
                        ) -> List[List[int]]:
    
        """Returns all scaffolds from the dataset.
        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.
        log_every_n: int, optional (default 1000)
          Controls the logger by dictating how often logger outputs
          will be produced.
        Returns
        -------
        scaffold_sets: List[List[int]]
          List of indices of each scaffold in the dataset.
        """

        scaffolds = {}
        data_len = len(dataset)

        #for ind, smiles in enumerate(dataset.ids):
        for ind, smiles in enumerate(dataset.smiles.to_list()): ## inserting a pandas df with smiles column
          scaffold = _generate_scaffold(smiles)
          if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
          else:
            scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

        return scaffold_sets

def _generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
      
      """
        Compute the Bemis-Murcko scaffold for a SMILES string.
        Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
        They are essentially that part of the molecule consisting of
        rings and the linker atoms between them.
        Paramters
        ---------
        smiles: str
            SMILES
        include_chirality: bool, default False
            Whether to include chirality in scaffolds or not.
        Returns
        -------
        str
            The MurckScaffold SMILES from the original SMILES
        References
        ----------
        .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
            1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
        Note
        ----
        This function requires RDKit to be installed.
      """
      try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
      except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

      mol = Chem.MolFromSmiles(smiles)
      scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
      return scaffold  

# Random Splitter 
class RandomSplitter(Splitter):
  
  """Class for doing random data splits.
        Examples
        --------
        >>> import numpy as np
        >>> import deepchem as dc
        >>> # Creating a dummy NumPy dataset
        >>> X, y = np.random.randn(5), np.random.randn(5)
        >>> dataset = dc.data.NumpyDataset(X, y)
        >>> # Creating a RandomSplitter object
        >>> splitter = dc.splits.RandomSplitter()
        >>> # Splitting dataset into train and test datasets
        >>> train_dataset, test_dataset = splitter.train_test_split(dataset)
  """

  def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Splits internal compounds randomly into train/validation/test.
        Parameters
        ----------
        dataset: Dataset
        Dataset to be split.
        seed: int, optional (default None)
        Random seed to use.
        frac_train: float, optional (default 0.8)
        The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
        The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
        The fraction of data to be used for the test split.
        seed: int, optional (default None)
        Random seed to use.
        log_every_n: int, optional (default None)
        Log every n examples (not currently used).
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of train indices, valid indices, and test indices.
        Each indices is a numpy array.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if seed is not None:
      np.random.seed(seed)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    shuffled = np.random.permutation(range(num_datapoints))
    return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:])

# Random Stratified Splitter 
class RandomStratifiedSplitter_m(Splitter): # '_m' stands for slight modification
 
  """RandomStratified Splitter class.
        For sparse multitask datasets, a standard split offers no guarantees
        that the splits will have any active compounds. This class tries to
        arrange that each split has a proportional number of the actives for each
        task. This is strictly guaranteed only for single-task datasets, but for
        sparse multitask datasets it usually manages to produces a fairly accurate
        division of the actives for each task.
        Note
        ----
        This splitter is primarily designed for boolean labeled data. It considers
        only whether a label is zero or non-zero. When labels can take on multiple
        non-zero values, it does not try to give each split a proportional fraction
        of the samples with each value.
  """

  def split(self,
            dataset, # inserting a pandas df
            tasks_range: List[int] = [2,6], # range of task indexes that should be considered # default: range for the (covid) clinical trials
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            log_every_n: Optional[int] = None) -> Tuple:
 
    """Return indices for specified split
        Parameters
        ----------
        dataset: dc.data.Dataset
        Dataset to be split.
        seed: int, optional (default None)
        Random seed to use.
        frac_train: float, optional (default 0.8)
        The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
        The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
        The fraction of data to be used for the test split.
        log_every_n: int, optional (default None)
        Controls the logger by dictating how often logger outputs
        will be produced.
        Returns
        -------
        Tuple
        A tuple `(train_inds, valid_inds, test_inds)` of the indices (integers) for
        the various splits.
    """

    if seed is not None:
      np.random.seed(seed)

    # Figure out how many positive samples we want for each task in each dataset.
    y_present = dataset.fillna(0).iloc[:,tasks_range[0]:tasks_range[1]].to_numpy().astype('int64') # pandas df # .fillna(0) is very important

    n_tasks = y_present.shape[1]
    indices_for_task = [
        np.random.permutation(np.nonzero(y_present[:, i])[0])
        for i in range(n_tasks)]

    count_for_task = np.array([len(x) for x in indices_for_task])
    train_target = np.round(frac_train * count_for_task).astype(int)
    valid_target = np.round(frac_valid * count_for_task).astype(int)
    test_target = np.round(frac_test * count_for_task).astype(int)

    # Assign the positive samples to datasets.  Since a sample may be positive
    # on more than one task, we need to keep track of the effect of each added
    # sample on each task.  To try to keep everything balanced, we cycle through
    # tasks, assigning one positive sample for each one.
    train_counts = np.zeros(n_tasks, int)
    valid_counts = np.zeros(n_tasks, int)
    test_counts = np.zeros(n_tasks, int)
    set_target = [train_target, valid_target, test_target]
    set_counts = [train_counts, valid_counts, test_counts]
    set_inds: List[List[int]] = [[], [], []]
    assigned = set()
    max_count = np.max(count_for_task)
    for i in range(max_count):
      for task in range(n_tasks):
        indices = indices_for_task[task]
        if i < len(indices) and indices[i] not in assigned:
          # We have a sample that hasn't been assigned yet.  Assign it to
          # whichever set currently has the lowest fraction of its target for
          # this task.
          index = indices[i]
          set_frac = [
              1 if set_target[i][task] == 0 else
              set_counts[i][task] / set_target[i][task] for i in range(3)
          ]
          set_index = np.argmin(set_frac)
          set_inds[set_index].append(index)
          assigned.add(index)
          set_counts[set_index] += y_present[index]

    # The remaining samples are negative for all tasks.  Add them to fill out
    # each set to the correct total number.
    n_samples = y_present.shape[0]
    set_size = [
        int(np.round(n_samples * f))
        for f in (frac_train, frac_valid, frac_test)]

    s = 0
    for i in np.random.permutation(range(n_samples)):
      if i not in assigned:
        while s < 2 and len(set_inds[s]) >= set_size[s]:
          s += 1
        set_inds[s].append(i)

    return tuple(sorted(x) for x in set_inds)


"""Some functions"""
def fg_idx(mol):
    fgs = identify_functional_groups(mol)[0]
    return [list(fgs[i][0]) for i in range(len(fgs)) if len(fgs)>0]

def fg_types_idx(mol):
    fg_types = identify_functional_groups(mol)[1]
    return fg_types

def convert_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def make_df(url):
    data = pd.read_csv(url)
    df = pd.DataFrame()
    df["Smiles"] = data["smiles"]
    df["FGs"] = data["smiles"].apply(convert_mol).apply(fg_idx)
    return df

def fgs_connections_idx(df, smiles, mol_dgl_graph):
    mol_dgl_graph.edata["edges_fgs"] = torch.zeros(mol_dgl_graph.num_edges(), 1)
    mol_dgl_graph.edata["edges_non_fgs"] = torch.zeros(mol_dgl_graph.num_edges(), 1)
    nodes_fgs =[]
    if df[df["Smiles"] == smiles]["FGs"].squeeze() != []:
        for fgs in df[df["Smiles"] == smiles]["FGs"].squeeze():
            mol_dgl_sub_graph=dgl.node_subgraph(mol_dgl_graph, fgs)
            mol_dgl_graph.edata["edges_fgs"][mol_dgl_sub_graph.edata[dgl.EID].long()]=torch.ones(len(mol_dgl_sub_graph.edata[dgl.EID]), 1)
            nodes_fgs += fgs
        nodes_non_fgs=[node for node in range(mol_dgl_graph.num_nodes()) if node not in nodes_fgs]
        mol_dgl_sub_graph=dgl.node_subgraph(mol_dgl_graph, nodes_non_fgs)
        mol_dgl_graph.edata["edges_non_fgs"][mol_dgl_sub_graph.edata[dgl.EID].long()]=torch.ones(len(mol_dgl_sub_graph.edata[dgl.EID]), 1)
    else:
        mol_dgl_graph.edata["edges_non_fgs"] = torch.ones(mol_dgl_graph.num_edges(), 1)
    return mol_dgl_graph.edata["edges_non_fgs"], mol_dgl_graph.edata["edges_fgs"]

def graph_constructor(df, smiles, types_in2out=True, types_out2in=True, fragment=True):
    mol = Chem.MolFromSmiles(smiles)
    mol_dgl_graph = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False).int()

    ### Determine edges among Functional Groups and among Non-Functional Groups (both vertices)  
    mol_dgl_graph.edata["edges_non_fgs"], mol_dgl_graph.edata["edges_fgs"] = fgs_connections_idx(df, smiles, mol_dgl_graph)

    return mol_dgl_graph

def splitted_data(path_smiles, dataset, dataset_smiles_series, string="train"):
    """
    A function to find train, validation, or test set split based on splitted smiles files
    string+"_smiles": Name of the splitted smiles file
    """
    Smiles = np.load(path_smiles+string+"_smiles", allow_pickle=True)
    print(path_smiles+string+"_smiles")
    splitted_idxs = []
    for smiles in Smiles:
        splitted_idxs.append(dataset_smiles_series[dataset_smiles_series==smiles].index.values[0])
    splitted_data = Subset(dataset, splitted_idxs)
    return splitted_data    


"""Module for converting graph to other NetworkX graph"""
# Reference: https://docs.dgl.ai/en/0.8.x/generated/dgl.to_networkx.html

def to_networkx(g, node_attrs=None, edge_attrs=None, digraph=True):
    src, dst = g.edges()
    src = dgl.backend.asnumpy(src)
    dst = dgl.backend.asnumpy(dst)
    # xiangsx: Always treat graph as multigraph
    # nx_graph = nx.MultiDiGraph()
    if digraph: 
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(g.number_of_nodes()))
    for eid, (u, v) in enumerate(zip(src, dst)):
        nx_graph.add_edge(u, v, id=eid)

    if node_attrs is not None:
        for nid, attr in nx_graph.nodes(data=True):
            feat_dict = g._get_n_repr(0, nid)
            attr.update({key: dgl.backend.squeeze(feat_dict[key], 0) for key in node_attrs})
    if edge_attrs is not None:
        for _, _, attr in nx_graph.edges(data=True):
            eid = attr['id']
            feat_dict = g._get_e_repr(0, eid)
            attr.update({key: dgl.backend.squeeze(feat_dict[key], 0) for key in edge_attrs})
    return nx_graph

"""Quotient graph generator"""
def quotient_generator(dgl_graph, edge_condition_feature, op="mean", another_edges_feature=False, another_nodes_feature=False):
    '''
    graph: dgl Graph 
    edge_condition_feature str: which shows a Boolean Feature. We would like to save the edges with True labels 
    nodes_feature: A string which is used as the output of function
    edges_feature: A string which is used as the output of function
    '''
    def edges_with_feature_True(edges):
        return (edges.data[edge_condition_feature]== False).squeeze(1)

    id_positive_edges= dgl_graph.filter_edges(edges_with_feature_True).to(dtype=torch.int32)
    # print("id graph", dgl_graph.nodes())
    if len(id_positive_edges)>0:
        dgl_graph_subgraph = dgl.remove_edges(dgl_graph, id_positive_edges, etype=None, store_ids=True)
        dgl_graph_subgraph_simple = dgl.to_simple(dgl_graph_subgraph)
        nx_subgraph = to_networkx(dgl_graph_subgraph_simple, digraph=False)
        connected_components_graph = nx.connected_components(nx_subgraph)
        list_partition_nodes=[]
        count_0 =1
        dgl_graph.ndata["qn1"]=torch.zeros((dgl_graph.num_nodes(),1))
        dgl_graph.edata["qe1"]=torch.zeros((dgl_graph.num_edges(),1))
        for a in connected_components_graph:
            real_idx_nodes= dgl_graph_subgraph.ndata[dgl.NID][list(a)]
            real_idx_nodes=real_idx_nodes.tolist()
            dgl_graph.ndata["qn1"][real_idx_nodes] = dgl_graph.ndata["qn1"][real_idx_nodes] +count_0
            count_0+=1
            list_partition_nodes.append(real_idx_nodes)
    else:
        dgl_graph_subgraph = dgl_graph
        dgl_graph_subgraph_simple = dgl.to_simple(dgl_graph_subgraph)
        nx_subgraph = to_networkx(dgl_graph_subgraph_simple, digraph=False)
        connected_components_graph = nx.connected_components(nx_subgraph)
        list_partition_nodes=[]
        count_0 =1
        dgl_graph.ndata["qn1"]=torch.zeros((dgl_graph.num_nodes(),1))
        dgl_graph.edata["qe1"]=torch.zeros((dgl_graph.num_edges(),1))
        for a in connected_components_graph:
            dgl_graph.ndata["qn1"][list(a)] = dgl_graph.ndata["qn1"][list(a)] +count_0
            count_0+=1
            list_partition_nodes.append(list(a))

    def node_func(nodes):
        if another_nodes_feature:
            op1 =eval("torch."+op)
            return {"v":op1(dgl_graph.ndata["v"][torch.tensor(list(nodes)).long()], 0),\
                    "qn2": dgl_graph.ndata["qn1"][list(nodes)[0]],\
                    another_nodes_feature: op1(dgl_graph.ndata[another_nodes_feature][torch.tensor(list(nodes)).long()], 0)}
        else:
            op1 =eval("torch."+op)
            return {"v":op1(dgl_graph.ndata["v"][torch.tensor(list(nodes)).long()], 0),\
                    "qn2": dgl_graph.ndata["qn1"][list(nodes)[0]]}  

    count =1
    def edge_func(nodes1, nodes2):
        nonlocal count
        nodes1=list(nodes1)
        nodes2=list(nodes2)
        ids_edges=[]
        for a in nodes1:
            for b in nodes2:
                try:
                    t=dgl_graph.edge_ids(a,b)
                    ids_edges.append(t)
                except:
                    pass
        dgl_graph.edata["qe1"][ids_edges]= dgl_graph.edata["qe1"][ids_edges] + count
        count+= 1
        if another_edges_feature: 
            op2 =eval("torch."+op)
            return {"e": op2(dgl_graph.edata["e"][ids_edges], 0), "qe2": dgl_graph.edata["qe1"][ids_edges[0]],\
                    another_edges_feature: op2(dgl_graph.edata[another_edges_feature][ids_edges], 0)}
        else:
            op2 =eval("torch."+op)
            return {"e": op2(dgl_graph.edata["e"][ids_edges], 0), "qe2": dgl_graph.edata["qe1"][ids_edges[0]]}

    if another_nodes_feature:
        features_nodes_q1 = ["v", "qn1", another_nodes_feature]
        features_nodes_q2 = ["v", "qn2", another_nodes_feature]
    else:
        features_nodes_q1 = ["v", "qn1"]
        features_nodes_q2 = ["v", "qn2"]

    if another_edges_feature:
        features_edges_q1 = ["e", "qe1", another_edges_feature]
        features_edges_q2 = ["e", "qe2", another_edges_feature]
    else:
        features_edges_q1 = ["e", "qe1"]
        features_edges_q2 = ["e", "qe2"]

    nx_graph = to_networkx(dgl_graph, node_attrs= features_nodes_q1, edge_attrs= features_edges_q1)
    nx_quotient= nx.algorithms.minors.quotient_graph(nx_graph, list_partition_nodes, node_data=node_func, edge_data=edge_func,\
                                                    relabel=False)
    if nx_quotient.number_of_edges()==0:
        dgl_graph_quotient = dgl.from_networkx(nx_quotient, node_attrs=features_nodes_q2, edge_attrs=[]).int()
        dgl_graph_quotient.edata["e"] = torch.zeros(dgl_graph_quotient.num_edges(), 20)
        dgl_graph_quotient.edata["qe2"] = torch.zeros(dgl_graph_quotient.num_edges(), 1)
        dgl_graph_quotient.edata["qe1"] = torch.zeros(dgl_graph_quotient.num_edges(), 1)
        if another_edges_feature:
            dgl_graph_quotient.edata[another_edges_feature] = torch.zeros(dgl_graph_quotient.num_edges(), 1)
    else:
        dgl_graph_quotient =dgl.from_networkx(nx_quotient, node_attrs=features_nodes_q2, edge_attrs=features_edges_q2).int()
    return dgl_graph_quotient


"""Datasets"""

"""Tox21 dataset"""
class DatasetTox21(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        # Unzip the dataset and read its csv file, and fill in NaN values with 0
        self.csv = pd.read_csv(csv_address).fillna(0) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Make masks for labels (0 as NaN value, and 1 as other values)
        self.masks_csv = pd.read_csv(csv_address).replace({0: 1}).fillna(0)

        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:,13]
        self.labels = self.csv.iloc[:,:12].values
        self.masks = self.masks_csv.iloc[:,:12].values

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()


"""BBBP dataset"""
class DatasetBBBP(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        self.csv = pd.read_csv(csv_address) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:,3]
        self.labels = self.csv.iloc[:,2]
        self.masks = torch.ones((len(self.smiles), 1))

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),\
        torch.tensor(self.global_feats[idx]).float()

"""Bace dataset"""
class DatasetBace(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        self.csv = pd.read_csv(csv_address) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:,0]
        self.labels = self.csv.iloc[:,2]
        self.masks = torch.ones((len(self.smiles), 1))

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),\
        torch.tensor(self.global_feats[idx]).float()

"""Toxcast dataset"""
class DatasetToxcast(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        # Unzip the dataset and read its csv file, and fill in NaN values with 0
        self.csv = pd.read_csv(csv_address).fillna(0) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Make masks for labels (0 as NaN value, and 1 as other values)
        self.masks_csv = pd.read_csv(csv_address).replace({0: 1}).fillna(0)

        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:, 0]
        self.labels = self.csv.iloc[:, 1:].values
        self.masks = self.masks_csv.iloc[:, 1:].values

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()

"""Clintox dataset"""
class DatasetClintox(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        # Unzip the dataset and read its csv file, and fill in NaN values with 0
        self.csv = pd.read_csv(csv_address).fillna(0) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Make masks for labels (0 as NaN value, and 1 as other values)
        self.masks_csv = pd.read_csv(csv_address).replace({0: 1}).fillna(0)

        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:, 0]
        self.labels = self.csv.iloc[:, 1:].values
        self.masks = self.masks_csv.iloc[:, 1:].values

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()

"""Sider dataset"""
class DatasetSider(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        # Unzip the dataset and read its csv file, and fill in NaN values with 0
        self.csv = pd.read_csv(csv_address).fillna(0) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Make masks for labels (0 as NaN value, and 1 as other values)
        self.masks_csv = pd.read_csv(csv_address).replace({0: 1}).fillna(0)

        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:, 0]
        self.labels = self.csv.iloc[:, 1:].values
        self.masks = self.masks_csv.iloc[:, 1:].values

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()

"""Lipophilicity dataset"""
class DatasetLipo(torch.utils.data.Dataset):
    def __init__(self, csv_address, path_global_csv):
        self.csv = pd.read_csv(csv_address) 
        self.path_global_csv = pd.read_csv(path_global_csv)
        
        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:,2]
        self.labels = self.csv.iloc[:,1]
        self.masks = torch.ones((len(self.smiles), 1))

        self.global_feats = self.path_global_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),\
        torch.tensor(self.global_feats[idx]).float()

"""ESOL dataset"""
class DatasetESOL(torch.utils.data.Dataset):
    def __init__(self, csv_address, global_feats_csv):
        self.csv = pd.read_csv(csv_address) 
        self.global_feats_csv = pd.read_csv(global_feats_csv)

        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:,9]
        self.labels = self.csv.iloc[:,8]
        self.masks = torch.ones((len(self.smiles), 1))

        self.global_feats = self.global_feats_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()


"""FreeSolv dataset"""
class DatasetFreeSolv(torch.utils.data.Dataset):
    def __init__(self, csv_address, global_feats_csv):
        self.csv = pd.read_csv(csv_address) 
        self.global_feats_csv = pd.read_csv(global_feats_csv)

        # Split smiles, labels, and masks columns as lists
        self.smiles = self.csv.iloc[:,1]
        self.labels = self.csv.iloc[:,2]
        self.masks = torch.ones((len(self.smiles), 1))

        self.global_feats = self.global_feats_csv.iloc[:,1:].values

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()

