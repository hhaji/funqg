
import numpy as np
import pandas as pd
from rdkit import Chem
from dgllife.utils import featurizers as fs
from rdkit import Chem
from collections import namedtuple
from descriptastorus.descriptors import rdNormalizedDescriptors
from deepchem.data import Dataset
from deepchem.splits import Splitter
from rdkit import Chem
from random import Random
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

"""" Node and Edge Features """
def atom_is_in_ring_list_one_hot(atom, allowable_set=None, encode_unknown=False):
    list = [3, 4, 5]  # List denotes the size of cycles
    return [atom.IsInRing()]+[atom.IsInRingSize(i) for i in list]

def bond_is_in_ring_list_one_hot(bond, allowable_set=None, encode_unknown=False):
    list = [3, 4, 5]  # List denotes the size of cycles
    return [bond.IsInRing()]+[bond.IsInRingSize(i) for i in list]

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

class ElecAtomFeaturizer(fs.BaseAtomFeaturizer):
    def __init__(self, atom_data_field='elec'):
        super(ElecAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: lambda atom: element(atom.GetSymbol()).electronegativity(scale='pauling')})


"""" Functional Group by Ertl """
#  Original authors: Richard Hall and Guillaume Godin
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
# Richard hall 2017 # IFG main code # Guillaume Godin 2017 # refine output function
# astex_ifg: identify functional groups a la Ertl, J. Cheminform (2017) 9:36
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
        #print("grp-1",grp)
        merge(mol, marked, grp)
        #print("grp-2",grp)
        groups.append(grp)
        #print("group",groups)

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
# Reference: https://github.com/bp-kelley/descriptastorus/blob/master/descriptastorus/descriptors/DescriptorGenerator.py
# def create_descriptors(df: pd.DataFrame,
#                        mols_column_name: str,
#                        generator_names: list):
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
# References:
# https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py
# https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py

# Scaffold Splitting    
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

          # seed = self.seed  # using instead the seed input of the split method
          
          np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

          train_size = frac_train * len(dataset)
          valid_size = frac_valid * len(dataset)
          test_size = frac_test * len(dataset)
          train_inds: List[int] = []
          valid_inds: List[int] = []
          test_inds: List[int] = []
          # train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

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

# Random Splitting 
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

# Random Stratified Splitting 
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


""" Settings """
def init_settings(name, generation_model, generation_row, split, generation_seeds, HQ_first_aggregation_op='mean'):
 
    global num_tasks, name_node_feature, name_data, task_type, kind, name_final_zip,\
            division, name_model_print, idx_row_graph_gen, kind1, node_feature_size, \
            edge_feature_size, name_model, global_size, add_complement, add_union, \
            list_seeds, current_dir, dataset_name, types_status, num_quotient, \
            num_layers_including_quotient, both_direction, universal_vertex

    # "Hierarchical_Quotient", "Quotient_complement", "Hierarchical_types_FGs"
    name_model = generation_model

    # Row of graph generator: Hierarchical_Quotient < 20, Quotient_complement < 6, Hierarchical_types_FGs < 16
    idx_row_graph_gen = generation_row

    node_feature_size = 127

    list_seeds = generation_seeds

    division = split 

    name_data = name 

    dataset_names = {
        
        "tox21" : "DatasetTox21",

        "bbbp" : "DatasetBBBP",

        "bace" : "DatasetBace",

        "toxcast" : "DatasetToxcast",

        "clintox" : "DatasetClintox",

        "sider" : "DatasetSider",

        "muv" : "DatasetMuv",

        "qm7" : "DatasetQm7",

        "qm8" : "DatasetQm8",

        "hiv": "DatasetHiv",
    
        "lipo": "DatasetLipo",

        "delaney" : "DatasetDelaney",

        "sampl" : "DatasetSampl",

        "amu" : "DatasetAmu",

        "ellinger" : "DatasetEllinger",
        
        "amu_ellinger" : "DatasetCovidAmu",

        "covid_amu" : "DatasetCovidAmu",

        "covid_ellinger" : "DatasetCovidEllinger",

        "covid_amu_ellinger" : "DatasetCovidAmuEllinger",

        "covid_amu_multitask" : "DatasetCovidMultitask",

        "covid_ellinger_multitask" : "DatasetCovidMultitask",
        
        "covid_amu_ellinger_multitask" : "DatasetCovidMultitask",

        "ionic" : "DatasetIonic"
    }

    dataset_name = dataset_names[name]

    current_dir = './'

    save_csv = current_dir + "data/graph/"

    """ Load csv of the specific graph generator"""
    graph_gen_csv = pd.read_csv(save_csv + name_model+".csv")
    row_graph_gen = graph_gen_csv.iloc[idx_row_graph_gen]

    if name_model== "Hierarchical_Quotient":
        types_status, num_quotient, num_layers_including_quotient, both_direction, universal_vertex = row_graph_gen[:-1]
        name_model_print = name_model+"_"+"type_"+str(types_status)
        kind1="Both_"+str(both_direction) +"_"+"Uni_Vert_"+str(universal_vertex) +"_"+"#quotient_"+str(num_quotient) +"_"+"#layers_"+str(num_layers_including_quotient)
        kind = name_model_print +"_"+ kind1
        # to resolve the confilict with mean-sum datasets
        if HQ_first_aggregation_op == 'sum':
            kind = kind + '_sum_aggregated'
            print('Caution (Hierarchical_Quotient): _sum_aggregated added after kind:', kind)

    if name_model== "Quotient_complement":
        types_status, num_quotient, add_complement = row_graph_gen[:-1]
        name_model_print = name_model+"_"+"type_"+str(types_status)
        kind1="#quotient_"+str(num_quotient) +"_"+"Complement_"+str(add_complement)
        kind = name_model_print +"_"+ kind1   

    if name_model== "Hierarchical_types_FGs":
        types_status, num_quotient, both_direction, universal_vertex = row_graph_gen[:-1]
        name_model_print = name_model+"_"+"type_"+str(types_status)
        kind1="#quotient_"+str(num_quotient) +"_Both_"+str(both_direction) +"_"+"Uni_Vert_"+str(universal_vertex)
        kind = name_model_print +"_"+ kind1

    name_node_feature="_"+str(node_feature_size)+"_one_hot"
    name_final = kind+name_node_feature
    name_final_zip = name_final+".zip"



