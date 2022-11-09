
import pandas as pd
import numpy as np
import torch
import sys
import dgl
import dgllife
import os
import networkx as nx
from rdkit import Chem
import pickle
import shutil
from rdkit.Chem import Recap
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch.utils.data import Subset
import zipfile

import utils_graph_generator 
from utils_graph_generator import create_descriptors, identify_functional_groups, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, init_settings, RandomScaffoldSplitter, RandomSplitter, RandomStratifiedSplitter_m

# [name_data_1, name_data_2, ...] like ['qm7', 'amu']
data_names = ['bbbp']
# {name_model : idx_row_graph_gens} like {'Hierarchical_Quotient' : [11]}
# generation_models_rows = {'Hierarchical_Quotient' : [11,10,1,0], "Quotient_complement" : [0]} 
generation_models_rows = {'Hierarchical_Quotient' : [10]} 
splits = ['random', 'scaffold']
generation_seeds = list(np.arange(2))
HQ_first_aggregation_op = 'mean'  # or 'sum'

current_dir = './'

# generating global features
for name in data_names:

    name_global_csv = name + '_global_cdf_rdkit.csv'
    name_global_zip = name + '_global_cdf_rdkit.zip'
    saving_adress = current_dir + 'data/global_features/' + name_global_zip

    if not os.path.exists(saving_adress):
        print('Generating Global Features for', name)
        raw_data_url = current_dir + 'data/raw/' + name + '.csv'
        data = pd.read_csv(raw_data_url)
        descriptors = create_descriptors(data, mols_column_name='smiles')
        compression_opts = dict(method='zip', archive_name=name_global_csv)
        os.makedirs(os.path.dirname(saving_adress), exist_ok=True)  
        descriptors.to_csv(saving_adress, index=False, compression=compression_opts)

# generating node features (for faster graph generation)
atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()
for name in data_names:

    saving_adress = current_dir + 'data/node_features/' + name + '_node_127_one_hot' + '.zip'
    saving_adress_pickle = current_dir + 'data/node_features/' + 'node_features.pickle'

    if not os.path.exists(saving_adress):
        print('Generating Node Features for', name)
        raw_data_url = current_dir + 'data/raw/' + name + '.csv'
        data = pd.read_csv(raw_data_url)
        node_features=[]
        for i in range(len(data.smiles)):
            node_features.append(atom_featurizer(Chem.MolFromSmiles(data.smiles[i]))['h'])
        os.makedirs(os.path.dirname(saving_adress), exist_ok=True)
        with open(saving_adress_pickle , 'wb') as handle:
            pickle.dump(node_features, handle)
        zf = zipfile.ZipFile(saving_adress, 'w', zipfile.ZIP_DEFLATED) 
        zf.write(saving_adress_pickle, 'node_features.pickle')  #archname is necessary to remove the path once unpacked
        zf.close()
        os.remove(saving_adress_pickle) 

# generating splits
scaffold_splitter = RandomScaffoldSplitter()
random_splitter = RandomSplitter()
stratified_splitter = RandomStratifiedSplitter_m()
splitters = [scaffold_splitter, random_splitter, stratified_splitter]
type_indexs = {'scaffold' : 0, 'random' : 1, 'stratified' : 2}
# splitting
for name in data_names:
    raw_data_url = current_dir + 'data/raw/' + name + '.csv'
    data = pd.read_csv(raw_data_url)
    for split in splits:
        for seed in generation_seeds:

            saving_adress = current_dir + 'data/splits/' + name + '/' + split + '_' + str(seed) + '/'

            if not os.path.exists(saving_adress + 'train_smiles') or not os.path.exists(saving_adress + 'val_smiles') or not os.path.exists(saving_adress + 'test_smiles'):
                print('Generating', split, '_', seed, 'split for', name)
                splitted_sets = splitters[type_indexs[split]].split(data, seed=seed)
                smiles_train = [data.smiles[i] for i in splitted_sets[0]]
                smiles_val = [data.smiles[i] for i in splitted_sets[1]]
                smiles_test =  [data.smiles[i] for i in splitted_sets[2]]
                os.makedirs(os.path.dirname(saving_adress), exist_ok=True)
                with open(saving_adress + 'train_smiles', 'wb') as handle:
                    pickle.dump(smiles_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(saving_adress + 'val_smiles', 'wb') as handle:
                    pickle.dump(smiles_val, handle, protocol=pickle.HIGHEST_PROTOCOL)      
                with open(saving_adress + 'test_smiles', 'wb') as handle:
                    pickle.dump(smiles_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


#generating graphs
for name in data_names:
    for generation_model in generation_models_rows.keys():
        for generation_row in generation_models_rows[generation_model]:

            ## using generated_featurized_graphs for all seeds 
            generated_featurized_graphs = {}
            are_graphs_generated = False

            for split in splits: 

                # specific settings for this scenario
                init_settings(name,generation_model,generation_row,split,generation_seeds,HQ_first_aggregation_op) 

                print( '\n\n\n\n ========================= \n  Generating', generation_model, 'with row number:', generation_row, 'graphs', \
                    'for', split, 'seeds:', utils_graph_generator.list_seeds, 'splitted',\
                         name, 'dataset is now started! \n ========================= \n\n\n\n')
                    
                ### Generation
                incorrect = False

                list_gen_seeds = []
                for seed in utils_graph_generator.list_seeds: 
                    path_save= utils_graph_generator.current_dir+"data/graph/"+utils_graph_generator.name_data+"/"+utils_graph_generator.division+"_"+str(seed)+"/"+utils_graph_generator.name_final_zip
                    if not os.path.exists(path_save):
                        save_results_status = True
                        list_gen_seeds.append(seed)

                        # Print the current seed
                        print("Seed ", seed, " is started!")

                        name_node_feature = utils_graph_generator.name_node_feature
                        name_final = utils_graph_generator.kind+name_node_feature
                        name_final_zip = name_final+".zip"
                        name_node_feats_zip = utils_graph_generator.name_data+ "_node"+name_node_feature+".zip"
                        name_global_csv = utils_graph_generator.name_data+"_global_cdf_rdkit.csv"
                        name_global_zip = utils_graph_generator.name_data+"_global_cdf_rdkit.zip"

                        """## Set Path"""
                        
                        folder_data_temp = current_dir +"data/buffer/" 
                        path_global_csv = folder_data_temp + name_global_csv       
                        path_save_current_dir = folder_data_temp + name_final + "/" 
                        path_save_temp = path_save_current_dir + utils_graph_generator.division + "_" + str(seed)

                        path_save_0= current_dir+"data/graph/"+utils_graph_generator.name_data
                        path_save= current_dir+"data/graph/"+utils_graph_generator.name_data+"/"+utils_graph_generator.division+"_"+str(seed)+"/"
                        # path_deepchem = current_dir+"data/deepchem/"

                        path_node_feats = current_dir + 'data/node_features/' 
                        path_node_feats_zip = path_node_feats + name_node_feats_zip
                        path_smiles = current_dir + 'data/splits/' + utils_graph_generator.name_data + "/" + utils_graph_generator.division+"_"+str(seed)+"/"
                        path_data_csv = current_dir + 'data/raw/' + utils_graph_generator.name_data + ".csv"
                        path_global_zip = current_dir + 'data/global_features/' + name_global_zip

                        ## Draw a Molecule and Find its Rings

                        # Draw molecule with atom index (see RDKitCB_0)
                        def mol_with_atom_index(mol):
                            for atom in mol.GetAtoms():
                                atom.SetAtomMapNum(atom.GetIdx())
                            return mol

                        def ring_idx(mol):
                            rings = mol.GetRingInfo()
                            return [list(ring) for ring in rings.AtomRings() if len(rings.AtomRings()) > 0]

                        def GetRingSystems(mol, includeSpiro=False):
                            ri = mol.GetRingInfo()
                            systems = []
                            for ring in ri.AtomRings():
                                ringAts = set(ring)
                                nSystems = []
                                for system in systems:
                                    nInCommon = len(ringAts.intersection(system))
                                    if nInCommon and (includeSpiro or nInCommon>1):
                                        ringAts = ringAts.union(system)
                                    else:
                                        nSystems.append(system)
                                nSystems.append(ringAts)
                                systems = nSystems
                            return systems

                        ### Find Linkers of a Molecule
                        def idx_atom_linker(mol):
                            lin=mol.GetSubstructMatches(Chem.MolFromSmarts('[r]!@[r]'))
                            return list(lin)

                        def idx_bonds_linker(mol):
                            patt = Chem.MolFromSmarts('[r]!@[r]')
                            hit_ats = list(mol.GetSubstructMatch(patt))
                            list_ids_bonds=[]
                            if hit_ats:
                                for bond in patt.GetBonds():
                                    aid1 = hit_ats[bond.GetBeginAtomIdx()]
                                    aid2 = hit_ats[bond.GetEndAtomIdx()]
                                    list_ids_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
                            return list_ids_bonds

                        ## To detect aromatic rings, I would loop over the bonds in each ring and
                        ## flag the ring as aromatic if all bonds are aromatic:
                        def isRingAromatic(mol, bondRing):
                            for id in bondRing:
                                if not mol.GetBondWithIdx(id).GetIsAromatic():
                                    return False
                            return True

                        def convert_bonds_atoms(mol, bond_indices):
                            return [mol.GetBondWithIdx(i).GetBeginAtomIdx() for i in bond_indices]

                        def aromatic_ring_idx(mol):
                            ri = mol.GetRingInfo()
                            return [convert_bonds_atoms(mol, bond_indices) for bond_indices in ri.BondRings() if isRingAromatic(mol, bond_indices)]


                        """## Electronegativity"""

                        def atom_finder(mol):
                            pt='[Cl,F,N,O,Br,I,S,s,n,o]'
                            pt=Chem.MolFromSmarts(pt)
                            return mol.GetSubstructMatches(pt)

                        """## Functional Groups and Their Types"""

                        def fg_idx(mol):
                            fgs = identify_functional_groups(mol)[0]
                            return [list(fgs[i][0]) for i in range(len(fgs)) if len(fgs)>0]

                        def fg_types_idx(mol):
                            fg_types = identify_functional_groups(mol)[1]
                            return fg_types

                        # carbonyl='[CX3]=[OX1]'
                        # hydroxyl = '[OX2H]'#'[OX2H;!$([CX3](=O)[OX2H1])]'#'[OX2H]'
                        # methyl = '[CX3][HX1]' # by me
                        # carboxyl ='[CX3](=O)[OX2H1]'
                        # amino = '[NX3;H2,H1;!$(NC=O)]'
                        # phosphate ='[#8]P([#8])([#8])=O' #by me
                        # sulfhydryl = '[#16]'  #by me
                        # list_functional_1=[methyl,  carboxyl, amino, phosphate, sulfhydryl]

                        # Ertel
                        a = '[*;R]-[#7](-[*;R])-[#6](-[*;R])=O'
                        b='[*;R]-[#8]-[*;R]'
                        c='[*;R]-[#7](-[*;R])-[*;R]'
                        d ='F[*;R]'        
                        e = 'Cl[*;R]'
                        f ='[*;R]-[#7]-[*;R]' 
                        h =  '[#8]-[#6](-[*;R])=O'
                        k= '[*;R]-[#7](-[*;R])S([*;R])(=O)=O'  
                        n= '[*;R]-[#8]-[#6](-[*;R])=O'
                        o= '[#6]=[#6]'   
                        p = '[*;R]-[#6](-[*;R])=O' 
                        q= '[*;R]-[#7](-[*;R])-[#6](=O)-[#7](-[*;R])-[*;R]'
                        r= '[*;R]-[#16]-[*;R]'
                        s= "C#N"
                        t= '[*;R]-[#8]-[#6](=O)-[#7](-[*;R])-[*;R]'    
                        u ='Br[*;R]'   
                        v = '[*;R]S([*;R])(=O)=O'   
                        w ='[*;R]\[#7]=[#6](\[#7](-[*;R])-[*;R])-[#7](-[*;R])-[*;R]'     
                        x ='[*;R]N(=O)=O'
                        y ="C#C"
                        z ='[#8]-[#7](-[*;R])-[#6](-[*;R])=O'
                        aa='[*;R]-[#8]-[#6]-[#8]-[*;R]'
                        ab  ='[*;R]-[#7]=[#6]-[#7](-[*;R])-[*;R]'
                        ac  ='[*;R]-[#7](-[*;R])-[#6](=O)-[#6]=[#6;A]'
                        ae  ='I[*;R]'
                        af  ='[#16]-[*;R]'
                        ag  ='[*;R]-[#16]-[#16]-[*;R]'
                        ah ='[*;R]-[#6](=O)-[#6]=[#6;A]'
                        ai ='[*;R]-[#7](-[#6](-[*;R])=O)S([*;R])(=O)=O'  
                        aj ='[#8]P([#8])([*;R])=O'
                        ak ='[*;R]-[#7]=[#6;A]' 
                        am ='[*;R]-[#7](-[*;R])-[#6](=O)-[#6](-[*;R])=O'   
                        an ='[*;R]-[#8]-[#6](=O)-[#6]=[#6;A]'
                        ao  ='[*;R]-[#7](-[#6](-[*;R])=O)-[#6](-[*;R])=O'
                        ap ='[*;R]-[#8]-[#7]=[#6;A]'
                        aq  ='[*;R]-[#6]=O'
                        ar ='[*;R][N+]([*;R])([*;R])[*;R]' 
                        asas ='[#8]P([#8])(=O)[#8]-[*;R]'
                        av  ='[*;R]-[#7](-[#7]=[#6;A])-[#6](-[*;R])=O'
                        aw ='[#8]-[#6](=O)-[#6]=[#6]'
                        ax ='[*;R]-[#7](-[*;R])-[#7](-[*;R])-[#6](-[*;R])=O'
                        list_functional =[a,b,c,d,e,f,h,k,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ae,af,ag,ah,ai,aj,ak,am,an,ao,ap,aq,ar,ar,asas,av,aw,ax]

                        list_impo_functional_groups=[]
                        def identify_impo_functional_groups(mol):
                            for i in list_functional:
                                patt = Chem.MolFromSmarts(i)
                                list_impo_functional_groups.append([list(idx) for idx in mol.GetSubstructMatches(patt)])
                            return list_impo_functional_groups


                        """### Fragments"""

                        def fragments_idx(mol):
                            res = Recap.RecapDecompose(mol,minFragmentSize=1)
                            frag= res.children.keys()
                            list_fragments=[]
                            for member in frag:
                                if member.find('/',1,2) == True:
                                    list_fragments.append(list(mol.GetSubstructMatch(Chem.MolFromSmarts(member))))
                                else: #without / at [1,2]
                                    member=member.replace('*','')
                                    list_fragments.append(list(mol.GetSubstructMatch(Chem.MolFromSmarts(member))))
                            return list_fragments

                        smiles="O=Cc1ccc(O)c(OC)c1COc1cc(C=O)ccc1O"
                        mol=Chem.MolFromSmiles(smiles)
                        fragments_idx(mol)!= []

                        scaffold = MurckoScaffoldSmiles(smiles, includeChirality=False)
                        mol=Chem.MolFromSmiles(scaffold)
                        mol_with_atom_index(mol)

                        def scaffold_idx(mol):
                            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                            scaffold_ids = mol.GetSubstructMatches(Chem.MolFromSmiles(scaffold))
                            return [list(i) for i in scaffold_ids if scaffold_ids!=((),)]


                        """## Molecule DataFrame  
                        Construct a DataFrame Containing Some Properties of a Molecule

                        """

                        def convert_mol(smiles):
                            return Chem.MolFromSmiles(smiles)

                        def make_df(url):
                            data = pd.read_csv(url)
                            df = pd.DataFrame()
                            df["Smiles"] = data["smiles"]
                            #   df["Rings"] = data["smiles"].apply(convert_mol).apply(ring_idx)
                            #   df["Ring_Systems"] = data["smiles"].apply(convert_mol).apply(GetRingSystems)
                            #   df["Aromatic_Rings"] = data["smiles"].apply(convert_mol).apply(aromatic_ring_idx)
                            #   df["Scaffolds"] = data["smiles"].apply(convert_mol).apply(scaffold_idx)
                            df["FGs"] = data["smiles"].apply(convert_mol).apply(fg_idx)
                            df["FG_Types"] = data["smiles"].apply(convert_mol).apply(fg_types_idx)
                            return df

                        # Extract rings, functional groups, etc. 

                        df = make_df(path_data_csv)

                        """## Graphs

                        ### Connections

                        #### Here are some codes to determine bonds between some parts of a molecule
                        """

                        ################################################### Some functions ###################################################
                        def fg_node_neighbors(mol_dgl_graph, fg):
                            lists_fg_neighbors = [mol_dgl_graph.predecessors(node_id).tolist() + mol_dgl_graph.successors(node_id).tolist() for node_id in fg]
                            set_all_fg_neighbors = set([neighbor for sublist in lists_fg_neighbors for neighbor in sublist]) # May contain fg nodes
                            return list(set_all_fg_neighbors-set(fg))

                        def type_minus_fg(types, fg):
                            return list(set(types)-set(fg))

                        """#### Here we determine the bonds between types and the other nodes"""

                        # def types_connections_idx(smiles, mol_dgl_graph):
                        #     mol_dgl_graph.edata["connections_types"] = torch.ones(mol_dgl_graph.num_edges(), 1)
                        #     mol_dgl_graph.ndata["nodes_non_types"] = torch.ones(mol_dgl_graph.num_nodes(), 1)
                        #     nodes_fg_types =[]
                        #     if df[df["Smiles"] == smiles]["FG_Types"].squeeze() != []:
                        #         for fg_types in df[df["Smiles"] == smiles]["FG_Types"].squeeze():
                        #             mol_dgl_sub_graph=dgl.node_subgraph(mol_dgl_graph, fg_types)
                        #             mol_dgl_graph.edata["connections_types"][mol_dgl_sub_graph.edata[dgl.EID].long()]=torch.zeros(len(mol_dgl_sub_graph.edata[dgl.EID]), 1)
                        #             mol_dgl_graph.ndata["nodes_non_types"][mol_dgl_sub_graph.ndata[dgl.NID].long()]= torch.zeros(len(mol_dgl_sub_graph.ndata[dgl.NID]), 1)
                        #             nodes_fg_types += fg_types
                        #         nodes_non_types=[node for node in range(mol_dgl_graph.num_nodes()) if node not in nodes_fg_types]
                        #         mol_dgl_sub_graph=dgl.node_subgraph(mol_dgl_graph, nodes_non_types)
                        #         mol_dgl_graph.edata["connections_types"][mol_dgl_sub_graph.edata[dgl.EID].long()]=torch.zeros(len(mol_dgl_sub_graph.edata[dgl.EID]), 1)
                        #     else:
                        #         mol_dgl_graph.edata["connections_types"] = torch.zeros(mol_dgl_graph.num_edges(), 1)
                        #     return mol_dgl_graph.edata["connections_types"]

                        def types_connections_idx(smiles, mol_dgl_graph):
                            mol_dgl_graph.edata["edges_fgs"] = torch.zeros(mol_dgl_graph.num_edges(), 1)
                            mol_dgl_graph.edata["edges_non_fgs"] = torch.zeros(mol_dgl_graph.num_edges(), 1)
                            nodes_fgs =[]
                            if df[df["Smiles"] == smiles]["FG_Types"].squeeze() != []:
                                for fgs in df[df["Smiles"] == smiles]["FG_Types"].squeeze():
                                    mol_dgl_sub_graph=dgl.node_subgraph(mol_dgl_graph, fgs)
                                    mol_dgl_graph.edata["edges_fgs"][mol_dgl_sub_graph.edata[dgl.EID].long()]=torch.ones(len(mol_dgl_sub_graph.edata[dgl.EID]), 1)
                                    nodes_fgs += fgs
                                nodes_non_fgs=[node for node in range(mol_dgl_graph.num_nodes()) if node not in nodes_fgs]
                                mol_dgl_sub_graph=dgl.node_subgraph(mol_dgl_graph, nodes_non_fgs)
                                mol_dgl_graph.edata["edges_non_fgs"][mol_dgl_sub_graph.edata[dgl.EID].long()]=torch.ones(len(mol_dgl_sub_graph.edata[dgl.EID]), 1)
                            else:
                                mol_dgl_graph.edata["edges_non_fgs"] = torch.ones(mol_dgl_graph.num_edges(), 1)
                            return mol_dgl_graph.edata["edges_non_fgs"], mol_dgl_graph.edata["edges_fgs"]
                        def fgs_connections_idx(smiles, mol_dgl_graph):
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

                        # for i in range(fg_type_df.shape[0]):
                        #     mol = Chem.MolFromSmiles(fg_type_df['Smiles'][i])
                        #     mol_dgl_graph = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)
                        #     types = fg_type_df['Type'][i]
                        #     fg = fg_type_df['FGs'][i]
                        #     for j in range(len(fg_type_df['FGs'][i])):
                        #         if sorted(fg_node_neighbors(mol_dgl_graph, fg[j])) !=  sorted(type_minus_fg(types[j], fg[j])):
                        #           print(fg_node_neighbors(mol_dgl_graph, fg[j]), type_minus_fg(types[j], fg[j]))

                        """## Construct a Graph From a Molecule

                        ### Graph of Functional Groups and Rings
                        """

                        # ################################################### smiles to graph ###################################################

                        # def add_fgs_vertices_edges(smiles, mol_dgl_graph, fg_empty_out2in=True, fg_empty_in2out=True, fg_out2in=True, fg_in2out=True):
                        #     '''
                        #     Add nodes and edges related to FGs and their neighbors (together or seperately).
                        #     '''
                        #     if df[df['Smiles'] == smiles]['FGs'].squeeze() != []:
                        #         for fg in df[df['Smiles'] == smiles]['FGs'].squeeze():
                        #             mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #             if fg_in2out:
                        #                 src_neighbors = fg_node_neighbors(mol_dgl_graph, fg)
                        #                 mol_dgl_graph.add_edges(src_neighbors, [mol_dgl_graph_num_nodes])
                        #                 mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], src_neighbors)
                        #             if fg_out2in:
                        #                 dst_neighbors = fg
                        #                 mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], dst_neighbors)
                        #                 mol_dgl_graph.add_edges(dst_neighbors, [mol_dgl_graph_num_nodes])
                        #     else:
                        #         mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #         if fg_empty_in2out:
                        #             mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #         if fg_empty_out2in:
                        #             mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                        #     # mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #     # mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #     # mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))

                        #     '''
                        #     Add nodes and edges related to FGs.
                        #     '''
                        #     # if df[df['Smiles'] == smiles]['FGs'].squeeze() != []:
                        #     #     for fg in df[df['Smiles'] == smiles]['FGs'].squeeze():
                        #     #         mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #     #         if fg_in2out:
                        #     #             src_neighbors = fg
                        #     #             mol_dgl_graph.add_edges(src_neighbors, [mol_dgl_graph_num_nodes])
                        #     #         if fg_out2in:
                        #     #             dst_neighbors = fg
                        #     #             mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], dst_neighbors)
                        #     # else:
                        #     #     mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #     #     if fg_empty_in2out:
                        #     #         mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #     #     if fg_empty_out2in:
                        #     #         mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                        #     # mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #     # mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #     # mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                            
                        #     return mol_dgl_graph


                        # def add_rings_vertices_edges(smiles, mol_dgl_graph, ring_empty_out2in=True, ring_empty_in2out=True, ring_out2in=True, ring_in2out=True):
                        #     '''
                        #     Add nodes and edges related to FGs.
                        #     '''
                        #     if df[df['Smiles'] == smiles]['Ring'].squeeze() != []:
                        #         for ring in df[df['Smiles'] == smiles]['Ring'].squeeze():
                        #             mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #             if ring_in2out:
                        #                 src_neighbors = ring
                        #                 mol_dgl_graph.add_edges(src_neighbors, [mol_dgl_graph_num_nodes])
                        #             if ring_out2in:
                        #                 dst_neighbors = ring
                        #                 mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], dst_neighbors)
                        #     else:
                        #         mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #         # if ring_empty_in2out and fg == False:
                        #         if ring_empty_in2out:

                        #             mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #         # if ring_empty_out2in and fg == False:
                        #         if ring_empty_out2in:
                        #             mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                        #     return mol_dgl_graph

                        
                        # def mol_fg_ring_graph(smiles):
                        #     mol = Chem.MolFromSmiles(smiles)
                        #     mol_dgl_graph = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)
                        #     mol_dgl_graph_num_nodes_init = mol_dgl_graph.num_nodes()

                        #     ## Add FG Nodes
                        #     mol_dgl_graph = add_fgs_vertices_edges(smiles, mol_dgl_graph, fg_empty_out2in=True, fg_empty_in2out=True, fg_out2in=True, fg_in2out=True)
                        #     id_nodes_readout_fg = list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph.num_nodes()))
                        #     mol_dgl_graph_num_nodes_after_fg = mol_dgl_graph.num_nodes()
                            
                        #     ## Add Ring Nodes
                        #     # mol_dgl_graph = add_rings_vertices_edges(smiles, mol_dgl_graph, fg=True, ring_empty_out2in=True, ring_empty_in2out=True, ring_out2in=True, ring_in2out=True)
                        #     # id_nodes_readout_ring = list(range(mol_dgl_graph_num_nodes_after_fg, mol_dgl_graph.num_nodes()))

                        #     mol_dgl_graph.ndata["FGs_new_nodes"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                        #     # mol_dgl_graph.ndata["Rings"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                        #     if id_nodes_readout_fg != []:
                        #         mol_dgl_graph.ndata["FGs_new_nodes"][id_nodes_readout_fg] = torch.ones(len(id_nodes_readout_fg), 1)
                        #     # if id_nodes_readout_ring != []:
                        #         # mol_dgl_graph.ndata["Rings"][id_nodes_readout_ring] = torch.ones(len(id_nodes_readout_ring), 1)
                        #     mol_dgl_graph.ndata["FGs_new_nodes"]=mol_dgl_graph.ndata["FGs_new_nodes"].view(-1,1).to(dtype=torch.float32) 
                        #     # mol_dgl_graph.ndata["Rings"]=mol_dgl_graph.ndata["Rings"].view(-1,1).to(dtype=torch.float32)
                        #     # mol_dgl_graph.ndata["r"]= mol_dgl_graph.ndata["FGs"] + mol_dgl_graph.ndata["Rings"]

                        #     # # Add readout nodes
                        #     # if fg == True:
                        #     #     mol_dgl_graph = add_fgs_vertices_edges(smiles, mol_dgl_graph, empty_out2in=True, empty_in2out=True, fg_out2in=True, fg_in2out=True)
                        #     # if ring == True:
                        #     #     mol_dgl_graph = add_rings_vertices_edges(smiles, mol_dgl_graph, fg=True, empty_out2in=True, empty_in2out=True, ring_out2in=True, ring_in2out=True)
                            
                        #     # id_nodes_readout = list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph.num_nodes()))
                        #     # mol_dgl_graph.ndata['r']= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                        #     # mol_dgl_graph.ndata['r'][id_nodes_readout] = torch.ones(len(id_nodes_readout))[0] 

                        #     return mol_dgl_graph

                        ### Graph of Functional Groups 
                        def add_fgs_vertices_edges(smiles, mol_dgl_graph, fg_empty_out2in=True, fg_empty_in2out=True, fg_out2in=True, fg_in2out=True):
                        # def add_fgs_vertices_edges(smiles, mol_dgl_graph, fg_empty_out2in=False, fg_empty_in2out=True, fg_neighbor_out2in=True, fg_neighbor_in2out=True):
                            # if df[df["Smiles"] == smiles]["FGs"].squeeze() != []:
                            #     for fg in df[df["Smiles"] == smiles]["FGs"].squeeze():
                            #         if fg_neighbor_in2out:
                            #             src_neighbors = fg_node_neighbors(mol_dgl_graph, fg)
                            #             mol_dgl_graph.add_edges(src_neighbors, [mol_dgl_graph.num_nodes()])
                            #         if fg_neighbor_out2in:
                            #             dst_neighbors = fg
                            #             mol_dgl_graph.add_edges([mol_dgl_graph.num_nodes()], dst_neighbors)
                            if df[df["Smiles"] == smiles]["FGs"].squeeze() != []:
                                for fg in df[df["Smiles"] == smiles]["FGs"].squeeze():
                                    mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                                    if fg_in2out:
                                        mol_dgl_graph.add_edges(fg, [mol_dgl_graph_num_nodes])
                                    if fg_out2in:
                                        mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], fg)
                            else:
                                mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                                if fg_empty_in2out:
                                    mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                                if fg_empty_out2in:
                                    mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                            return mol_dgl_graph


                        def add_rings_vertices_edges(smiles, mol_dgl_graph, fg=True, ring_empty_out2in=True, ring_empty_in2out=True, ring_out2in=True, ring_in2out=True):
                            if df[df["Smiles"] == smiles]["Rings"].squeeze() != []:
                                for ring in df[df["Smiles"] == smiles]["Rings"].squeeze():
                                    mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                                    if ring_in2out:
                                        mol_dgl_graph.add_edges(ring, [mol_dgl_graph_num_nodes])
                                    if ring_out2in:
                                        mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], ring)
                            else:
                                mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                                if ring_empty_in2out and fg == False:
                                    mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes), [mol_dgl_graph_num_nodes]))
                                if ring_empty_out2in and fg == False:
                                    mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                            return mol_dgl_graph


                        def mol_fg_ring_graph(smiles):
                            mol = Chem.MolFromSmiles(smiles)
                            mol_dgl_graph = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)

                            ## Add readout nodes
                            mol_dgl_graph_num_nodes_init = mol_dgl_graph.num_nodes()
                            
                            ## Add Functionl Nodes
                            mol_dgl_graph = add_fgs_vertices_edges(smiles, mol_dgl_graph, fg_empty_out2in=True, fg_empty_in2out=True, fg_out2in=True, fg_in2out=True)
                            id_nodes_readout_fg = list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph.num_nodes()))
                            mol_dgl_graph_num_nodes_after_fg = mol_dgl_graph.num_nodes()
                            
                            ## Add Ring Nodes
                            # mol_dgl_graph = add_rings_vertices_edges(smiles, mol_dgl_graph, fg=True, ring_empty_out2in=True, ring_empty_in2out=True, ring_out2in=True, ring_in2out=True)
                            # id_nodes_readout_ring = list(range(mol_dgl_graph_num_nodes_after_fg, mol_dgl_graph.num_nodes()))

                            mol_dgl_graph.ndata["FGs_new_nodes"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                            # mol_dgl_graph.ndata["Rings"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                            if id_nodes_readout_fg != []:
                                mol_dgl_graph.ndata["FGs_new_nodes"][id_nodes_readout_fg] = torch.ones(len(id_nodes_readout_fg), 1)
                            # if id_nodes_readout_ring != []:
                                # mol_dgl_graph.ndata["Rings"][id_nodes_readout_ring] = torch.ones(len(id_nodes_readout_ring), 1)
                            mol_dgl_graph.ndata["FGs_new_nodes"]=mol_dgl_graph.ndata["FGs_new_nodes"].view(-1,1).to(dtype=torch.float32) 
                            # mol_dgl_graph.ndata["Rings"]=mol_dgl_graph.ndata["Rings"].view(-1,1).to(dtype=torch.float32)
                            # mol_dgl_graph.ndata["r"]= mol_dgl_graph.ndata["FGs"] + mol_dgl_graph.ndata["Rings"]
                            
                            return mol_dgl_graph
                        ############################################################

                        """### Graph of Types and Aromatic Rings """

                        # def add_types_vertices_edges(smiles, mol_dgl_graph, types_empty_out2in=True, types_empty_in2out=True, types_out2in=True, types_in2out=True):
                        #     if df[df['Smiles'] == smiles]['FGs'].squeeze() != []:
                        #         for types in df[df['Smiles'] == smiles]['Type'].squeeze():
                        #             mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #             if fg_in2out:
                        #                 src_neighbors = types
                        #                 mol_dgl_graph.add_edges(src_neighbors, [mol_dgl_graph_num_nodes])
                        #             if fg_out2in:
                        #                 dst_neighbors = types
                        #                 mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], dst_neighbors)
                        #     # else:
                        #     #     mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #     #     if empty_in2out:
                        #     #         mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #     #     if empty_out2in:
                        #     #         mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))                  
                        #     mol_dgl_graph_num_nodes = mol_dgl_graph.num_nodes()
                        #     mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                        #     mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes)))
                            
                        #     return mol_dgl_graph


                        # def mol_fg_types_graph_v1(smiles):
                        #     mol = Chem.MolFromSmiles(smiles)
                        #     mol_dgl_graph = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)
                        #     mol_dgl_graph_num_nodes_init = mol_dgl_graph.num_nodes()

                        #     mol_dgl_graph = add_types_vertices_edges(smiles, mol_dgl_graph, types_empty_out2in=True, types_empty_in2out=True, types_out2in=True, types_in2out=True)
                            
                        #     id_nodes_readout = list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph.num_nodes()))
                        #     mol_dgl_graph.ndata["types_new_nodes"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                        #     mol_dgl_graph.ndata["types_new_nodes"][id_nodes_readout] = torch.ones(len(id_nodes_readout))[0] 

                        #     return mol_dgl_graph

                        ######## Type and Aromatic Ring Graph
                        def graph_constructor(smiles, types_in2out=True, types_out2in=True, fragment=True):
                            mol = Chem.MolFromSmiles(smiles)
                            mol_dgl_graph = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False).int()
                            # mol_dgl_graph_num_nodes_init = mol_dgl_graph.num_nodes()
                            ######################
                            ### Determine edges among Functional Groups and among Non-Functional Groups (both vertices)
                            if utils_graph_generator.types_status ==True:
                                mol_dgl_graph.edata["edges_non_fgs"], mol_dgl_graph.edata["edges_fgs"] = types_connections_idx(smiles, mol_dgl_graph)
                            else:
                                mol_dgl_graph.edata["edges_non_fgs"], mol_dgl_graph.edata["edges_fgs"] = fgs_connections_idx(smiles, mol_dgl_graph)
                            ######################
                            ######################
                            # mol_dgl_graph.edata["bonds_linker"]= torch.zeros(mol_dgl_graph.num_edges(), 1)
                            # if idx_bonds_linker(mol) != []:
                            #     mol_dgl_graph.edata["bonds_linker"][idx_bonds_linker(mol)]=torch.ones(len(idx_bonds_linker(mol)), 1)
                            # ######################
                            # mol_dgl_graph.edata["connections_types"]=types_connections_idx(smiles, mol_dgl_graph)

                            #################################################################################################
                            #################################################################################################
                            ###    Types   Types   or Functonl Groups   Functonl Groups   
                            # mol_dgl_graph.ndata["idx_nodes_types"]= torch.ones(mol_dgl_graph.num_nodes(), 1)
                            # mol_dgl_graph.edata["idx_edges_types"]= torch.tensor(list(range(1,1+mol_dgl_graph.num_edges()))).float().view(-1,1)

                            # ### First we specify idx nodes in types or functional groups
                            # #### FGs:
                            # count=mol_dgl_graph.num_nodes()+1
                            # all_nodes_types=[]
                            # for types in identify_functional_groups(mol)[0]:
                            #     all_nodes_types += types
                            #     types=list(types[0])
                            #     mol_dgl_graph.ndata["idx_nodes_types"][types] = count
                            #     count+=1 
                            #################################################################################################
                            #################################################################################################
                            ################## Find non type edges:
                            # mol_dgl_graph.edata["idx_edges_non_types"]= torch.zeros(mol_dgl_graph.num_edges(), 1)
                            # # id_new_edge=mol_dgl_graph.edge_ids(torch.tensor([mol_dgl_graph.num_nodes()-1]), torch.tensor([mol_dgl_graph.num_nodes()])) 
                            # for i in range(mol_dgl_graph.num_edges()):
                            #     src, dst= mol_dgl_graph.find_edges(torch.tensor([i]).to(dtype=torch.int32))
                            #     if src in all_nodes_types or dst in all_nodes_types:
                            #         pass
                            #     else:
                            #         mol_dgl_graph.edata["idx_edges_non_types"][i]=1
                            #################################################################################################
                            #################################################################################################
                            ########## Important Functional Groups:
                            # mol_dgl_graph.ndata["idx_important_FGs"]= torch.ones(mol_dgl_graph.num_nodes(), 1)
                            # count= 2
                            # num_FGs = 42
                            # for i in range(num_FGs):
                            #     for fgs in identify_impo_functional_groups(mol)[i]:
                            #         print(fgs)
                            #         mol_dgl_graph.ndata["idx_important_FGs"][fgs] = count
                            #     count+=1       
                            #################################################################################################
                            #################################################################################################
                            # ### Types:
                            # count=mol_dgl_graph.num_nodes()
                            # for types in identify_functional_groups(mol)[1]:
                            #     # types=list(types[0])
                            #     mol_dgl_graph.ndata["idx_nodes_types"][types] = count
                            #     count+=1  
                            # ## Fragments:
                            # count=mol_dgl_graph.num_nodes()
                            # for types in fragments_idx(mol):
                            #     # types=list(types[0])
                            #     mol_dgl_graph.ndata["idx_nodes_types"][types] = count
                            #     count+=1  
                            #     #### Add a new vertex for any type:
                            #     mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                            #     if types_in2out:
                            #         mol_dgl_graph.add_edges(types, [mol_dgl_graph_num_nodes])
                            #     if types_out2in:
                            #         mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], types)

                                #####################################    
                                ##### Begining: Connect New Nodes of Types (Complete Graph)
                                # mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph_num_nodes)))
                                # mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph_num_nodes)), [mol_dgl_graph_num_nodes])
                            # mol_dgl_graph=dgl.remove_self_loop(mol_dgl_graph)
                                ##### End: Connect New Nodes of Types (Complete Graph)
                            #### Readout Vertices
                            ### Consider New Vertices of Types as Readout
                            # mol_dgl_graph.ndata["types_new_nodes"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                            # idx=list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph.num_nodes()))
                            # mol_dgl_graph.ndata["types_new_nodes"][idx]=torch.ones(len(idx), 1)
                            #############################################
                            #################################################################################################
                            ##### Add a Universal Vertex
                            # mol_dgl_graph_num_nodes_updated = mol_dgl_graph.num_nodes()
                            # mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes_init)), [mol_dgl_graph_num_nodes_updated])
                            # mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes_updated], list(range(mol_dgl_graph_num_nodes_init)))

                            ### Consider the former vertices of the graph as Readout
                            # mol_dgl_graph.ndata["types_former_nodes"]= torch.ones(mol_dgl_graph.num_nodes(), 1)
                            # idx=list(range(mol_dgl_graph_num_nodes_init, mol_dgl_graph.num_nodes()))
                            # mol_dgl_graph.ndata["types_former_nodes"][idx]=torch.zeros(len(idx), 1)

                            #################################################################################################
                            #################################################################################################
                            # ###   Fragment   Fragment   Fragment    Fragment
                            # mol_dgl_graph_num_nodes_after_type = mol_dgl_graph.num_nodes()
                            # if fragment and fragments_idx(mol) != []:
                            #     ### Join  a new vertex to any type
                            #     for frag in fragments_idx(mol):
                            #         mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                            #         if types_in2out:
                            #             mol_dgl_graph.add_edges(frag, [mol_dgl_graph_num_nodes])
                            #         if types_out2in:
                            #             mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], frag)

                            # #### Readout Vertices
                            # ### Consider New Vertices of Fragments as Readout
                            # mol_dgl_graph.ndata["fragment_new_nodes"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                            # idx=list(range(mol_dgl_graph_num_nodes_after_type, mol_dgl_graph.num_nodes()))
                            # mol_dgl_graph.ndata["fragment_new_nodes"][idx]=torch.ones(len(idx), 1)

                            #################################################################################################
                            #################################################################################################
                            ###   Ring_Systems  Ring_Systems    Ring_Systems
                            # mol_dgl_graph_num_nodes_after_type_fragment = mol_dgl_graph.num_nodes()

                            # if df[df["Smiles"] == smiles]["Ring_Systems"].squeeze() != []:
                            #     for ring_sys in df[df["Smiles"] == smiles]["Ring_Systems"].squeeze():
                            # ### Join  a new vertex to any type
                            #         mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                            #         if types_in2out:
                            #             mol_dgl_graph.add_edges(ring_sys, [mol_dgl_graph_num_nodes])
                            #         if types_out2in:
                            #             mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], ring_sys)

                            # #### Readout Vertices
                            # ### Consider New Vertices of Ring System as Readout
                            # mol_dgl_graph.ndata["Ring_Systems"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                            # idx=list(range(mol_dgl_graph_num_nodes_after_type_fragment, mol_dgl_graph.num_nodes()))
                            # mol_dgl_graph.ndata["Ring_Systems"][idx]=torch.ones(len(idx), 1)

                            #################################################################################################
                            #################################################################################################
                            ### Add a Node for any Aromatic Ring
                            # mol_dgl_graph_num_nodes_before_aromatic = mol_dgl_graph.num_nodes()
                            # for aromatic_ring in aromatic_ring_idx(mol):
                            #     mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                            #     if types_in2out:
                            #         mol_dgl_graph.add_edges(aromatic_ring, [mol_dgl_graph_num_nodes])
                            #     if types_out2in:
                            #         mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], aromatic_ring)

                            # #### Readout Vertices
                            # ### Consider New Vertices of Aromatic Ring as Readout
                            # mol_dgl_graph.ndata["Aromatic_Ring"]= torch.zeros(mol_dgl_graph.num_nodes(), 1)
                            # idx=list(range(mol_dgl_graph_num_nodes_before_aromatic, mol_dgl_graph.num_nodes()))
                            # mol_dgl_graph.ndata["Aromatic_Ring"][idx]=torch.ones(len(idx), 1)

                            #################################################################################################
                            #################################################################################################
                            #### Add a Node for any Functional Group   
                            # if identify_functional_groups(mol)[1] != []:
                            #     mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes_updated], list(range(mol_dgl_graph_num_nodes_updated)))
                            #     mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes_updated)), [mol_dgl_graph_num_nodes_updated])
                            # if identify_functional_groups(mol)[1]== []:
                            #     if aromatic_ring_idx(smiles) == []:
                            #         mol_dgl_graph.add_edges(list(range(mol_dgl_graph_num_nodes_init)), [mol_dgl_graph_num_nodes_init])
                            #         mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes_init], list(range(mol_dgl_graph_num_nodes_init)))
                            #################################################################################################
                            #################################################################################################
                            #### Add a Feature for Scaffold  
                            # mol_dgl_graph.ndata["scaffold_nodes"]= torch.zeros(mol_dgl_graph_num_nodes, 1)
                            # id_scaffold_nodes = df[df['Smiles'] == smiles]['Scaffolds'].squeeze()
                            # if id_scaffold_nodes != []:
                            #     mol_dgl_graph.ndata["scaffold_nodes"][id_scaffold_nodes] = torch.ones(len(id_scaffold_nodes), 1)
                            # else:
                            #     mol_dgl_graph.ndata["scaffold_nodes"] = torch.ones(mol_dgl_graph_num_nodes, 1)

                            return mol_dgl_graph

                        """#### Hierarchical graph"""

                        # def mol_hierarchi_graph(smiles):
                        #     mol = Chem.MolFromSmiles(smiles)
                        #     mol_dgl_graph_layer1 = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)
                        #     mol_dgl_graph_layer1 = add_features(smiles, mol_dgl_graph_layer1)
                        #     mol_dgl_graph_layer1_num_nodes = mol_dgl_graph_layer1.num_nodes()
                        #     mol_dgl_graph_layer1.ndata[new_feature_name]= torch.zeros(mol_dgl_graph_layer1_num_nodes, 1)

                        #     ## Construct Layer2 graph
                        #     mol_dgl_graph_layer2 = eval(graph_constructor_init)(smiles)
                        #     mol_dgl_graph_layer2 = add_features(smiles, mol_dgl_graph_layer2)

                        #     hierarchi_graph = dgl.batch([mol_dgl_graph_layer1, mol_dgl_graph_layer2])

                        #     if df[df['Smiles'] == smiles]['FGs'].squeeze() != []:
                        #         for fg in df[df['Smiles'] == smiles]['FGs'].squeeze():
                        #             for fg_one in fg:
                        #                 hierarchi_graph.add_edges([mol_dgl_graph_layer1_num_nodes+fg_one], [fg_one])
                        #                 hierarchi_graph.add_edges([fg_one], [mol_dgl_graph_layer1_num_nodes+fg_one])
                                        
                        #                 fg_one_neighbors = fg_node_neighbors(mol_dgl_graph_layer1, [fg_one])
                        #                 hierarchi_graph.add_edges(fg_one_neighbors, [mol_dgl_graph_layer1_num_nodes+fg_one])
                        #                 hierarchi_graph.add_edges([mol_dgl_graph_layer1_num_nodes+fg_one], fg_one_neighbors)
                        #     else:
                        #         for node in range(mol_dgl_graph_layer1_num_nodes):
                        #             hierarchi_graph.add_edges([mol_dgl_graph_layer1_num_nodes+node], [node])
                        #             hierarchi_graph.add_edges([node], [mol_dgl_graph_layer1_num_nodes+node])

                        #     return hierarchi_graph

                        # def mol_hierarchi_graph_batch(smiles):
                        #     mol = Chem.MolFromSmiles(smiles)
                        #     mol_dgl_graph_layer1 = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)
                        #     mol_dgl_graph_layer1 = add_features(smiles, mol_dgl_graph_layer1)
                        #     mol_dgl_graph_layer1_num_nodes = mol_dgl_graph_layer1.num_nodes()
                        #     mol_dgl_graph_layer1.ndata[new_feature_name]= torch.zeros(mol_dgl_graph_layer1_num_nodes, 1)

                        #     ## Construct Layer2 graph
                        #     mol_dgl_graph_layer2 = eval(graph_constructor_init)(smiles)
                        #     mol_dgl_graph_layer2 = add_features(smiles, mol_dgl_graph_layer2)

                        #     hierarchi_graph = dgl.batch([mol_dgl_graph_layer1, mol_dgl_graph_layer2])

                        #     return hierarchi_graph

                        # smiles="COc1ccc2c(c1OC)CN1CCc3cc4c(cc3C1C2)OCO4"
                        # mol_hierarchi_graph_batch(smiles)

                        # dgl.unbatch(mol_hierarchi_graph_batch(smiles))

                        # def mol_hierarchi_graph_batch_layer1(smiles):
                        #     mol = Chem.MolFromSmiles(smiles)
                        #     mol_dgl_graph_layer1 = dgllife.utils.mol_to_bigraph(mol, canonical_atom_order=False)
                        #     mol_dgl_graph_layer1 = add_features(smiles, mol_dgl_graph_layer1)
                        #     mol_dgl_graph_layer1_num_nodes = mol_dgl_graph_layer1.num_nodes()
                        #     mol_dgl_graph_layer1.ndata[new_feature_name]= torch.zeros(mol_dgl_graph_layer1_num_nodes, 1)

                        #     ## Construct Layer2 graph
                        #     mol_dgl_graph_layer2 = eval(graph_constructor_init)(smiles)
                        #     mol_dgl_graph_layer2 = add_features(smiles, mol_dgl_graph_layer2)

                        #     hierarchi_graph = dgl.batch([mol_dgl_graph_layer1, mol_dgl_graph_layer2])

                        #     return mol_dgl_graph_layer1

                        # def mol_hierarchi_graph_batch_layer2(smiles):
                        #     ## Construct Layer2 graph
                        #     mol_dgl_graph_layer2 = eval(graph_constructor_init)(smiles)
                        #     mol_dgl_graph_layer2 = add_features(smiles, mol_dgl_graph_layer2)
                        #     mol_dgl_graph_layer2_num_nodes = mol_dgl_graph_layer2.num_nodes()
                        #     mol_dgl_graph_layer2.ndata["scaffold_nodes"]= torch.zeros(mol_dgl_graph_layer2_num_nodes, 1)
                        #     id_scaffold_nodes = df[df['Smiles'] == smiles]['Scaffolds'].squeeze()
                        #     if id_scaffold_nodes != []:
                        #         mol_dgl_graph_layer2.ndata["scaffold_nodes"][id_scaffold_nodes] = torch.ones(len(id_scaffold_nodes), 1)
                        #     else:
                        #         mol_dgl_graph_layer2.ndata["scaffold_nodes"] = torch.ones(mol_dgl_graph_layer2_num_nodes, 1)

                        #     return mol_dgl_graph_layer2

                        # smiles = 'Cl[Yb](Cl)Cl'
                        # # smiles="COc1ccc2c(c1OC)CN1CCc3cc4c(cc3C1C2)OCO4"
                        # mol_hierarchi_graph_batch_layer2(smiles).ndata["scaffold_nodes"]

                        # mol_hierarchi_graph_batch_layer2(smiles).ndata["FGs_new_nodes"]

                        # def mol_hierarchi_graph_batch_layer3(smiles):
                        #     ## Construct Layer2 graph
                        #     mol_dgl_graph_layer2 = eval(graph_constructor_init)(smiles)
                        #     mol_dgl_graph_layer2 = add_features(smiles, mol_dgl_graph_layer2)
                        #     mol_dgl_graph_layer2_num_nodes = mol_dgl_graph_layer2.num_nodes()
                        #     mol_dgl_graph_layer2.ndata["scaffold_nodes"]= torch.zeros(mol_dgl_graph_layer2_num_nodes, 1)
                        #     id_scaffold_nodes = df[df['Smiles'] == smiles]['Scaffolds'].squeeze()
                        #     if id_scaffold_nodes != []:
                        #         mol_dgl_graph_layer2.ndata["scaffold_nodes"][id_scaffold_nodes] = torch.ones(len(id_scaffold_nodes), 1)
                        #     else:
                        #         mol_dgl_graph_layer2.ndata["scaffold_nodes"] = torch.ones(mol_dgl_graph_layer2_num_nodes, 1)

                        #     return mol_dgl_graph_layer2

                        """## Add Node and Edge Features to Molecule Graph"""

                        # # Simple version
                        #
                        # def add_features(smiles, mol_dgl_graph):
                        #     mol = Chem.MolFromSmiles(smiles)

                        #     # Add two new vertices with an edge, in order to edata be non-empty. 
                        #     mol_dgl_graph.add_edges([mol_dgl_graph.num_nodes()], [mol_dgl_graph.num_nodes()+1])

                        #     # Add edge features
                        #     try:
                        #         base_bond_features = bond_featurizer(mol)["e"]
                        #         zero_rows_edge = torch.zeros(mol_dgl_graph.num_edges()-base_bond_features.shape[0], base_bond_features.shape[1])
                        #         mol_dgl_graph.edata["e"]= torch.cat((base_bond_features, zero_rows_edge), 0)
                        #     except:
                        #         mol_dgl_graph.edata["e"] = torch.zeros(mol_dgl_graph.num_edges(), 19)

                        #     # Add node features
                        #     smiles_idx = dataset_smiles_series[dataset_smiles_series==smiles].index.values[0]
                        #     zero_rows_node = torch.zeros(mol_dgl_graph.num_nodes()-Node_features_loaded[smiles_idx].shape[0], Node_features_loaded[smiles_idx].shape[1])
                        #     mol_dgl_graph.ndata["v"] = torch.cat((Node_features_loaded[smiles_idx], zero_rows_node), 0)

                        #     return mol_dgl_graph

                        """Module for converting graph to other NetworkX graph."""

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

                        def quotient_generator(dgl_graph, edge_condition_feature, op="mean", another_edges_feature=False, another_nodes_feature=False):
                            '''
                            graph: dgl Graph 
                            edge_condition_feature str: which shows a Boolean Feature. We would like to save the edges with True label 
                            nodes_feature: A string which is used as the output of function
                            edges_feature: A string which is used as the output of function
                            '''
                        ######################
                            def edges_with_feature_True(edges):
                                return (edges.data[edge_condition_feature]== False).squeeze(1)
                            ######    
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
                        #   ###########################################################
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
                            ###########################################################
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
                            ###############
                            if another_nodes_feature:
                                features_nodes_q1 = ["v", "qn1", another_nodes_feature]
                                features_nodes_q2 = ["v", "qn2", another_nodes_feature]
                            else:
                                features_nodes_q1 = ["v", "qn1"]
                                features_nodes_q2 = ["v", "qn2"]
                            ######
                            if another_edges_feature:
                                features_edges_q1 = ["e", "qe1", another_edges_feature]
                                features_edges_q2 = ["e", "qe2", another_edges_feature]
                            else:
                                features_edges_q1 = ["e", "qe1"]
                                features_edges_q2 = ["e", "qe2"]
                            ###############
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

                        """#### Some examples"""

                        # edges 0->1, 1->2
                        # u, v = torch.tensor([0, 1, 2, 3]), torch.tensor([3, 3, 3, 0])
                        # g = dgl.graph((u, v))
                        # print(g)

                        # g.ndata["v"]=torch.tensor([1,2,3,4.])
                        # g.edata["e"]=torch.tensor([10, 100, 1000., 222])

                        # g.edata["c"]= torch.tensor([True, True, False, True])

                        # h=quotient_generator(g, "c")

                        # g.ndata

                        # count=1
                        # quotient_generator(g, "c")[1].edata["ee1"]

                        # h.edata

                        # h.edata["cond"]=torch.tensor([False, False, True])
                        # h.edata["cond"].view(-1,1)

                        # h, f = quotient_generator(h, "cond")

                        # h.ndata

                        # Some codes related to the clustering based on atomic numbers

                        # list_atomic_number=[[6], [14, 32, 50, 82], [1, 3, 11, 19], [4, 12, 20], [5, 13, 49, 81], [7, 15, 33, 51], 
                        # [8, 16, 34], [9, 17, 35, 53], [22, 40, 23, 24, 25, 26, 27, 28, 46, 78, 29, 47, 79, 30, 48, 80, 70]]

                        # # edge_types=["usual_edges", "connections_types", "bonds_linker"]

                        # def node_condition(nodes, i):
                        #     return (nodes.data["v"][:,161] in atomic_number[i]).squeeze(1)

                                ###########################################################
                                ### Cluster based atomic Number
                                # count= 2
                                # b = mol_dgl_graph.ndata["v"][:,161]
                                # for item in list_atomic_number:
                                #     a=torch.zeros(mol_dgl_graph.ndata["v"][:,161].shape)
                                #     for j in item:
                                #         a=torch.logical_or(a, mol_dgl_graph.ndata["v"][:,161]== j)
                                #     mol_dgl_graph.ndata["v"][:,161][a]== count
                                #     a= torch.logical_not(a)
                                #     b = torch.logical_and(b, a)
                                #     count+=1
                                # mol_dgl_graph.ndata["v"][:,161][b]== 1
                                ###########################################################

                        """#### Hierarchical_Quotient"""

                        if utils_graph_generator.name_model == "Hierarchical_Quotient": 
                            def add_features(smiles, mol_dgl_graph):
                                global incorrect
                                mol = Chem.MolFromSmiles(smiles)

                                # Add edge features
                                try:
                                    mol_dgl_graph.edata["e"] = bond_featurizer(mol)["e"]
                                except:
                                    # pass
                                    mol_dgl_graph.edata["e"] = torch.zeros(mol_dgl_graph.num_edges(), 19)
                                ### Adding a new column
                                mol_dgl_graph.edata["e"]=torch.cat((mol_dgl_graph.edata["e"], torch.ones(mol_dgl_graph.num_edges()).view(-1,1)), 1)
                                
                                # Add node features
                                smiles_idx = dataset_smiles_series[dataset_smiles_series==smiles].index.values[0]
                                mol_dgl_graph.ndata["v"] = Node_features_loaded[smiles_idx]
                                ### Adding a new column
                                mol_dgl_graph.ndata["v"]=torch.cat((mol_dgl_graph.ndata["v"], torch.ones(mol_dgl_graph.num_nodes()).view(-1,1)), 1)


                                ##### Quotient based on Carbons, FGs, Types
                                #### One quotient based on Carbons
                                if utils_graph_generator.num_quotient == 1: 
                                    mol_dgl_graph_q1 = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs", op=HQ_first_aggregation_op,\
                                                                        another_edges_feature="edges_fgs").int()
                                elif utils_graph_generator.num_quotient == 2:
                                #### Two Quotients based on Carbons and FGs (or types)
                                    ## fisrt approach:
                                    # mol_dgl_graph_q0 = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs", op="mean",\
                                    #                                     another_edges_feature="edges_fgs").int()
                                    # mol_dgl_graph_q1 = quotient_generator(mol_dgl_graph_q0, edge_condition_feature="edges_fgs", op="sum").int()
                                    ## second approach:
                                    mol_dgl_graph_q0 = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs", op=HQ_first_aggregation_op,\
                                                                        another_edges_feature="edges_fgs").int()
                                    mol_dgl_graph_q1 = quotient_generator(mol_dgl_graph_q0, edge_condition_feature="edges_fgs", op="sum").int()
                                ####
                                else: #num_quotient == 0
                                    mol_dgl_graph_q1 = mol_dgl_graph
                                    mol_dgl_graph_q1.ndata["qn2"]= torch.range(0, mol_dgl_graph_q1.num_nodes()-1)
                                    mol_dgl_graph.ndata["qn1"]= torch.range(0, mol_dgl_graph.num_nodes()-1)
                                    if mol_dgl_graph_q1.num_edges()>0:
                                        mol_dgl_graph_q1.edata["qe2"]= torch.range(0, mol_dgl_graph_q1.num_edges()-1)
                                        mol_dgl_graph.edata["qe1"]= torch.range(0, mol_dgl_graph.num_edges()-1)
                                    else:
                                        mol_dgl_graph_q1.edata["qe2"]= torch.zeros(mol_dgl_graph_q1.num_edges(),1)
                                        mol_dgl_graph.edata["qe1"]= torch.zeros(mol_dgl_graph.num_edges(),1)

                                mol_dgl_graph.ndata["qn2"]=torch.full((mol_dgl_graph.num_nodes(),1), 0).to(torch.int32)
                                mol_dgl_graph.edata["qe2"]=torch.full((mol_dgl_graph.num_edges(),1), 0).to(torch.float32)
                                ####        
                                mol_dgl_graph_q1.ndata["qn1"]=torch.full((mol_dgl_graph_q1.num_nodes(),1), -1).to(torch.int32)
                                mol_dgl_graph_q1.edata["qe1"]=torch.full((mol_dgl_graph_q1.num_edges(),1), -1).to(torch.float32)
                                ####
                                mol_dgl_graph_q1.ndata["qn2"]= mol_dgl_graph_q1.ndata["qn2"].to(dtype=torch.int32)
                                mol_dgl_graph.ndata["qn1"]= mol_dgl_graph.ndata["qn1"].to(dtype=torch.int32)
                                
                                mol_dgl_graph_q1.ndata["v"][:, -1] = 2
                                mol_dgl_graph_q1.edata["e"][:, -1] = 2


                                if utils_graph_generator.num_quotient == 1: 
                                    if utils_graph_generator.num_layers_including_quotient==3:
                                        # mol_dgl_graph_q1.ndata["qn1"]=torch.full((mol_dgl_graph_q1.num_nodes(),1), -1).to(torch.int32)
                                        # mol_dgl_graph_q1.edata["qe1"]=torch.full((mol_dgl_graph_q1.num_edges(),1), -1).to(torch.float32)
                                        # mol_dgl_graph.ndata["qn2"]=torch.full((mol_dgl_graph.num_nodes(),1), 0).to(torch.int32)
                                        # mol_dgl_graph.edata["qe2"]=torch.full((mol_dgl_graph.num_edges(),1), 0).to(torch.float32)
                                        # mol_dgl_graph_q1.ndata["qn2"]= mol_dgl_graph_q1.ndata["qn2"].to(dtype=torch.int32)
                                        # mol_dgl_graph.ndata["qn1"]= mol_dgl_graph.ndata["qn1"].to(dtype=torch.int32)
                                        ######### Use Just One Quotient
                                        mol_dgl_graph_final = dgl.batch([mol_dgl_graph, mol_dgl_graph_q1], ndata=["v", "qn1", "qn2"],\
                                                                        edata=["e", "qe1", "qe2"]).int()
                                        for i in mol_dgl_graph_q1.ndata["qn2"][mol_dgl_graph_q1.ndata["qn2"]!=0]:
                                            list_nodes= (mol_dgl_graph.ndata["qn1"]== i).nonzero(as_tuple=True)[0].tolist()
                                            list_nodes=torch.tensor(list_nodes, dtype=torch.int32)
                                            index =(mol_dgl_graph_q1.ndata["qn2"]==i).nonzero(as_tuple=True)[0]+mol_dgl_graph.num_nodes()
                                            index=torch.full((1,len(list_nodes)), index.item(), dtype=torch.int32).view(-1)
                                            mol_dgl_graph_final.add_edges(list_nodes, index)
                                            mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 12
                                            if utils_graph_generator.both_direction:
                                                ## Backward Direction
                                                mol_dgl_graph_final.add_edges(index, list_nodes)
                                                mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 21
                                        if utils_graph_generator.universal_vertex==True:  
                                            #### Adding a universal vertex connected to second layer
                                            num_nodes_init= mol_dgl_graph_final.num_nodes()
                                            mol_dgl_graph_final.add_edges(list(range(mol_dgl_graph.num_nodes(), num_nodes_init)), [num_nodes_init])
                                            mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 23
                                            if utils_graph_generator.both_direction:
                                                ## Backward Direction
                                                mol_dgl_graph_final.add_edges([num_nodes_init], list(range(mol_dgl_graph.num_nodes(), num_nodes_init)))
                                                mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 32
                                            ###########################################################
                                            mol_dgl_graph_final.ndata["v"][mol_dgl_graph_final.num_nodes()-1, -1]= 3
                                        ###########################################################
                                        mol_dgl_graph_final.ndata.pop('qn1')
                                        mol_dgl_graph_final.ndata.pop('qn2')
                                        mol_dgl_graph_final.edata.pop('qe1')
                                        mol_dgl_graph_final.edata.pop('qe2')
                                        ####################################
                                        return mol_dgl_graph_final 

                                    elif utils_graph_generator.num_layers_including_quotient ==2:  
                                        #### Make just two layers (contain of quotient and universal vertex)
                                        num_nodes_init=mol_dgl_graph_q1.num_nodes()
                                        mol_dgl_graph_q1.add_edges(list(range(num_nodes_init)), [num_nodes_init])
                                        mol_dgl_graph_q1.edata["e"][mol_dgl_graph_q1.edata["e"][:, -1]==0][:, -1] = 23
                                        if utils_graph_generator.both_direction:
                                            ## Backward Direction
                                            mol_dgl_graph_q1.add_edges([num_nodes_init], list(range(num_nodes_init)))
                                            mol_dgl_graph_q1.edata["e"][mol_dgl_graph_q1.edata["e"][:, -1]==0][:, -1] = 32
                                        ### Node Feature
                                        mol_dgl_graph_q1.ndata["v"][mol_dgl_graph_q1.num_nodes()-1, -1]= 3
                                        mol_dgl_graph_q1.ndata.pop('qn1')
                                        mol_dgl_graph_q1.ndata.pop('qn2')
                                        mol_dgl_graph_q1.edata.pop('qe1')
                                        mol_dgl_graph_q1.edata.pop('qe2') 
                                        return mol_dgl_graph_q1

                                    else: #num_layers_including_quotient ==1
                                        mol_dgl_graph_q1.ndata.pop('qn1')
                                        mol_dgl_graph_q1.ndata.pop('qn2')
                                        mol_dgl_graph_q1.edata.pop('qe1')
                                        mol_dgl_graph_q1.edata.pop('qe2') 
                                        return mol_dgl_graph_q1

                                elif utils_graph_generator.num_quotient == 2:
                                ####################### We need both mol_dgl_graph_q1 and mol_dgl_graph_q0
                                    if utils_graph_generator.num_layers_including_quotient==3: 
                                        # mol_dgl_graph_q1.ndata["qn1"]=torch.full((mol_dgl_graph_q1.num_nodes(),1), -1).to(torch.int32)
                                        # mol_dgl_graph_q1.edata["qe1"]=torch.full((mol_dgl_graph_q1.num_edges(),1), -1).to(torch.float32)
                                        # mol_dgl_graph.ndata["qn2"]=torch.full((mol_dgl_graph.num_nodes(),1), 0).to(torch.int32)
                                        # mol_dgl_graph.edata["qe2"]=torch.full((mol_dgl_graph.num_edges(),1), 0).to(torch.float32)
                                        # mol_dgl_graph_q1.ndata["qn2"]= mol_dgl_graph_q1.ndata["qn2"].to(dtype=torch.int32)
                                        # mol_dgl_graph.ndata["qn1"]= mol_dgl_graph.ndata["qn1"].to(dtype=torch.int32)
                                        ######### 
                                        mol_dgl_graph_final = dgl.batch([mol_dgl_graph, mol_dgl_graph_q1], ndata=["v", "qn1", "qn2"],\
                                                                        edata=["e", "qe1", "qe2"]).int()
                                        for i in mol_dgl_graph_q1.ndata["qn2"][mol_dgl_graph_q1.ndata["qn2"]!=0]:
                                            list_nodes_q0= (mol_dgl_graph_q0.ndata["qn1"]== i).nonzero(as_tuple=True)[0].tolist()
                                            list_nodes_q0 =torch.tensor(list_nodes_q0, dtype=torch.int32)
                                            for j in list_nodes_q0:
                                                list_nodes = (mol_dgl_graph.ndata["qn1"]== mol_dgl_graph_q0.ndata["qn2"][j]).nonzero(as_tuple=True)[0].tolist()
                                                list_nodes =torch.tensor(list_nodes, dtype=torch.int32)
                                                index =(mol_dgl_graph_q1.ndata["qn2"]==i).nonzero(as_tuple=True)[0]+mol_dgl_graph.num_nodes()
                                                index=torch.full((1,len(list_nodes)), index.item(), dtype=torch.int32).view(-1)
                                                mol_dgl_graph_final.add_edges(list_nodes, index)
                                                mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 12
                                                if utils_graph_generator.both_direction:
                                                    ## Backward Direction
                                                    mol_dgl_graph_final.add_edges(index, list_nodes)
                                                    mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 21
                                        if utils_graph_generator.universal_vertex==True:        
                                            #### Adding a universal vertex connected to second layer
                                            num_nodes_init= mol_dgl_graph_final.num_nodes()
                                            mol_dgl_graph_final.add_edges(list(range(mol_dgl_graph.num_nodes(), num_nodes_init)), [num_nodes_init])
                                            mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 23
                                            if utils_graph_generator.both_direction:
                                                ## Backward Direction
                                                mol_dgl_graph_final.add_edges([num_nodes_init], list(range(mol_dgl_graph.num_nodes(), num_nodes_init)))
                                                mol_dgl_graph_final.edata["e"][mol_dgl_graph_final.edata["e"][:, -1]==0][:, -1] = 32
                                            ###########################################################
                                            mol_dgl_graph_final.ndata["v"][mol_dgl_graph_final.num_nodes()-1, -1]= 3
                                        ###########################################################
                                        mol_dgl_graph_final.ndata.pop('qn1')
                                        mol_dgl_graph_final.ndata.pop('qn2')
                                        mol_dgl_graph_final.edata.pop('qe1')
                                        mol_dgl_graph_final.edata.pop('qe2')
                                        ####################################
                                        return mol_dgl_graph_final 

                                    elif utils_graph_generator.num_layers_including_quotient ==2:  
                                        '''
                                        A graph that is contained of the quotient of quotient graph and a universal vertex
                                        '''
                                        #### Make just two layers
                                        num_nodes_init=mol_dgl_graph_q1.num_nodes()
                                        mol_dgl_graph_q1.add_edges(list(range(num_nodes_init)), [num_nodes_init])
                                        mol_dgl_graph_q1.edata["e"][mol_dgl_graph_q1.edata["e"][:, -1]==0][:, -1] = 23
                                        if utils_graph_generator.both_direction:
                                            ## Backward Direction
                                            mol_dgl_graph_q1.add_edges([num_nodes_init], list(range(num_nodes_init)))
                                            mol_dgl_graph_q1.edata["e"][mol_dgl_graph_q1.edata["e"][:, -1]==0][:, -1] = 32
                                        ### Node Feature
                                        mol_dgl_graph_q1.ndata["v"][mol_dgl_graph_q1.num_nodes()-1, -1]= 3
                                        mol_dgl_graph_q1.ndata.pop('qn1')
                                        mol_dgl_graph_q1.ndata.pop('qn2')
                                        mol_dgl_graph_q1.edata.pop('qe1')
                                        mol_dgl_graph_q1.edata.pop('qe2')  
                                        return mol_dgl_graph_q1
                                    else: 
                                        mol_dgl_graph_q1.ndata.pop('qn1')
                                        mol_dgl_graph_q1.ndata.pop('qn2')
                                        mol_dgl_graph_q1.edata.pop('qe1')
                                        mol_dgl_graph_q1.edata.pop('qe2')  
                                        return mol_dgl_graph_q1

                                else:
                                    incorrect = True
                                    return print("The model settings are incorrect!")

                        """#### Quotient_complement"""

                        if utils_graph_generator.name_model == "Quotient_complement": 
                            def add_features(smiles, mol_dgl_graph):
                                mol = Chem.MolFromSmiles(smiles)

                                # Add edge features
                                try:
                                    mol_dgl_graph.edata["e"] = bond_featurizer(mol)["e"]
                                except:
                                    # pass
                                    mol_dgl_graph.edata["e"] = torch.zeros(mol_dgl_graph.num_edges(), 19)
                                ### Adding a new column
                                mol_dgl_graph.edata["e"]=torch.cat((mol_dgl_graph.edata["e"], torch.ones(mol_dgl_graph.num_edges()).view(-1,1)), 1)
                                
                                # Add node features
                                smiles_idx = dataset_smiles_series[dataset_smiles_series==smiles].index.values[0]
                                mol_dgl_graph.ndata["v"] = Node_features_loaded[smiles_idx]
                                ### Adding a new column
                                mol_dgl_graph.ndata["v"]=torch.cat((mol_dgl_graph.ndata["v"], torch.ones(mol_dgl_graph.num_nodes()).view(-1,1)), 1)

                                ###########  Quotients
                                ##### Quotient based on Carbons, FGs, Types
                                if utils_graph_generator.num_quotient == 1: 
                                    mol_dgl_graph_q1 = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs", op="mean",\
                                                                        another_edges_feature="edges_fgs").int()
                                elif utils_graph_generator.num_quotient == 2:
                                #### Two Quotients based on Carbons and FGs (or types)
                                    mol_dgl_graph_q0 = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs", op="mean",\
                                                                        another_edges_feature="edges_fgs").int()
                                    mol_dgl_graph_q1 = quotient_generator(mol_dgl_graph_q0, edge_condition_feature="edges_fgs", op="sum").int()
                                ####
                                else:
                                    mol_dgl_graph_q1 = mol_dgl_graph
                                    
                                mol_dgl_graph_q1.ndata["v"][:, -1] = 2
                                mol_dgl_graph_q1.edata["e"][:, -1] = 2

                                ###########################################################
                                ### Add Complement of Graph
                                ###########################################################
                                if utils_graph_generator.add_complement==True:
                                    list_idx_nodes=list(range(mol_dgl_graph_q1.num_nodes()))
                                    num_nodes = mol_dgl_graph_q1.num_nodes()
                                    for i in list_idx_nodes:
                                        for j in list_idx_nodes:
                                            if i != j:
                                                try:
                                                    mol_dgl_graph_q1.edge_ids([i], [j])
                                                except:
                                                    mol_dgl_graph_q1.add_edges([num_nodes+i], [num_nodes+j])
                                                    idx_new_edge = (mol_dgl_graph_q1.edata["e"][:, -1]==0).nonzero(as_tuple=True)[0].tolist()
                                                    mol_dgl_graph_q1.edata["e"][idx_new_edge, -1] = 29
                                    for i in list_idx_nodes:
                                        try: 
                                            mol_dgl_graph_q1.ndata["v"][num_nodes+i, :] = mol_dgl_graph_q1.ndata["v"][i, :]
                                            mol_dgl_graph_q1.ndata["v"][num_nodes+i, -1] = 29
                                        except:
                                            return mol_dgl_graph_q1
                                ################################
                                return mol_dgl_graph_q1

                        """#### Hierarchical_types_FGs"""

                        if utils_graph_generator.name_model == "Hierarchical_types_FGs": 
                            def add_features(smiles, mol_dgl_graph):
                                global incorrect
                                mol = Chem.MolFromSmiles(smiles)

                                # Add edge features
                                try:
                                    mol_dgl_graph.edata["e"] = bond_featurizer(mol)["e"]
                                except:
                                    # pass
                                    mol_dgl_graph.edata["e"] = torch.zeros(mol_dgl_graph.num_edges(), 19)
                                ### Adding a new column
                                mol_dgl_graph.edata["e"]=torch.cat((mol_dgl_graph.edata["e"], torch.ones(mol_dgl_graph.num_edges()).view(-1,1)), 1)
                                
                                # Add node features
                                smiles_idx = dataset_smiles_series[dataset_smiles_series==smiles].index.values[0]
                                mol_dgl_graph.ndata["v"] = Node_features_loaded[smiles_idx]
                                ### Adding a new column
                                mol_dgl_graph.ndata["v"]=torch.cat((mol_dgl_graph.ndata["v"], torch.ones(mol_dgl_graph.num_nodes()).view(-1,1)), 1)

                                ### Types:
                                if utils_graph_generator.types_status ==True:
                                    num_nodes_init= mol_dgl_graph.num_nodes()
                                    for types in identify_functional_groups(mol)[1]:
                                        #### Add a new vertex for any type:
                                        mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                                        mol_dgl_graph.add_edges(types, [mol_dgl_graph_num_nodes])
                                        mol_dgl_graph.ndata["v"][mol_dgl_graph.ndata["v"][:, -1]==0][:, -1] = 2
                                        mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 12
                                        if utils_graph_generator.both_direction:
                                            mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], types)
                                            mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 21
                                    ###################
                                    num_nodes_second_layer = mol_dgl_graph.num_nodes()
                                    if utils_graph_generator.universal_vertex==True:   
                                        #### Adding a universal vertex connected to second layer
                                        mol_dgl_graph.add_edges(list(range(num_nodes_init, num_nodes_second_layer)), [num_nodes_second_layer])
                                        mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 23
                                        if utils_graph_generator.both_direction:
                                            ## Backward Direction
                                            mol_dgl_graph.add_edges([num_nodes_second_layer], list(range(num_nodes_init, num_nodes_second_layer)))
                                            mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 32
                                        ###########################################################
                                        mol_dgl_graph.ndata["v"][mol_dgl_graph.num_nodes()-1, -1]= 3

                                else:
                                    num_nodes_init = mol_dgl_graph.num_nodes()
                                    for Fgs in identify_functional_groups(mol)[0]:
                                        #### Add a new vertex for any type:
                                        mol_dgl_graph_num_nodes= mol_dgl_graph.num_nodes()
                                        mol_dgl_graph.add_edges(Fgs[0], [mol_dgl_graph_num_nodes])
                                        mol_dgl_graph.ndata["v"][mol_dgl_graph.ndata["v"][:, -1]==0][:, -1] = 2
                                        mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 12
                                        if utils_graph_generator.both_direction:
                                            mol_dgl_graph.add_edges([mol_dgl_graph_num_nodes], Fgs[0])
                                            mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 21
                                    ###################
                                    num_nodes_second_layer = mol_dgl_graph.num_nodes()
                                    if utils_graph_generator.universal_vertex==True:   
                                        #### Adding a universal vertex connected to second layer
                                        mol_dgl_graph.add_edges(list(range(num_nodes_init, num_nodes_second_layer)), [num_nodes_second_layer])
                                        mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 23
                                        if utils_graph_generator.both_direction:
                                            ## Backward Direction
                                            mol_dgl_graph.add_edges([num_nodes_second_layer], list(range(num_nodes_init, num_nodes_second_layer)))
                                            mol_dgl_graph.edata["e"][mol_dgl_graph.edata["e"][:, -1]==0][:, -1] = 32
                                        ###########################################################
                                        mol_dgl_graph.ndata["v"][mol_dgl_graph.num_nodes()-1, -1]= 3
                                    ################################
                                if utils_graph_generator.num_quotient == 1: 
                                    mol_dgl_graph = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs",\
                                                                    op="mean", another_edges_feature="edges_fgs").int()
                                    # mol_dgl_graph.ndata.pop('qn1')
                                    mol_dgl_graph.ndata.pop('qn2')
                                    # mol_dgl_graph.edata.pop('qe1')
                                    mol_dgl_graph.edata.pop('qe2')
                                elif utils_graph_generator.num_quotient >= 2:
                                    incorrect = True
                                    return print("The model settings are incorrect!")
                        
                                return mol_dgl_graph

                        # node_list_elec = atom_finder(mol)
                        # node_list_elec_it = itertools.combinations(node_list_elec, 2)
                        # q=list(node_list_elec_it)
                        # for i in range(len(q)):
                        #     mol_dgl_graph.add_edges(q[i][0],q[i][1])

                        mylist = [1]
                        nodes1 = frozenset(mylist)
                        mylist = [4, 5, 6]
                        nodes2 = frozenset(mylist)
                        nodes =frozenset([nodes1, nodes2])
                        nodes

                        for a in nodes:
                            print(list(a))

                        """## Classification datasets

                        ### Tox21 dataset
                        """

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.Tensor(self.labels[idx]), torch.Tensor(self.masks[idx]), torch.Tensor(self.global_feats[idx])
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()

                        """### BBBP dataset"""

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),\
                                # torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),\
                                torch.tensor(self.global_feats[idx]).float()

                        """### Bace dataset"""

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),\
                                # torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),\
                                torch.tensor(self.global_feats[idx]).float()

                        """### Toxcast dataset"""

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.Tensor(self.labels[idx]), torch.Tensor(self.masks[idx]), torch.Tensor(self.global_feats[idx])
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()

                        """### Clintox dataset"""

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.Tensor(self.labels[idx]), torch.Tensor(self.masks[idx]), torch.Tensor(self.global_feats[idx])
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()

                        """### Sider dataset"""

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.Tensor(self.labels[idx]), torch.Tensor(self.masks[idx]), torch.Tensor(self.global_feats[idx])
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()


                        """### MUV dataset"""

                        # A class of custom dataset with PyTorch
                        class DatasetMuv(torch.utils.data.Dataset):
                            def __init__(self, csv_address, path_global_csv):
                                # Read csv file, and fill in NaN values with 0
                                self.csv = pd.read_csv(csv_address).fillna(0) 
                                self.path_global_csv = pd.read_csv(path_global_csv)
                                
                                # Make masks for labels (0 as NaN value, and 1 as other values)
                                self.masks_csv = pd.read_csv(csv_address).replace({0: 1}).fillna(0)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:, -1]
                                self.labels = self.csv.iloc[:, :-2].values
                                self.masks = self.masks_csv.iloc[:, :-2].values

                                self.global_feats = self.path_global_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                # return  self.smiles[idx], torch.Tensor(self.labels[idx]), torch.Tensor(self.masks[idx]), torch.Tensor(self.global_feats[idx])
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(), torch.tensor(self.global_feats[idx]).float()


                        """### AMU dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetAmu(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,1].values
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), self.masks[idx],  torch.tensor(self.global_feats[idx]).float()

                        """### Ellinger dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetEllinger(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,1].values
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), self.masks[idx],  torch.tensor(self.global_feats[idx]).float()

                        
                        """### CovidAmu dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetCovidAmu(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,2].values # for merged datasets the third row is considered for labels
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), self.masks[idx],  torch.tensor(self.global_feats[idx]).float()

                    
                        """### CovidEllinger dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetCovidEllinger(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,2].values # for merged datasets the third row is considered for labels
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), self.masks[idx],  torch.tensor(self.global_feats[idx]).float()


                        """### CovidAmuEllinger dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetCovidAmuEllinger(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,2].values # for merged datasets the third row is considered for labels
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), self.masks[idx],  torch.tensor(self.global_feats[idx]).float()


                        """ Covid Related datasets for multitask clinical trials: Phase-1 to Phase-4 
                            CovidAmuMultitask - CovidEllingerMultitask - CovidAmuEllingerMultitask
                        """

                        # A class of custom dataset with PyTorch 
                        class DatasetCovidMultitask(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address).fillna(0) # filling NaN values with 0 just as the convention for Tox21
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # For building masks, first -> (0 as NaN value, and 1 as other values)
                                self.masks_csv = pd.read_csv(csv_address).replace({0: 1}).fillna(0)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                # for the multitask (merged) datasets the last four columns (starting with index 2) are labels
                                self.labels = self.csv.iloc[:,2:].values
                                self.masks = self.masks_csv.iloc[:,2:].values

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).float(), torch.tensor(self.masks[idx]).float(),  torch.tensor(self.global_feats[idx]).float()

                        
                        """### hiv dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetHiv(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,1].values # for hiv dataset the second row is considered for labels
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), self.masks[idx],  torch.tensor(self.global_feats[idx]).float()
                        
                        
                        """## Regression datasets

                        ### Lipophilicity dataset
                        """

                        # A class of custom dataset with PyTorch
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
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),\
                                # torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),\
                                torch.tensor(self.global_feats[idx]).float()

                        """### Delaney(ESOL) dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetDelaney(torch.utils.data.Dataset):
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
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),  torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()


                        """### Sampl(FreeSolv) dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetSampl(torch.utils.data.Dataset):
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
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),  torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()

                        
                        """### Ionic dataset"""                        

                        # A class of custom dataset with PyTorch 
                        class DatasetIonic(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,1]
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),  torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()

                                                
                        """### QM7 dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetQm7(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,1].values
                                self.masks = torch.ones((len(self.smiles), 1))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),  torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()      
                        
                        """### QM8 dataset"""

                        # A class of custom dataset with PyTorch 
                        class DatasetQm8(torch.utils.data.Dataset):
                            def __init__(self, csv_address, global_feats_csv):
                                self.csv = pd.read_csv(csv_address) 
                                self.global_feats_csv = pd.read_csv(global_feats_csv)

                                # Split smiles, labels, and masks columns as lists
                                self.smiles = self.csv.iloc[:,0]
                                self.labels = self.csv.iloc[:,1:].values  # QM8 has tweleve tasks start from index 1 
                                self.masks = torch.ones((len(self.smiles), 12))

                                self.global_feats = self.global_feats_csv.iloc[:,1:].values

                            def __len__(self):
                                return len(self.smiles)

                            def __getitem__(self, idx):
                                # return  self.smiles[idx], torch.tensor(self.labels[idx]).view(-1,1).float(), torch.tensor(self.masks[idx]).float(),  torch.tensor(self.global_feats[idx]).float()
                                return  self.smiles[idx], torch.tensor(self.labels[idx]).view(1,-1).float(), torch.Tensor(self.masks[idx]),  torch.tensor(self.global_feats[idx]).float()      
                        


                        """ Replace global features NaN values with median (before making dataset)"""

                        # Global features

                        if not os.path.exists(current_dir+name_global_csv):
                            shutil.unpack_archive(path_global_zip, folder_data_temp)

                        glob_csv = pd.read_csv(path_global_csv)
                        glob_csv.head(5)

                        list_nan = []
                        for i in range(glob_csv.shape[1]):
                            if glob_csv.iloc[:,i].isnull().sum()>0:
                                list_nan.append(i)
                        print("list_nan_global_before: ", list_nan)
                        for i in list_nan:
                            glob_csv.iloc[:,i].fillna(glob_csv.iloc[:,i].median(), inplace=True)

                        list_nan = []
                        for i in range(glob_csv.shape[1]):
                            if glob_csv.iloc[:,i].isnull().sum()>0:
                                list_nan.append(i)
                        print("list_nan_global_after: ", list_nan)
                        glob_csv.to_csv(path_global_csv, index = False)

                        """### Load node features"""

                        if not os.path.exists(current_dir+"node_features.pickle"):
                            shutil.unpack_archive(path_node_feats_zip, folder_data_temp)

                        with open(folder_data_temp+"node_features.pickle", "rb") as handle:
                            Node_features_loaded = pickle.load(handle)

                        """### Make dataset"""

                        dataset = eval(utils_graph_generator.dataset_name)(path_data_csv, path_global_csv)

                        len(dataset)

                        """### Find global features NaN values in the dataset"""

                        # Find elements of global features with nan values

                        L=[]
                        for i in range(len(dataset)):
                            if torch.sum(torch.isnan(dataset[i][3]).int()).item() > 0:
                                L.append(i)

                        print(L)

                        # Find columns of global features with nan values

                        K1=[]
                        for a in L:
                            K2=[]
                            for t in range(len(dataset[a][3])):
                                if torch.isnan(dataset[a][3][t]):
                                    K2.append(t)
                            K1.append(K2)

                        dataset[0][3].shape

                        # Define a pandas series from dataset smiles
                        dataset_smiles_series = dataset.smiles.squeeze()

                        dataset_smiles_series

                        # Train, validation, and test set split
                        # train_set, val_set, test_set = dgllife.utils.RandomSplitter.train_val_test_split(dataset, frac_val=0.1, frac_test=0.1, random_state=213)
                        # train_set, val_set, test_set = dgllife.utils.ScaffoldSplitter.train_val_test_split(dataset, frac_val=0.1, frac_test=0.1, scaffold_func='smiles')

                        def splitted_data(string="train"):
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

                        # Train, validation, and test set split

                        train_set = splitted_data(string="train")
                        val_set = splitted_data(string="val")
                        test_set = splitted_data(string="test")

                        print(len(train_set), len(val_set), len(test_set))

                        train_set[0]

                        test_set[0][0]

                        dataset_smiles_series[0]

                        bond_featurizer(Chem.MolFromSmiles('CCOc1ccc2nc(S(N)(=O)=O)sc2c1'))["e"].shape

                        train_set[2][0]

                        # %timeit eval(graph_constructor)(train_set[0][0])

                        train_set[3][0]

                        """### Check if parameters of generator model are True or not"""

                        g=add_features(train_set[0][0], graph_constructor(train_set[0][0]))
                        # print(g)
                        if incorrect:
                            sys.exit("Seed "+str(seed)+" is exited, because the model settings are incorrect!")    
                        else:
                            print("The model settings are correct!")
                            nx.draw_networkx(g.to_networkx(), with_labels = True)

                        Chem.MolFromSmiles(train_set[0][0])

                        # a=Subset(train_set, list(range(1,10)))
                        # a[0]

                        """## Prepare Training, Validation, and Test Sets"""

                        # shutil.rmtree(current_dir +"data/", ignore_errors=True)
                        shutil.rmtree(folder_data_temp, ignore_errors=True)


                        # improving generation's speed with generating graphs only once for all seeds and splits in a (graph_model,row ID)
                        ######################

                        if not are_graphs_generated:

                            print('== \n Generating Featurized Graphs! \n ==')

                            for i in  dataset_smiles_series:

                                # like bg= add_features(member[0], graph_constructor(member[0]))
                                generated_featurized_graphs[i] = add_features(i, graph_constructor(i))

                            are_graphs_generated = True
                            print(g, generated_featurized_graphs[train_set[0][0]])
            
                        ######################


                        """### Training Set"""

                        dgl_train=[]
                        smiles_train = []
                        labels_train= torch.empty(0)
                        masks_train = torch.empty(0)
                        globals_train = torch.empty(0)
                        counter = 0

                        for member in train_set:
                            bg = generated_featurized_graphs[member[0]]
                            dgl_train.append(bg)
                            smiles_train.append(member[0])
                            labels_train= torch.cat((labels_train, member[1]), dim=0)
                            masks_train= torch.cat((masks_train, member[2]), dim=0)  
                            globals_train= torch.cat((globals_train, member[3]), dim=0) 
                            counter+=1    

                        print(counter)

                        label={"labels":labels_train, "masks":masks_train, "globals":globals_train}
                        new_path=path_save_temp+"_train.bin"
                        dgl.save_graphs(new_path, dgl_train, labels=label)

                        import pickle
                        new_path=path_save_temp+"_smiles_train.pickle"
                        pickle_out = open(new_path,"wb")
                        pickle.dump(smiles_train, pickle_out)
                        pickle_out.close()

                        """### Validtion Set"""

                        dgl_val=[]
                        smiles_val = []
                        labels_val= torch.empty(0)
                        masks_val = torch.empty(0)
                        globals_val = torch.empty(0)
                        counter = 0
                        for member in val_set:
                            bg = generated_featurized_graphs[member[0]]
                            dgl_val.append(bg)   
                            smiles_val.append(member[0])
                            labels_val= torch.cat((labels_val, member[1]), dim=0)
                            masks_val= torch.cat((masks_val, member[2]), dim=0) 
                            globals_val= torch.cat((globals_val, member[3]), dim=0) 
                            counter+=1 

                        print(counter)

                        label={"labels":labels_val, "masks":masks_val, "globals":globals_val}
                        new_path=path_save_temp+"_val.bin"
                        dgl.save_graphs(new_path, dgl_val, labels=label)

                        new_path=path_save_temp+"_smiles_val.pickle"
                        pickle_out = open(new_path,"wb")
                        pickle.dump(smiles_val, pickle_out)
                        pickle_out.close()

                        """### Test Set"""

                        dgl_test=[]
                        smiles_test = []
                        labels_test= torch.empty(0)
                        masks_test = torch.empty(0)
                        globals_test = torch.empty(0)
                        counter = 0
                        for member in test_set:
                            bg = generated_featurized_graphs[member[0]]
                            dgl_test.append(bg)
                            smiles_test.append(member[0])
                            labels_test= torch.cat((labels_test, member[1]), dim=0)
                            masks_test= torch.cat((masks_test, member[2]), dim=0) 
                            globals_test= torch.cat((globals_test, member[3]), dim=0) 
                            counter+=1  

                        print(counter)

                        label={"labels":labels_test, "masks":masks_test, "globals":globals_test}
                        new_path=path_save_temp+"_test.bin"
                        dgl.save_graphs(new_path, dgl_test, labels=label)

                        new_path=path_save_temp+"_smiles_test.pickle"
                        pickle_out = open(new_path,"wb")
                        pickle.dump(smiles_test, pickle_out)
                        pickle_out.close()

                        # Make a zip file from generated data
                        shutil.make_archive(path_save_current_dir, 'zip', path_save_current_dir)

                        # Save zip file to the Google drive

                        # os.makedirs(path_save_0, exist_ok=True)
                        os.makedirs(path_save, exist_ok=True)

                        b = os.path.join(path_save, name_final_zip)  
                        # shutil.copy(current_dir+"data/"+name_final_zip, b)
                        shutil.copy(folder_data_temp+name_final_zip, b)

                        # shutil.rmtree(current_dir +"data/", ignore_errors=True)
                        shutil.rmtree(folder_data_temp, ignore_errors=True)

                        print("Seed ", seed, " is finished!")

                        if incorrect == True:
                            os._exit(0)
                    else:
                        list_gen_seeds.append(seed)
                        print("Seed ", seed, " was generated before!")