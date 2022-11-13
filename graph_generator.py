
""" Graph Generator """

import pandas as pd
import torch
import sys
import dgl
import os
import networkx as nx
from rdkit import Chem
import pickle
import shutil
import zipfile

import arguments
from arguments import args
from utils_generator import create_descriptors, identify_functional_groups, \
                                  CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
                                  RandomScaffoldSplitter, RandomSplitter, RandomStratifiedSplitter_m,\
                                  DatasetTox21, DatasetBBBP, DatasetBace, DatasetToxcast, \
                                  DatasetClintox, DatasetSider, DatasetLipo, DatasetESOL, DatasetFreeSolv,\
                                  fg_idx, fg_types_idx, convert_mol, make_df, fgs_connections_idx, \
                                  graph_constructor, to_networkx, quotient_generator, splitted_data

current_dir = args.current_dir

dataset_names = {
    "tox21" : "DatasetTox21",
    "bbbp" : "DatasetBBBP",
    "bace" : "DatasetBace",
    "toxcast" : "DatasetToxcast",
    "clintox" : "DatasetClintox",
    "sider" : "DatasetSider",
    "lipo": "DatasetLipo",
    "esol" : "DatasetESOL",
    "freesolv" : "DatasetFreeSolv",
}

# generating global features
for name in args.gen_names_data:
    name_global_csv = name + '_global_cdf_rdkit.csv'
    name_global_zip = name + '_global_cdf_rdkit.zip'
    saving_address = current_dir + 'data/global_features/' + name_global_zip
    if not os.path.exists(saving_address):
        print('Generating Global Features for', name)
        raw_data_url = current_dir + 'data/raw/' + name + '.csv'
        data = pd.read_csv(raw_data_url)
        descriptors = create_descriptors(data, mols_column_name='smiles')
        compression_opts = dict(method='zip', archive_name=name_global_csv)
        os.makedirs(os.path.dirname(saving_address), exist_ok=True)  
        descriptors.to_csv(saving_address, index=False, compression=compression_opts)

# generating node features (for faster graph generation)
atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()
for name in args.gen_names_data:
    saving_address = current_dir + 'data/node_features/' + name + '_node_127_one_hot' + '.zip'
    saving_address_pickle = current_dir + 'data/node_features/' + 'node_features.pickle'
    if not os.path.exists(saving_address):
        print('Generating Node Features for', name)
        raw_data_url = current_dir + 'data/raw/' + name + '.csv'
        data = pd.read_csv(raw_data_url)
        node_features=[]
        for i in range(len(data.smiles)):
            node_features.append(atom_featurizer(Chem.MolFromSmiles(data.smiles[i]))['h'])
        os.makedirs(os.path.dirname(saving_address), exist_ok=True)
        with open(saving_address_pickle , 'wb') as handle:
            pickle.dump(node_features, handle)
        zf = zipfile.ZipFile(saving_address, 'w', zipfile.ZIP_DEFLATED) 
        zf.write(saving_address_pickle, 'node_features.pickle')  #archname is necessary to remove the path once unpacked
        zf.close()
        os.remove(saving_address_pickle) 

# generating splits
scaffold_splitter = RandomScaffoldSplitter()
random_splitter = RandomSplitter()
stratified_splitter = RandomStratifiedSplitter_m()
splitters = [scaffold_splitter, random_splitter, stratified_splitter]
type_indexs = {'scaffold' : 0, 'random' : 1, 'stratified' : 2}

# splitting
for name in args.gen_names_data:
    raw_data_url = current_dir + 'data/raw/' + name + '.csv'
    data = pd.read_csv(raw_data_url)
    for split in args.splits:
        for seed in args.generation_seeds:

            saving_address = current_dir + 'data/splits/' + name + '/' + split + '_' + str(seed) + '/'

            if not os.path.exists(saving_address + 'train_smiles') or not os.path.exists(saving_address + 'val_smiles') or not os.path.exists(saving_address + 'test_smiles'):
                print('Generating', split, 'seed_', seed, 'split for', name)
                splitted_sets = splitters[type_indexs[split]].split(data, seed=seed)
                smiles_train = [data.smiles[i] for i in splitted_sets[0]]
                smiles_val = [data.smiles[i] for i in splitted_sets[1]]
                smiles_test =  [data.smiles[i] for i in splitted_sets[2]]
                os.makedirs(os.path.dirname(saving_address), exist_ok=True)
                with open(saving_address + 'train_smiles', 'wb') as handle:
                    pickle.dump(smiles_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(saving_address + 'val_smiles', 'wb') as handle:
                    pickle.dump(smiles_val, handle, protocol=pickle.HIGHEST_PROTOCOL)      
                with open(saving_address + 'test_smiles', 'wb') as handle:
                    pickle.dump(smiles_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

#generating graphs
for name_data in args.gen_names_data:
    for name_model in arguments.generation_models_rows.keys():
        for idx_row_graph_gen in arguments.generation_models_rows[name_model]:

            ## using generated_featurized_graphs for all seeds 
            generated_featurized_graphs = {}
            are_graphs_generated = False

            # specific settings for this scenario
            dataset_name = dataset_names[name]

            # """ Load csv of the specific graph generator"""
            # save_csv = current_dir + "data/graph/"
            # graph_gen_csv = pd.read_csv(save_csv + name_model+".csv")
            # row_graph_gen = graph_gen_csv.iloc[idx_row_graph_gen]

            if args.HQ_first_aggregation_op == 'sum':
                print('Caution: Quotient by non-FGs with sum aggregation')

            for split in args.splits: 

                print( '\n\n ========================= \n  Generating', name_model, 'with row number', idx_row_graph_gen, '. Generating graphs', \
                    'for', split, 'splitting with seeds', args.generation_seeds, 'for',\
                         name_data, 'dataset is started! \n ========================= \n\n')
                    
                ### Generation

                # incorrect = False

                list_gen_seeds = []
                for seed in args.generation_seeds: 
                    path_save= current_dir+"data/graph/"+name_data+"/"+split+"_"+str(seed)+"/"+arguments.name_final_zip
                    if not os.path.exists(path_save):
                        
                        save_results_status = True
                        list_gen_seeds.append(seed)

                        # Print the current seed
                        print("Seed ", seed, " is started!")

                        name_node_feats_zip = name_data+ "_node"+arguments.name_node_feature+".zip"
                        name_global_csv = name_data+"_global_cdf_rdkit.csv"
                        name_global_zip = name_data+"_global_cdf_rdkit.zip"

                        """Set Path"""
                        folder_data_temp = current_dir +"data/buffer/" 
                        path_global_csv = folder_data_temp + name_global_csv       
                        path_save_current_dir = folder_data_temp + arguments.name_final + "/" 
                        path_save_temp = path_save_current_dir + split + "_" + str(seed)

                        path_save_0 = current_dir+"data/graph/"+name_data
                        path_save = current_dir+"data/graph/"+name_data+"/"+split+"_"+str(seed)+"/"

                        path_node_feats = current_dir + 'data/node_features/' 
                        path_node_feats_zip = path_node_feats + name_node_feats_zip
                        path_smiles = current_dir + 'data/splits/' + name_data + "/" + split+"_"+str(seed)+"/"
                        path_data_csv = current_dir + 'data/raw/' + name_data + ".csv"
                        path_global_zip = current_dir + 'data/global_features/' + name_global_zip


                        #dataframe containing some details about the molecule
                        df = make_df(path_data_csv)

                        """Hierarchical_Quotient"""
                        if name_model == "Hierarchical_Quotient": 
                            def add_features(smiles, mol_dgl_graph):
                                # global incorrect
                                mol = Chem.MolFromSmiles(smiles)

                                # Add edge features
                                try:
                                    mol_dgl_graph.edata["e"] = bond_featurizer(mol)["e"]
                                except:
                                    # pass
                                    mol_dgl_graph.edata["e"] = torch.zeros(mol_dgl_graph.num_edges(), 12)
                                ### Adding a new column
                                mol_dgl_graph.edata["e"]=torch.cat((mol_dgl_graph.edata["e"], torch.ones(mol_dgl_graph.num_edges()).view(-1,1)), 1)
                                
                                # Add node features
                                smiles_idx = dataset_smiles_series[dataset_smiles_series==smiles].index.values[0]
                                mol_dgl_graph.ndata["v"] = Node_features_loaded[smiles_idx]
                                ### Adding a new column
                                mol_dgl_graph.ndata["v"]=torch.cat((mol_dgl_graph.ndata["v"], torch.ones(mol_dgl_graph.num_nodes()).view(-1,1)), 1)


                                #Quotient based on Carbons, FGs
                                mol_dgl_graph_q0 = quotient_generator(mol_dgl_graph, edge_condition_feature="edges_non_fgs", op=args.HQ_first_aggregation_op,\
                                                                    another_edges_feature="edges_fgs").int()
                                mol_dgl_graph_q1 = quotient_generator(mol_dgl_graph_q0, edge_condition_feature="edges_fgs", op="sum").int()

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

                                mol_dgl_graph_q1.ndata.pop('qn1')
                                mol_dgl_graph_q1.ndata.pop('qn2')
                                mol_dgl_graph_q1.edata.pop('qe1')
                                mol_dgl_graph_q1.edata.pop('qe2')  
                                return mol_dgl_graph_q1                     


                        """Replace global features NaN values with median (before making dataset)"""
                        if not os.path.exists(current_dir+name_global_csv):
                            shutil.unpack_archive(path_global_zip, folder_data_temp)

                        glob_csv = pd.read_csv(path_global_csv)

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

                        """Load node features"""
                        if not os.path.exists(current_dir+"node_features.pickle"):
                            shutil.unpack_archive(path_node_feats_zip, folder_data_temp)

                        with open(folder_data_temp+"node_features.pickle", "rb") as handle:
                            Node_features_loaded = pickle.load(handle)

                        """Make dataset"""
                        dataset = eval(dataset_name)(path_data_csv, path_global_csv)

                        # Define a pandas series from dataset smiles
                        dataset_smiles_series = dataset.smiles.squeeze()

                        # Train, validation, and test set split
                        train_set = splitted_data(path_smiles, dataset, dataset_smiles_series, string="train")
                        val_set = splitted_data(path_smiles, dataset, dataset_smiles_series, string="val")
                        test_set = splitted_data(path_smiles, dataset, dataset_smiles_series, string="test")

                        print(len(train_set), len(val_set), len(test_set))

                        """Check if parameters of generator model are True or not"""
                        g = add_features(train_set[0][0], graph_constructor(df, train_set[0][0]))
                        # if incorrect:
                        #     sys.exit("Seed "+str(seed)+" is exited, because the model settings are incorrect!")    
                        # else:
                        #     print("The model settings are correct!")
                        #     nx.draw_networkx(g.to_networkx(), with_labels = True)

                        """Prepare Training, Validation, and Test Sets"""
                        shutil.rmtree(folder_data_temp, ignore_errors=True)


                        #improving generation's speed with generating graphs only once for all seeds 
                        #and splits in a (graph_model,row ID)
                        if not are_graphs_generated:
                            print('== \n Generating Featurized Graphs! \n ==')
                            for i in  dataset_smiles_series:
                                generated_featurized_graphs[i] = add_features(i, graph_constructor(df, i))
                            are_graphs_generated = True
                            print(g, generated_featurized_graphs[train_set[0][0]])

                        """Training Set"""
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
                        new_path=path_save_temp+"_smiles_train.pickle"
                        pickle_out = open(new_path,"wb")
                        pickle.dump(smiles_train, pickle_out)
                        pickle_out.close()

                        """Validtion Set"""
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

                        """Test Set"""
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

                        shutil.make_archive(path_save_current_dir, 'zip', path_save_current_dir)
                        os.makedirs(path_save, exist_ok=True)
                        b = os.path.join(path_save, arguments.name_final_zip)  
                        shutil.copy(folder_data_temp+arguments.name_final_zip, b)
                        shutil.rmtree(folder_data_temp, ignore_errors=True)

                        print("Seed ", seed, " is finished!")

                        # if incorrect == True:
                        #     os._exit(0)
                    else:
                        list_gen_seeds.append(seed)
                        print("Seed ", seed, " was generated before!")

