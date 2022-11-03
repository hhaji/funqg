
""" A GNN Model """

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import arguments

class GNN(nn.Module):
    def __init__(self, config, global_size, num_tasks, global_feature, atom_messages):
        super().__init__()

        self.config = config
        self.global_size = global_size
        self.num_tasks = num_tasks
        self.global_feature = global_feature
        self.atom_messages = atom_messages

        # Activation functions
        self.act_r1 = self.config.get("act_r1", "relu") 
        self.act_r2 = self.config.get("act_r2", "relu") 
        self.act_m1 = self.config.get("act_m1", "relu") 
        self.act_m2 = self.config.get("act_m2", "relu") 
        self.act_m3 = self.config.get("act_m3", "relu") 

        # Dropouts for readout
        self.d = torch.nn.Dropout(p=round(self.config.get("dropout", 0.0), 2))
        self.dd = torch.nn.Dropout(p=round(self.config.get("dropout", 0.0), 2))
        self.d1 = torch.nn.Dropout(p=round(self.config.get("dropout1", 0.0), 2))
        self.d2 = torch.nn.Dropout(p=round(self.config.get("dropout2", 0.0), 2))

        # Number of MPNNs
        self.GNN_Layers = int(self.config.get('GNN_Layers', 1))

        # Hidden size
        self.hidden_size = int(round(self.config.get('hidden_size', 200),0))

        # Input
        input_dim = arguments.node_feature_size if self.atom_messages else arguments.node_feature_size+arguments.edge_feature_size
        self.linear_i = nn.Linear(input_dim, self.hidden_size)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + arguments.edge_feature_size
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.linear_m = nn.Linear(w_h_input_size, self.hidden_size)

        self.linear_a = nn.Linear(arguments.node_feature_size+self.hidden_size, self.hidden_size)

        self.hidden_globals = self.global_size

        # FNNs for readout
        readout1_in = self.hidden_globals+self.hidden_size

        readout1_out = int(round(self.config.get('readout1_out', readout1_in+50),0))
        readout2_out = int(round(self.config.get('readout2_out', int(2/3*readout1_out)+self.num_tasks),0))

        self.linear_readout_1 = nn.Linear(readout1_in, readout1_out)
        self.linear_readout_2 = nn.Linear(readout1_out, readout2_out)
        self.linear_readout_3 = nn.Linear(readout2_out, self.num_tasks)

        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.linear_readout_1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_readout_2.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_i.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_m.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_a.weight, gain=gain)

    def state_edge_0(self, edges):
        concat = torch.cat((edges.src['v'],edges.data["e"]),1).float()
        if self.act_m1=="tanh":
            act_m1_eval =eval("torch."+self.act_m1)
        elif self.act_m1=="selu" or self.act_m1=="relu":   
            act_m1_eval =eval("torch.nn.functional."+self.act_m1)
        return {'h_0' : act_m1_eval(self.linear_i(concat))}

    def state_node_0(self, nodes):
        if self.act_m1=="tanh":
            act_m1_eval =eval("torch."+self.act_m1)
        elif self.act_m1=="selu" or self.act_m1=="relu":   
            act_m1_eval =eval("torch.nn.functional."+self.act_m1)
        return {'h_0' : act_m1_eval(self.linear_i(nodes.data["v"])),
                'h_input' : self.linear_i(nodes.data["v"])}

    def scr_edge_cat(self, edges):
        concat = torch.cat((edges.src['h'],edges.data["e"]),1).float()
        return {'out' : concat}        

    def forward(self, mol_dgl_graph, globals):
        with mol_dgl_graph.local_scope():
            mol_dgl_graph.ndata["v"]= mol_dgl_graph.ndata["v"][:,:arguments.node_feature_size]
            mol_dgl_graph.edata["e"] = mol_dgl_graph.edata["e"][:,:arguments.edge_feature_size] 

            if self.act_m2=="tanh":
                act_m2_eval =eval("torch."+self.act_m2)
            elif self.act_m2=="selu" or self.act_m2=="relu":   
                act_m2_eval =eval("torch.nn.functional."+self.act_m2)
            
            if self.act_m3=="tanh":
                act_m3_eval =eval("torch."+self.act_m3)
            elif self.act_m3=="selu" or self.act_m3=="relu":   
                act_m3_eval =eval("torch.nn.functional."+self.act_m3)

            if self.atom_messages:
                mol_dgl_graph.apply_nodes(self.state_node_0)
                mol_dgl_graph.ndata["h"] = mol_dgl_graph.ndata["h_0"]
                for i in range(self.GNN_Layers):
                    '''
                    The following code returns a feature for a node, n=v, which is the summation 
                    of concatanations of features of all w in N(v) with initial features of e_vw.
                    ''' 
                    mol_dgl_graph.apply_edges(self.scr_edge_cat)
                    mol_dgl_graph.update_all(fn.copy_e("out","m"), fn.sum("m", "temp"))
                    mol_dgl_graph.ndata["h"] = self.d(act_m2_eval(mol_dgl_graph.ndata["h_input"]+\
                                               self.linear_m(mol_dgl_graph.ndata["temp"])))
                
                mol_dgl_graph.update_all(fn.copy_u("h","m"), fn.sum("m", "m_v"))
                concat_atom_feat_m_v = torch.cat((mol_dgl_graph.ndata["v"], mol_dgl_graph.ndata["m_v"]),1)
                mol_dgl_graph.ndata["h_v"] = self.dd(act_m3_eval(self.linear_a(concat_atom_feat_m_v)))

            else:                
                mol_dgl_graph.apply_edges(self.state_edge_0)

                mol_dgl_graph.edata["h"] = mol_dgl_graph.edata["h_0"]
                mol_dgl_line_graph = dgl.line_graph(mol_dgl_graph, backtracking=False, shared=False) 

                for i in range(self.GNN_Layers):
                    '''
                    The following code returns a feature for an edge, e=vw, which is the summation 
                    of features of in_feat edges to the vertex v minus parallel edge wv.
                    ''' 
                    mol_dgl_line_graph.ndata["temp"] = mol_dgl_graph.edata["h"]
                    mol_dgl_line_graph.update_all(fn.copy_u("temp","mailbox"), fn.sum("mailbox", "temp"))
                    m_e = mol_dgl_line_graph.ndata["temp"]
                    ''''''
                    mol_dgl_graph.edata["h"] = self.d(act_m2_eval(mol_dgl_graph.edata["h_0"]+self.linear_m(m_e)))

                reverse_mol_dgl_graph = mol_dgl_graph.reverse(copy_ndata=True, copy_edata=True)
                reverse_mol_dgl_graph.update_all(fn.copy_e("h","mailbox"), fn.sum("mailbox", "m_v"))
                mol_dgl_graph.ndata["m_v"] = reverse_mol_dgl_graph.ndata["m_v"]
                concat_atom_feat_m_v = torch.cat((mol_dgl_graph.ndata["v"], mol_dgl_graph.ndata["m_v"]),1)
                mol_dgl_graph.ndata["h_v"] = self.dd(act_m3_eval(self.linear_a(concat_atom_feat_m_v)))

            if self.act_r1=="tanh":
                act_r1_eval =eval("torch."+self.act_r1)
            elif self.act_r1=="selu" or self.act_r1=="relu":   
                act_r1_eval =eval("torch.nn.functional."+self.act_r1)

            if self.act_r2=="tanh":
                act_r2_eval =eval("torch."+self.act_r2)
            elif self.act_r2=="selu" or self.act_r2=="relu":   
                act_r2_eval =eval("torch.nn.functional."+self.act_r2)

            out_feature = dgl.readout_nodes(mol_dgl_graph, "h_v", op='mean')

            if self.global_feature:
                out_feature=self.d1(act_r1_eval(self.linear_readout_1(torch.cat((out_feature,globals),1))))
            else:
                out_feature=self.d1(act_r1_eval(self.linear_readout_1(out_feature)))

            out_feature=self.d2(act_r2_eval(self.linear_readout_2(out_feature)))
            out_feature=self.linear_readout_3(out_feature)
            
            return out_feature 

    def __repr__(self):
        return "("+str(self.act_r1)+", "+ str(self.act_r2)+")"
