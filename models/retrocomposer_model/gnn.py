# this file is adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter_add
from torch_geometric.data import Batch
from torch_geometric.nn.inits import glorot, zeros



class GATConv(MessagePassing):
    def __init__(self, emb_dim, bond_features_dim, heads=4, aggr="add"):
        super(GATConv, self).__init__(aggr=aggr, node_dim=0)
        self.emb_dim = emb_dim
        self.bond_features_dim = bond_features_dim
        self.heads = heads
        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
        self.edge_embedding = torch.nn.Linear(bond_features_dim, heads * emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.act = torch.nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.bond_features_dim)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).to(torch.float32)
        edge_embeddings = self.act(self.edge_embedding(edge_attr))
        x = self.act(self.weight_linear(x))
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_j += edge_attr
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = self.act(alpha)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK="last", drop_ratio=0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.JK = JK
        self.act = torch.nn.PReLU()

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding = torch.nn.Linear(atom_feat_dim, emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GATConv(emb_dim, bond_feat_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.LayerNorm(emb_dim))

    def forward(self, x, edge_index, edge_attr, embed_input=True):
        if embed_input:
            x = self.act(self.x_embedding(x))

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(self.act(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("Unknown JK method.")

        return node_representation


class BeamNode(object):
    def __init__(self, index, hidden, log_prob, input_next):
        self.index = index
        self.hidden = hidden
        self.log_prob = log_prob
        self.input_next = input_next
        self.targets_predict = []


class RnnModel(torch.nn.Module):
    def __init__(self, prod_size, react_size, n_layers=2, embedding_dim=300, hidden_size=300,
                 gnn_feat_dim=300, center_loss_type='ce'):
        super(RnnModel, self).__init__()
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.prod_size = prod_size
        self.react_size = react_size
        self.center_loss_type = center_loss_type
        self.act = torch.nn.PReLU()

        # add one start token
        self.embedding_prod = nn.Embedding(num_embeddings=prod_size, embedding_dim=embedding_dim)
        # add one end token and one padding token
        self.embedding_react = nn.Embedding(num_embeddings=react_size + 2, embedding_dim=embedding_dim)
        self.embedding_mol = nn.Sequential(
            nn.Linear(in_features=gnn_feat_dim, out_features=hidden_size * n_layers),
            nn.PReLU(),
            nn.Dropout(p=0.3)
        )

        self.MLP_prod = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=embedding_dim, out_features=1)
        )
        self.MLP_react = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size * 2),
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_size * 2, out_features=react_size + 2)
        )
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.CE = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch, atoms_embedding, mols_embedding):
        max_len = 0
        sequences_sampled = []
        for seqs in batch.sequences:
            # randomly sample one template sequence for each reaction
            seq = random.choice(seqs)[:3]
            max_len = max(max_len, len(seq))
            # padding sequences to max length 4
            sequences_sampled.append(seq + [self.react_size] * (4 - len(seq)))

        sequences_input = torch.LongTensor(sequences_sampled).to(mols_embedding.device)
        sequences_input = sequences_input[:, :(max_len + 1)]

        # first prediction prod centers
        accu_atom_len = 0
        prod_pred_res_max = []
        prod_center_embedding = []
        loss_prod_list = []
        for i, atom_indexes in enumerate(batch.reaction_center_atom_indexes):
            atom_indexes = torch.from_numpy(atom_indexes).float().to(atoms_embedding.device)
            atoms_embedding_cur = atoms_embedding[accu_atom_len:accu_atom_len + batch.atom_len[i]]
            accu_atom_len += batch.atom_len[i]
            # sum pooling for reaction center
            reaction_center_embeddings = torch.mm(atom_indexes, atoms_embedding_cur)
            reaction_center_idx = batch.reaction_center_cands[i].index(sequences_input[i][0])
            prod_center_embedding.append(reaction_center_embeddings[reaction_center_idx])

            reaction_center_output = self.MLP_prod(reaction_center_embeddings).squeeze(dim=1)
            reaction_center_labels = batch.reaction_center_cands_labels[i]
            reaction_center_labels = torch.FloatTensor(reaction_center_labels).to(atoms_embedding.device)
            assert reaction_center_labels.sum() > 0
            if self.center_loss_type == 'ce':
                # prevent from overflowing
                logits_exp = torch.exp(reaction_center_output - reaction_center_output.max())
                center = logits_exp * reaction_center_labels
                if reaction_center_labels.sum() == 0:
                    center_min = 0.001
                else:
                    center = center[reaction_center_labels > 0]
                    center_min = center.min()
                probability = center_min / ((logits_exp * (1 - reaction_center_labels)).sum() + center_min + 1e-6)
                loss_prod_list.append(-torch.log(probability + 1e-6))
                # calculate prediction accuracy
                prob_max_idx = reaction_center_output.detach().argmax()
                # only look at the most probable one
                prod_pred_res_max.append(reaction_center_labels[prob_max_idx].item())
            elif self.center_loss_type == 'bce':
                loss_prod = self.BCE(reaction_center_output, reaction_center_labels)
                loss_prod += loss_prod * reaction_center_labels * self.pos_weight
                loss_prod_list.append(loss_prod.mean())
                # calculate prediction accuracy
                reaction_center_output = torch.sigmoid(reaction_center_output.detach())
                prob_max_idx = reaction_center_output.argmax()
                # only look at the most probable one
                prod_pred_res_max.append(reaction_center_labels[prob_max_idx].item())
            else:
                raise ValueError('unknown center_loss_type: {}'.format(self.center_loss_type))

        assert accu_atom_len == atoms_embedding.shape[0]
        loss_prod = torch.stack(loss_prod_list).mean()
        prod_center_embedding = torch.stack(prod_center_embedding).unsqueeze(dim=1)
        # embedding sequence tokens
        react_embedding = self.embedding_react(sequences_input[:, 1:-1])
        input_embedding = torch.cat((prod_center_embedding, react_embedding), dim=1)

        hidden = self.embedding_mol(mols_embedding)
        hidden = hidden.view(self.n_layers, -1, self.hidden_size)
        output, _ = self.rnn(input_embedding, hidden)

        react_pred = self.MLP_react(output)[:, :-1]
        target = sequences_input[:, 1:-1]
        react_pred = react_pred.reshape(-1, react_pred.shape[-1])
        loss_react = self.CE(react_pred, target.flatten())
        react_pred = torch.argmax(react_pred, dim=-1).view(output.shape[0], -1)
        react_pred = (react_pred == target).cpu().numpy()

        return loss_prod, loss_react, prod_pred_res_max, react_pred


    def decode(self, batch, atoms_embedding, mols_embedding, beam_size=1):
        beam_nodes = []
        hidden = self.embedding_mol(mols_embedding)
        hidden = hidden.view(self.n_layers, -1, self.hidden_size)
        # first prediction prod centers
        accu_atom_len = 0
        for i, atom_indexes in enumerate(batch.reaction_center_atom_indexes):
            atom_indexes = torch.from_numpy(atom_indexes).float().to(atoms_embedding.device)
            atoms_embedding_cur = atoms_embedding[accu_atom_len:(accu_atom_len + batch.atom_len[i])]
            accu_atom_len += batch.atom_len[i]
            reaction_center_embeddings = torch.mm(atom_indexes, atoms_embedding_cur)

            reaction_center_output = self.MLP_prod(reaction_center_embeddings).squeeze(dim=1)
            # note that some examples' labels are all zeros
            if self.center_loss_type == 'ce':
                reaction_center_output = torch.softmax(reaction_center_output, dim=0)
            elif self.center_loss_type == 'bce':
                reaction_center_output = torch.sigmoid(reaction_center_output.detach())
            else:
                raise ValueError('unknown center_loss_type: {}'.format(self.center_loss_type))

            k = min(reaction_center_output.shape[0], beam_size)
            vals, idxs = torch.topk(reaction_center_output, k=k)
            for idx, val in zip(idxs, vals):
                cnt = 0
                # may have a minimal probability
                if val > 0.001 and cnt < beam_size:
                    cnt += 1
                    prod_smarts_idx = batch.reaction_center_cands[i][idx]
                    prod_smarts_idx = torch.LongTensor([prod_smarts_idx]).to(atoms_embedding.device)
                    node = BeamNode(i, hidden[:, i:i+1], torch.log(val), prod_smarts_idx)
                    node.center_embedding = reaction_center_embeddings[idx]
                    node.targets_predict.append(prod_smarts_idx.item())
                    beam_nodes.append(node)

        assert accu_atom_len == atoms_embedding.shape[0]

        # predict react token index
        for t in range(2):
            hidden = [node.hidden for node in beam_nodes]
            hidden = torch.cat(hidden, dim=1)
            inputs = [node.input_next for node in beam_nodes]
            inputs = torch.vstack(inputs)
            # first one is the product
            if t == 0:
                inputs_embedding = [node.center_embedding for node in beam_nodes]
                inputs_embedding = torch.stack(inputs_embedding).unsqueeze(dim=1)
            else:
                inputs_embedding = self.embedding_react(inputs)
            # forward single step
            output, hidden = self.rnn(inputs_embedding, hidden)
            react_pred = self.MLP_react(output).squeeze(dim=1)
            # softmax output logits
            react_pred = torch.softmax(react_pred, dim=1)
            vals, idxs = torch.topk(react_pred, k=beam_size, dim=1)
            beam_nodes_indexed = {}
            for row, node in enumerate(beam_nodes):
                hid = hidden[:, row:row + 1]
                if node.index not in beam_nodes_indexed:
                    beam_nodes_indexed[node.index] = [[], []]
                for idx, val in zip(idxs[row], vals[row]):
                    if val < 0.001: break
                    node_cur = BeamNode(node.index, hid, node.log_prob + torch.log(val), idx)
                    node_cur.targets_predict = node.targets_predict + [idx.item()]
                    beam_nodes_indexed[node_cur.index][0].append(node_cur.log_prob)
                    beam_nodes_indexed[node_cur.index][1].append(node_cur)

            # keep only top beam_size predictions
            beam_nodes = []
            for _, queues in beam_nodes_indexed.items():
                log_probs = torch.stack(queues[0])
                vals, idxs = log_probs.topk(k=min(beam_size, log_probs.shape[0]))
                for i in idxs.tolist():
                    beam_nodes.append(queues[1][i])

        return beam_nodes


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, atom_feat_dim, bond_feat_dim, center_loss_type, fusion_function,
                 prod_word_size, react_word_size, JK="last", drop_ratio=0, graph_pooling="mean"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.fusion_function = fusion_function

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK, drop_ratio)
        self.gnn_diff = GNN(num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK, drop_ratio)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        # RNN model for transformation prediction
        self.rnn_model = RnnModel(
            prod_size=prod_word_size,
            react_size=react_word_size,
            embedding_dim=emb_dim,
            hidden_size=emb_dim,
            gnn_feat_dim=emb_dim,
            center_loss_type=center_loss_type,
        )

        multiply = self.num_layer + 1 if self.JK == "concat" else 1
        self.node_embedding = nn.Linear(emb_dim * multiply, emb_dim)
        self.graph_embedding = nn.Linear(emb_dim * multiply, emb_dim)
        self.type_embedding = nn.Embedding(num_embeddings=10, embedding_dim=emb_dim)

        self.scoring = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim // 2),
            nn.PReLU(),
            nn.Linear(emb_dim // 2, 1),
        )

        self.dense1 = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.dense2 = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.dense3 = nn.Linear(emb_dim, emb_dim - 10, bias=False)

        self.CE = nn.CrossEntropyLoss(reduction='mean')
        self.BCE = nn.BCELoss(reduction='mean')

    def from_pretrained(self, model_file, device=0):
        self.load_state_dict(torch.load(model_file, map_location='cuda:{}'.format(device)))

    def node_fusion(self, product, react):
        if self.fusion_function == 1:
            return product - react
        elif self.fusion_function == 2:
            return torch.cat((self.dense1(product - react), self.dense2(product + react)), dim=1)
        elif self.fusion_function == 3:
            react = F.normalize(react, dim=1)
            product = F.normalize(product, dim=1)
            return torch.cat((self.dense1(product - react), self.dense2(product * react)), dim=1)
        else:
            raise ValueError("unknown node fusion function:", self.fusion_function)

    def ranking(self, product, reactants, typed=False, loss_type='ce'):
        # recover reactions batch from data batch
        loss, logits_list, probs_list = [], [], []
        # prod_nodes, _ = self.run_gnn(product, typed)
        prod_nodes, prod_mols = self.run_gnn(product, typed)
        prod_graphs = Batch.to_data_list(product)
        reactants.type = product.type
        # reacts_nodes, _ = self.run_gnn(reactants, typed)
        reacts_nodes, reacts_mols = self.run_gnn(reactants, typed)
        del reactants.type
        reactants = Batch.to_data_list(reactants)
        prod_atom_len = 0
        reacts_atom_len = 0
        for idx, reacts in enumerate(reactants):
            cur_prod_nodes = prod_nodes[prod_atom_len:prod_atom_len + prod_graphs[idx].atom_len]
            prod_atom_len += prod_graphs[idx].atom_len
            cur_reacts_atom_len = reacts.atom_len.sum().item()
            cur_reacts_nodes = reacts_nodes[reacts_atom_len:reacts_atom_len + cur_reacts_atom_len]
            reacts_atom_len += cur_reacts_atom_len
            atom_len = 0
            diff_graphs = []
            for alen, indexes in zip(reacts.atom_len, reacts.product_associated_indexes):
                react_graph = cur_reacts_nodes[atom_len:atom_len + alen][indexes]
                atom_len += alen
                prod_graph = copy.deepcopy(prod_graphs[idx])
                prod_graph.x = self.node_fusion(cur_prod_nodes, react_graph)
                diff_graphs.append(prod_graph)
            assert atom_len == cur_reacts_atom_len

            batch = Batch.from_data_list(diff_graphs)
            if typed:
                type_feat_onehot = torch.eye(10, dtype=torch.float32)[prod_graph.type - 1]
                type_feat_onehot = type_feat_onehot.repeat(batch.x.shape[0], 1).to(batch.x.device)
                batch.x = torch.cat((self.dense3(batch.x), type_feat_onehot), dim=1)

            diff_nodes = self.gnn_diff(batch.x, batch.edge_index, batch.edge_attr, embed_input=False)
            graph_representation = self.pool(diff_nodes, batch.batch)
            graph_representation = self.graph_embedding(graph_representation)
            if typed:
                type_embedding = self.type_embedding(prod_graph.type - 1)
                graph_representation += type_embedding

            product_embed = prod_mols[idx].view(1, -1).repeat(graph_representation.shape[0], 1)
            reactants_embed = reacts_mols[idx].view(1, -1).repeat(graph_representation.shape[0], 1)
            graph_representation = torch.cat((graph_representation, product_embed, reactants_embed), dim=1)
            scores = self.scoring(graph_representation).reshape(1, -1)
            logits_list.append(scores.detach().cpu())

            probs = torch.exp(reacts.log_prob.float()).to(scores.device)

            if loss_type == 'ce':
                # use cross entropy loss for training
                # first reactant is the ground-truth
                targets = torch.LongTensor([0]).to(scores.device)
                loss.append(F.cross_entropy(scores, targets))
                scores = torch.softmax(scores, dim=1)
                if not self.training:
                    scores = 0.5 * scores + 0.5 * probs
            elif loss_type == 'bce':
                scores = torch.sigmoid(scores)
                targets = torch.FloatTensor([[0] + [1] * (scores.shape[1] - 1)]).to(scores.device)
                loss.append(self.BCE(scores, targets))
                if not self.training:
                    scores = 0.5 * scores + 0.5 * probs
            else:
                raise ValueError('unknown loss type:', loss_type)

            probs_list.append(scores.detach().cpu())
        assert prod_atom_len == prod_nodes.shape[0]
        assert reacts_atom_len == reacts_nodes.shape[0]

        return loss, logits_list, probs_list


    def run_gnn(self, batch, typed=False):
        node_feat = batch.x
        if typed:
            type_feat = batch.type[batch.batch]
            type_feat_onehot = torch.eye(10, dtype=torch.float32).to(node_feat.device)[type_feat - 1]
            node_feat = torch.cat((batch.x, type_feat_onehot), dim=1)

        # get node representations by gnn embedding
        node_representation = self.gnn(node_feat, batch.edge_index, batch.edge_attr)
        graph_representation = self.pool(node_representation, batch.batch)
        graph_representation = self.graph_embedding(graph_representation)
        node_representation = self.node_embedding(node_representation)
        if typed:
            type_embedding = self.type_embedding(batch.type - 1)
            graph_representation += type_embedding

        return node_representation, graph_representation

    def forward(self, batch, typed=False, decode=False, beam_size=1):
        node_representation, graph_representation = self.run_gnn(batch, typed)
        # if decoding mode
        if decode:
            beam_nodes = self.rnn_model.decode(batch, node_representation, graph_representation, beam_size=beam_size)
            for node in beam_nodes:
                del node.hidden
            return beam_nodes
        else:
            res = self.rnn_model(batch, node_representation, graph_representation)
            return res


if __name__ == "__main__":
    pass

