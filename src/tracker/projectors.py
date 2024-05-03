import numpy as np

import torch
from torch_scatter import scatter_add
from torch_geometric.utils import unbatch_edge_index, is_undirected

from src.data.graph import Graph
from src.utils.evaluation import compute_constr_satisfaction_rate

import pulp as plp
from scipy.optimize import linear_sum_assignment
from lapsolver import solve_dense

class GreedyProjector:
    """
    Applies the greedy rounding scheme described in https://arxiv.org/pdf/1912.07515.pdf, Appending B.1
    """
    def __init__(self, full_graph):
        self.final_graph = full_graph
        self.num_nodes = full_graph.num_nodes

    def project(self, threshold):
        round_preds = (self.final_graph.edge_preds > threshold).float()

        self.constr_satisf_rate, flow_in, flow_out =compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        # Determine the set of constraints that are violated
        nodes_names = torch.arange(self.num_nodes).to(flow_in.device)
        in_type = torch.zeros(self.num_nodes).to(flow_in.device)  # Violation type
        out_type = torch.ones(self.num_nodes).to(flow_in.device)  # Violation type

        # Merge the violations and their types
        flow_in_info = torch.stack((nodes_names.float(), in_type.float())).t()
        flow_out_info = torch.stack((nodes_names.float(), out_type.float())).t()
        all_violated_constr = torch.cat((flow_in_info, flow_out_info))

        # Nodes with a violation
        mask = torch.cat((flow_in > 1, flow_out > 1))

        # Sort violated constraints by the value of thei maximum pred value among incoming / outgoing edges
        all_violated_constr = all_violated_constr[mask]
        vals, sorted_ix = torch.sort(all_violated_constr[:, 1], descending=True)
        all_violated_constr = all_violated_constr[sorted_ix]

        # Iterate over violated constraints.
        for viol_constr in all_violated_constr:
            node_name, viol_type = viol_constr

            # Determine the set of incoming / outgoing edges
            mask = torch.zeros(self.num_nodes).bool()
            mask[node_name.int()] = True
            if viol_type == 0:  # Flow in violation
                mask = mask[self.final_graph.edge_index[1]]

            else:  # Flow Out violation
                mask = mask[self.final_graph.edge_index[0]]

            # Get the ixs in which there is a violation for the corresponding node
            flow_edges_ix = torch.where(mask)[0]

            # If the constraint is still violated, set to 1 the edge with highest score, and set the rest to 0
            if round_preds[flow_edges_ix].sum() > 1:
                max_pred_ix = max(flow_edges_ix, key=lambda ix: self.final_graph.edge_preds[ix]*round_preds[ix]) # Multiply for round_preds so that if the edge has been set to 0
                                                                                                                 # it can not be set back to 1
                # Set the max to 1 and everything else to zero
                round_preds[mask] = 0
                round_preds[max_pred_ix] = 1

        # Assert that there are no constraint violations
        assert scatter_add(round_preds, self.final_graph.edge_index[1], dim_size=self.num_nodes).max() <= 1
        assert scatter_add(round_preds, self.final_graph.edge_index[0], dim_size=self.num_nodes).max() <= 1

        # return round_preds, constr_satisf_rate
        self.final_graph.edge_preds = round_preds

class ExactProjector:
    """
    Constructs a Subgraph with all nodes in a graph that are involved in a violated constraint
    (e.g. their incoming / outgoing flow is >1), and then rounds solutions with a MCF Linear Program.
    the full approach is explained in https://arxiv.org/pdf/1912.07515.pdf, Appending B.2

    """
    def __init__(self, full_graph, solver_backend = 'pulp'):
        # 1. Determine for which edges we can directly get the values
        self.final_graph = full_graph
        self.num_nodes = full_graph.num_nodes
        self.solver_backend = solver_backend

    def project(self, threshold):
        round_preds = (self.final_graph.edge_preds > threshold).float()

        self.constr_satisf_rate, flow_in, flow_out =compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        #self.constr_satisf_rate = 1 - ((flow_in > 1).sum() + (flow_out > 1).sum()).float() / (self.num_nodes*2)

        # Concat all violated_constraint info
        nodes_mask = (flow_in > 1) | (flow_out >1)
        edges_mask = nodes_mask[self.final_graph.edge_index[0]] | nodes_mask[self.final_graph.edge_index[1]]
        if edges_mask.sum() > 0:
            graph_to_project = Graph()
            graph_to_project.edge_preds = self.final_graph.edge_preds[edges_mask]
            graph_to_project.edge_index = self.final_graph.edge_index.T[edges_mask].T
            graph_to_project.node_names = self.final_graph.node_names[nodes_mask]
            graph_to_project.node_preds = torch.zeros_like(graph_to_project.node_names)

            if self.solver_backend == 'gurobi':
                #mcf_solver = GurobiMinCostFlowSolver(graph_to_project.numpy())
                raise Exception('Uncomment gurobi code to run gorubi solver')
            else:
                mcf_solver = PuLPMinCostFlowSolver(graph_to_project.numpy())

            mcf_solver.solve()

            # Assign the right values to the original graph's predictions
            self.final_graph.edge_preds = self.final_graph.edge_preds.cpu().numpy()
            edges_mask = edges_mask.cpu().numpy()
            self.final_graph.edge_preds[~edges_mask] = round_preds[~edges_mask].cpu().numpy()
            self.final_graph.edge_preds[edges_mask] = graph_to_project.edge_preds
        else:
            self.final_graph.edge_preds = self.final_graph.edge_preds.cpu().numpy()
            edges_mask = edges_mask.cpu().numpy()
            self.final_graph.edge_preds[~edges_mask] = round_preds[~edges_mask].cpu().numpy()

        assert np.all(np.logical_or(self.final_graph.edge_preds == 1, self.final_graph.edge_preds < 0.01)), "Exact rounding has floating point errors"

class HungarianProjector:

    def __init__(self, full_graph):
        self.final_graph = full_graph
        self.num_nodes = full_graph.num_nodes
    
    def project(self, threshold):
        
        round_preds = (self.final_graph.edge_preds > threshold).float()
        self.constr_satisf_rate, flow_in, flow_out =compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        # First we need to unbatch the bipartite graphs in a Hierarchic Graph
        node_batch_labels = self.final_graph.node_batch_labels.squeeze()
        n_edges = self.final_graph.edge_index.shape[1]

        unbatched_edge_idx = unbatch_edge_index(self.final_graph.edge_index, node_batch_labels.to(torch.int64))
        round_preds = torch.zeros((n_edges,), device = self.final_graph.edge_index.device)
        prev_length = 0

        for i_ in range(len(unbatched_edge_idx)):
            n_nodes = sum(node_batch_labels == i_)
            bipartite_edges = unbatched_edge_idx[i_]
            if bipartite_edges.nelement() == 0:
                continue

            left_node_ixs  = torch.unique(bipartite_edges[0, :])
            left_node_ixs  = torch.arange(0, left_node_ixs.max()+1)
            n_left_nodes = left_node_ixs.size(0)

            right_node_ixs = torch.unique(bipartite_edges[1, :])
            right_node_ixs = torch.arange(n_left_nodes, n_nodes)
            n_right_nodes = right_node_ixs.size(0)

            edge_index_tuple = [tuple(bipartite_edges[:,i].tolist()) for i in range(bipartite_edges.shape[1])]
            all_edges_tuple  = [(i,n_left_nodes+j) for i in range(n_left_nodes) for j in range(n_right_nodes)]
            missing_edges = list(set(all_edges_tuple) - set(edge_index_tuple))

            rows = [missing_edges[k_][0] for k_ in range(len(missing_edges))]
            cols = [missing_edges[k_][1] - n_left_nodes  for k_ in range(len(missing_edges))]

            similarity_mat = torch.zeros((n_left_nodes, n_right_nodes), device = self.final_graph.edge_index.device)
            similarity_mat[bipartite_edges[0], bipartite_edges[1] - n_left_nodes] = self.final_graph.edge_preds[prev_length: prev_length + bipartite_edges.shape[1]]
            similarity_mat[rows, cols] = 0

            row_ind, col_ind = solve_dense(1-similarity_mat.cpu())
            assignment_mat = torch.zeros((n_left_nodes, n_right_nodes), device = self.final_graph.edge_index.device)
            assignment_mat[row_ind, col_ind] = 1

            preds_ = assignment_mat[bipartite_edges[0], bipartite_edges[1] - n_left_nodes]
    
            round_preds[prev_length: prev_length + bipartite_edges.shape[1]] = preds_
            prev_length += bipartite_edges.shape[1]
        
        self.final_graph.edge_preds = round_preds

class HungarianProjector_full:

    def __init__(self, full_graph):
        self.final_graph = full_graph
        self.num_nodes = full_graph.num_nodes
    
    def project(self, threshold):
        round_preds = (self.final_graph.edge_preds > threshold).float()
        self.constr_satisf_rate, flow_in, flow_out =compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        
        # Apply Threshold to Edge Predictions
        edges_mask = self.final_graph.edge_preds >= threshold
        edge_index = self.final_graph.edge_index.T[edges_mask].T
        edge_preds = self.final_graph.edge_preds[edges_mask]

        num_nodes = self.final_graph.bipartite_labels.shape[0]
        
        if is_undirected(self.final_graph.edge_index):
            directed_mask = edge_index[1, :] > edge_index[0, :]
            edge_index = edge_index.T[directed_mask].T
            edge_preds = edge_preds[directed_mask]

        sim_mat = torch.full((num_nodes,num_nodes), torch.nan, device = edge_index.device)
        sim_mat[edge_index[0], edge_index[1]] = edge_preds

        row_ind, col_ind = solve_dense(1-sim_mat.cpu())
        
        res_mat = torch.full((num_nodes,num_nodes), torch.tensor(0.), device = edge_index.device)
        res_mat[row_ind, col_ind] = 1

        round_preds = res_mat[edge_index[0], edge_index[1]]

        self.final_graph.edge_index = edge_index
        self.final_graph.edge_preds = round_preds


class PuLPMinCostFlowSolver:
    """
    See https://arxiv.org/pdf/1912.07515.pdf, Appending B.2
    """
    def __init__(self, graph):
        assert (graph.edge_index[0] < graph.edge_index[1]).all(), "Cannot project a graph with duplicated edges!"

        self.graph = graph
        self.edges = [tuple(edge) for edge in self.graph.edge_index.T]
        self.nodes = np.unique(self.graph.edge_index)

        self.edge_vals = {edge: edge_val for edge, edge_val in zip(self.edges, self.graph.edge_preds)}

    def _add_constraints(self):
        max_node = self.nodes.max()
        for node in self.nodes:

            node_mask = np.zeros(max_node + 1).astype(bool)
            node_mask[node] = True

            in_flow_mask = node_mask[self.graph.edge_index[1]]
            in_flow_edges = self.graph.edge_index.T[in_flow_mask]
            inc_flow = plp.lpSum((self.model_vars[tuple(edge)] for edge in in_flow_edges))
            self.m.addConstraint(plp.LpConstraint(e=inc_flow, sense=plp.LpConstraintLE, rhs=1))

            out_flow_mask = node_mask[self.graph.edge_index[0]]
            out_flow_edges = self.graph.edge_index.T[out_flow_mask]
            out_flow = plp.lpSum((self.model_vars[tuple(edge)] for edge in out_flow_edges))
            self.m.addConstraint(plp.LpConstraint(e=out_flow, sense=plp.LpConstraintLE, rhs=1))

    def solve(self):
        # Define Model, variables (i.e. edges in the subgraph)
        self.m = plp.LpProblem(name='MinCostFlowLP')
        self.model_vars = {edge: plp.LpVariable(lowBound=0, upBound=1, cat=plp.LpContinuous, name=str(edge)) for edge in self.edges}

        # Define rounding objective
        obj = plp.lpSum(self.model_vars[edge] * (1 - 2 * edge_val) for edge, edge_val in self.edge_vals.items())
        self.m.sense = plp.LpMinimize
        self.m.setObjective(obj)

        # Add flow constraints and solve problem
        self._add_constraints()
        self.m.solve()

        self.graph.edge_preds = np.array([self.model_vars[edge].varValue for edge in self.edges])

        return self.graph


