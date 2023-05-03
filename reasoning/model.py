import math

import torch
from torch import nn, autograd
from torch.nn import functional as F

from torch_scatter import segment_add_coo, scatter_add, scatter_max

from torchdrug import core, data, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import VirtualTensor, Range, RepeatGraph
from .functional import bincount, variadic_topks


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, base_layer, num_layer, short_cut=False, concat_hidden=False, num_mlp_layer=2,
                 remove_one_hop=False, shared_graph=True, edge_dropout=0, num_beam=10, path_topk=10):
        super(NeuralBellmanFordNetwork, self).__init__()

        self.num_relation = base_layer.num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.shared_graph = shared_graph
        self.num_beam = num_beam
        self.path_topk = path_topk

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = None

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(core.Configurable.load_config_dict(base_layer.config_dict()))
        feature_dim = base_layer.output_dim * (num_layer if concat_hidden else 1) + base_layer.input_dim
        self.query = nn.Embedding(base_layer.num_relation * 2, base_layer.input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def search(self, graph, h_index, r_index, edge_grad=False, all_loss=None, metric=None):
        query = self.query(r_index)
        boundary = self.indicator(graph, h_index, query)
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        graphs = []
        layer_input = boundary
        for layer in self.layers:
            if edge_grad:
                graph = graph.clone().detach().requires_grad_()
            hidden = layer(graph, layer_input)
            if self.short_cut:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            graphs.append(graph)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        if isinstance(graph, data.PackedGraph):
            node_query = query.repeat_interleave(graph.num_nodes, dim=0)
        else:
            node_query = query.expand(graph.num_node, -1, -1)
        score = self.score(hidden, node_query)

        return {
            "node_feature": hidden,
            "node_score": score,
            "step_graphs": graphs,
        }

    def indicator(self, graph, index, query):
        if isinstance(graph, data.PackedGraph):
            boundary = torch.zeros(graph.num_node, *query.shape[1:], device=self.device)
            boundary[index] = query
        else:
            boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
            index = index.unsqueeze(-1).expand_as(query)
            boundary.scatter_(0, index.unsqueeze(0), query.unsqueeze(0))
        return boundary

    def score(self, hidden, node_query):
        hidden = torch.cat([hidden, node_query], dim=-1)
        score = self.mlp(hidden).squeeze(-1)
        return score

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        if self.edge_dropout:
            graph = graph.clone()
            graph._edge_weight = self.edge_dropout(graph.edge_weight)
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        if not self.shared_graph:
            batch_size = len(h_index)
            graph = RepeatGraph(graph, batch_size)
            offset = graph.num_cum_nodes - graph.num_nodes
            h_index = h_index + offset.unsqueeze(-1)
            t_index = t_index + offset.unsqueeze(-1)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.search(graph, h_index[:, 0], r_index[:, 0], all_loss=all_loss, metric=metric)
        score = output["node_score"]
        if self.shared_graph:
            score = score.transpose(0, 1).gather(1, t_index)
        else:
            score = score[t_index]

        return score

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)
        batch_size = len(h_index)
        graph = graph.repeat(batch_size)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset
        t_index = t_index + offset

        output = self.search(graph, h_index, r_index, edge_grad=True)
        score = output["node_score"]
        graphs = output["step_graphs"]
        score = score[t_index]
        edge_weights = [graph.edge_weight for graph in graphs]
        edge_grads = autograd.grad(score.sum(), edge_weights)
        for graph, edge_grad in zip(graphs, edge_grads):
            with graph.edge():
                graph.edge_grad = edge_grad
        lengths, source_indexes = self.beam_search_length(graphs, h_index, t_index)
        paths, weights, num_steps = self.topk_average_length(graph, lengths, source_indexes, t_index)

        return paths, weights, num_steps

    def beam_search_length(self, graphs, h_index, t_index):
        inf = float("inf")
        input = torch.full((graphs[0].num_node, self.num_beam), -inf, device=self.device)
        input[h_index, 0] = 0

        lengths = []
        source_indexes = []
        for graph in graphs:
            edge_mask = graph.edge_list[:, 0] != t_index[graph.edge2graph]
            node_in, node_out = graph.edge_list[edge_mask, :2].t()

            message = input[node_in] + graph.edge_grad[edge_mask].unsqueeze(-1)
            edge_index = torch.arange(graph.num_edge, device=self.device)[edge_mask]
            beam_index = torch.arange(self.num_beam, device=self.device)
            edge_index, beam_index = torch.meshgrid(edge_index, beam_index)
            source_index = torch.stack([edge_index, beam_index], dim=-1)

            node_out, order = node_out.sort()
            num_messages = bincount(node_out, minlength=graph.num_node) * self.num_beam
            message = message[order].flatten()
            source_index = source_index[order].flatten(0, -2)
            ks = num_messages.clamp(max=self.num_beam)
            length, index = variadic_topks(message, num_messages, ks)
            source_index = source_index[index]
            length = functional.variadic_to_padded(length, ks, value=-inf)[0]
            source_index = functional.variadic_to_padded(source_index, ks)[0]

            lengths.append(length)
            source_indexes.append(source_index)
            input = length

        return lengths, source_indexes

    def topk_average_length(self, graph, lengths, source_indexes, t_index):
        num_layer = len(self.layers)
        weights = []
        num_steps = []
        beam_indexes = []
        for i, length in enumerate(lengths):
            weight = length[t_index] / (i + 1)
            num_step = torch.full(weight.shape, i + 1, device=self.device)
            beam_index = torch.arange(self.num_beam, device=self.device).expand_as(weight)
            weights.append(weight)
            num_steps.append(num_step)
            beam_indexes.append(beam_index)
        weights = torch.cat(weights, dim=-1)
        num_steps = torch.cat(num_steps, dim=-1)
        beam_index = torch.cat(beam_indexes, dim=-1)
        weights, index = weights.topk(self.path_topk)
        num_steps = num_steps.gather(-1, index)
        beam_index = beam_index.gather(-1, index)

        paths = []
        t_index = t_index.unsqueeze(-1).expand_as(beam_index)
        for i in range(num_layer)[::-1]:
            mask = num_steps > i
            edge_index, new_beam_index = source_indexes[i][t_index, beam_index].unbind(dim=-1)
            edge_index = torch.where(mask, edge_index, 0)
            edges = graph.edge_list[edge_index]
            edges[:, :, :2] -= graph._offsets[edge_index].unsqueeze(-1)
            paths.append(edges)
            t_index = torch.where(mask, graph.edge_list[edge_index, 0], t_index)
            beam_index = torch.where(mask, new_beam_index, beam_index)
        paths = torch.stack(paths[::-1], dim=-2)

        return paths, weights, num_steps

    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            any = -torch.ones_like(h_index_ext)
            pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
        else:
            pattern = torch.stack([h_index, t_index, r_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index


@R.register("model.AStarNet")
class AStarNetwork(NeuralBellmanFordNetwork, core.Configurable):

    def __init__(self, base_layer, num_layer, indicator_func="onehot", short_cut=False, num_mlp_layer=2,
                 num_indicator_bin=10, node_ratio=0.1, degree_ratio=1, test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, **kwargs):
        for k in ["concat_hidden", "shared_graph"]:
            if k in kwargs:
                raise TypeError("`%s` is not supported by AStarNet" % k)
        super(AStarNetwork, self).__init__(base_layer, num_layer, short_cut, num_mlp_layer=num_mlp_layer,
                                           shared_graph=False, **kwargs)

        assert not self.concat_hidden
        self.indicator_func = indicator_func
        self.num_indicator_bin = num_indicator_bin
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie

        if indicator_func == "ppr":
            self.distance = nn.Embedding(num_indicator_bin, base_layer.input_dim)
        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        self.mlp = layers.MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        ks = (node_ratio * graph.num_nodes).long()
        es = (degree_ratio * ks * graph.num_edges / graph.num_nodes).long()

        node_in = score.keys
        num_nodes = bincount(graph.node2graph[node_in], minlength=len(graph))
        ks = torch.min(ks, num_nodes)
        score_in = score[node_in]
        index = variadic_topks(score_in, num_nodes, ks=ks, break_tie=self.break_tie)[1]
        node_in = node_in[index]
        num_nodes = ks

        num_neighbors = graph.num_neighbors(node_in)
        num_edges = scatter_add(num_neighbors, graph.node2graph[node_in], dim_size=len(graph))
        es = torch.min(es, num_edges)
        # chunk batch to reduce peak memory usage
        chunk_size = max(int(1e7 / num_edges.float().mean()), 1)
        num_nodes = num_nodes.split(chunk_size)
        num_edges = num_edges.split(chunk_size)
        es = es.split(chunk_size)
        num_chunk_nodes = [num_node.sum() for num_node in num_nodes]
        node_ins = node_in.split(num_chunk_nodes)

        edge_indexes = []
        for node_in, num_node, num_edge, e in zip(node_ins, num_nodes, num_edges, es):
            edge_index, node_out = graph.neighbors(node_in)
            score_edge = score[node_out]
            index = variadic_topks(score_edge, num_edge, ks=e, break_tie=self.break_tie)[1]
            edge_index = edge_index[index]
            edge_indexes.append(edge_index)
        edge_index = torch.cat(edge_indexes)

        return edge_index

    def search(self, graph, h_index, r_index, all_loss=None, metric=None):
        query = self.query(r_index)
        boundary, score = self.indicator(graph, h_index, query)
        hidden = boundary.clone()
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary
            graph.hidden = hidden
            graph.score = score
            graph.node_id = Range(graph.num_node, device=self.device)
            graph.pna_degree_out = graph.degree_out
        with graph.edge():
            graph.edge_id = Range(graph.num_edge, device=self.device)
        pna_degree_mean = (graph[0].degree_out + 1).log().mean()

        num_nodes = []
        num_edges = []
        subgraphs = []
        for layer in self.layers:
            edge_index = self.select_edges(graph, graph.score)
            subgraph = graph.edge_mask(edge_index, compact=True)
            subgraph.pna_degree_mean = pna_degree_mean

            if subgraph.num_node > 0:
                layer_input = F.sigmoid(subgraph.score).unsqueeze(-1) * subgraph.hidden
                hidden = layer(subgraph, layer_input)

            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]
            if self.short_cut:
                graph.hidden[node_out] = graph.hidden[node_out] + hidden[out_mask]
            else:
                graph.hidden[node_out] = hidden[out_mask]
            index = graph.node2graph[node_out]
            graph.score[node_out] = self.score(graph.hidden[node_out], query[index])
            # update graph-level attributes
            data_dict, meta_dict = subgraph.data_by_meta("graph")
            graph.meta_dict.update(meta_dict)
            graph.__dict__.update(data_dict)

            num_nodes.append(subgraph.num_nodes.float().mean())
            num_edges.append(subgraph.num_edges.float().mean())
            subgraphs.append(subgraph)

        if metric is not None:
            metric["#node per layer"] = torch.stack(num_nodes).mean()
            metric["#edge per layer"] = torch.stack(num_edges).mean()

        return {
            "node_feature": graph.hidden,
            "node_score": graph.score,
            "step_graphs": subgraphs,
        }

    def indicator(self, graph, index, query):
        if self.indicator_func == "onehot":
            boundary = VirtualTensor.zeros(graph.num_node, query.shape[1], device=self.device)
            boundary[index] = query
            score = VirtualTensor.gather(self.score(torch.zeros_like(query), query), graph.node2graph)
            score[index] = self.score(query, query)
        elif self.indicator_func == "ppr":
            ppr = graph.personalized_pagerank(index)
            bin = torch.logspace(-1, 0, self.num_indicator_bin, base=graph.num_node, device=self.device)
            bin_index = torch.bucketize(ppr, bin)
            distance = self.distance.weight
            boundary = VirtualTensor.gather(distance, bin_index)
            boundary[index] = query
            hidden = distance.repeat(len(graph), 1)
            node_query = query.repeat_interleave(self.num_indicator_bin, dim=0)
            score_index = bin_index + torch.repeat_interleave(graph.num_nodes) * self.num_indicator_bin
            score = VirtualTensor.gather(self.score(hidden, node_query), score_index)
            score[index] = self.score(query, query)
        else:
            raise ValueError("Unknown indicator function `%s`" % self.indicator_func)
        return boundary, score

    def score(self, hidden, node_query):
        heuristic = self.linear(torch.cat([hidden, node_query], dim=-1))
        x = self.layers[0].compute_message(hidden, heuristic)
        score = self.mlp(x).squeeze(-1)
        return score

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        if self.training:
            return super(AStarNetwork, self).forward(graph, h_index, t_index, r_index, all_loss, metric)

        # adjust batch size for test node ratio
        num_chunk = math.ceil(self.test_node_ratio / self.node_ratio / 5)
        h_indexes = h_index.chunk(num_chunk)
        t_indexes = t_index.chunk(num_chunk)
        r_indexes = r_index.chunk(num_chunk)
        scores = []
        for h_index, t_index, r_index in zip(h_indexes, t_indexes, r_indexes):
            score = super(AStarNetwork, self).forward(graph, h_index, t_index, r_index, all_loss, metric)
            scores.append(score)
        score = torch.cat(scores)
        return score

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)
        batch_size = len(h_index)
        graph = RepeatGraph(graph, batch_size)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset
        t_index = t_index + offset

        output = self.search(graph, h_index, r_index)
        subgraphs = output["step_graphs"]
        lengths, source_indexes = self.beam_search_length(graph, subgraphs, graph.num_node, h_index, t_index)
        paths, weights, num_steps = self.topk_average_length(graph, lengths, source_indexes, t_index)

        return paths, weights, num_steps

    def beam_search_length(self, graph, subgraphs, num_node, h_index, t_index):
        inf = float("inf")
        input = VirtualTensor.full((num_node, self.num_beam), -inf, device=self.device)
        init = torch.full((len(h_index), self.num_beam), -inf, device=self.device)
        init[:, 0] = 0
        input[h_index] = init

        lengths = []
        source_indexes = []
        for subgraph in subgraphs:
            edge_mask = subgraph.node_id[subgraph.edge_list[:, 0]] != t_index[subgraph.edge2graph]
            node_in, node_out = subgraph.edge_list[edge_mask, :2].t()

            in_mask = subgraph.degree_in > 0
            sub_input = input[subgraph.node_id]
            score = F.sigmoid(subgraph.score) * in_mask
            score = score / scatter_max(score, subgraph.node2graph)[0][subgraph.node2graph]
            message = sub_input[node_in] + score[node_in].unsqueeze(-1)
            edge_index = subgraph.edge_id[edge_mask]
            beam_index = torch.arange(self.num_beam, device=self.device)
            edge_index, beam_index = torch.meshgrid(edge_index, beam_index)
            sub_source_index = torch.stack([edge_index, beam_index], dim=-1)

            node_out, order = node_out.sort()
            num_messages = bincount(node_out, minlength=subgraph.num_node) * self.num_beam
            message = message[order].flatten()
            sub_source_index = sub_source_index[order].flatten(0, -2)
            ks = num_messages.clamp(max=self.num_beam)
            sub_length, index = variadic_topks(message, num_messages, ks)
            sub_source_index = sub_source_index[index]
            sub_length = functional.variadic_to_padded(sub_length, ks, value=-inf)[0]
            sub_source_index = functional.variadic_to_padded(sub_source_index, ks)[0]

            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]
            length = VirtualTensor.full((num_node, self.num_beam), -inf, device=self.device)
            source_index = VirtualTensor.zeros(num_node, self.num_beam, 2, dtype=torch.long, device=self.device)
            length[node_out] = sub_length[out_mask]
            source_index[node_out] = sub_source_index[out_mask]

            lengths.append(length)
            source_indexes.append(source_index)
            input = length

        return lengths, source_indexes
