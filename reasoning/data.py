import torch

from torch_scatter import scatter_add
from torch_sparse import spmm

from torchdrug import core, data, utils

from .functional import bincount

allow_materialization = False


class VirtualTensor(object):

    def __init__(self, keys=None, values=None, index=None, input=None, shape=None, dtype=None, device=None):
        if shape is None:
            shape = index.shape + input.shape[1:]
        if index is None:
            index = torch.zeros(*shape[:1], dtype=torch.long, device=device)
        if input is None:
            input = torch.empty(1, *shape[1:], dtype=dtype, device=device)
        if keys is None:
            keys = torch.empty(0, dtype=torch.long, device=device)
        if values is None:
            values = torch.empty(0, *shape[1:], dtype=dtype, device=device)

        self.keys = keys
        self.values = values
        self.index = index
        self.input = input

    @classmethod
    def zeros(cls, *shape, dtype=None, device=None):
        input = torch.zeros(1, *shape[1:], dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def full(cls, shape, value, dtype=None, device=None):
        input = torch.full((1,) + shape[1:], value, dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)

    @classmethod
    def gather(cls, input, index):
        return cls(index=index, input=input, dtype=input.dtype, device=input.device)

    def clone(self):
        return VirtualTensor(self.keys.clone(), self.values.clone(), self.index.clone(), self.input.clone())

    @property
    def shape(self):
        return self.index.shape + self.input.shape[1:]

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    def __getitem__(self, indexes):
        if not isinstance(indexes, tuple):
            indexes = (indexes,)
        keys = indexes[0]

        assert keys.numel() == 0 or (keys.max() < len(self.index) and keys.min() >= 0)
        x = self.index[keys]
        values = self.input[(self.index[keys],) + indexes[1:]]
        if len(self.keys) > 0:
            index = torch.bucketize(keys, self.keys)
            index = index.clamp(max=len(self.keys) - 1)
            indexes = (index,) + indexes[1:]
            found = keys == self.keys[index]
            indexes = tuple(index[found] for index in indexes)
            values[found] = self.values[indexes]
        return values

    def __setitem__(self, keys, values):
        new_keys, inverse = torch.cat([self.keys, keys]).unique(return_inverse=True)
        new_values = torch.zeros(len(new_keys), *self.shape[1:], dtype=self.dtype, device=self.device)
        new_values[inverse[:len(self.keys)]] = self.values
        new_values[inverse[len(self.keys):]] = values
        self.keys = new_keys
        self.values = new_values

    def __len__(self):
        return self.shape[0]


class View(object):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [x._contiguous if isinstance(x, View) else x for x in args]
        return func(*args, **kwargs)

    @utils.cached_property
    def _contiguous(self):
        return self.contiguous()

    def is_contiguous(self, *args, **kwargs):
        return False

    @property
    def ndim(self):
        return len(self.shape)

    def __getattr__(self, name):
        return getattr(self._contiguous, name)

    def __repr__(self):
        return repr(self._contiguous)

    def __len__(self):
        return self.shape[0]


class Range(View):

    def __init__(self, end, device=None):
        self.end = end
        self.shape = (end,)
        self.device = device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return torch.arange(end, device=self.device)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            assert len(index) == 1
            index = index[0]
        return torch.as_tensor(index, device=self.device)


class Repeat(View):

    def __init__(self, input, repeats):
        super(Repeat, self).__init__()
        self.input = input
        self.repeats = repeats
        self.shape = (int(repeats) * input.shape[0],) + input.shape[1:]
        self.device = input.device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return self.input.repeat([self.repeats] + [1] * (self.input.ndim - 1))

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if index[0].numel() > 0:
            assert index[0].max() < len(self)
        index = (index[0] % len(self.input),) + index[1:]
        return self.input[index]


class RepeatInterleave(View):

    def __init__(self, input, repeats):
        super(RepeatInterleave, self).__init__()
        self.input = input
        self.repeats = repeats
        self.shape = (input.shape[0] * int(repeats),) + input.shape[1:]
        self.device = input.device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return self.input.repeat_interleave(self.repeats, dim=0)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if index[0].numel() > 0:
            assert index[0].max() < len(self)
        index = (index[0] // self.repeats,) + index[1:]
        return self.input[index]


class Add(View):

    def __init__(self, input, other):
        super(Add, self).__init__()
        self.input = input
        self.other = other
        shape = []
        for d, (i, o) in enumerate(zip(input.shape, other.shape)):
            if i != o and min(i, o) > 1:
                raise RuntimeError("The size of tensor a (%d) must match the size of tensor b (%d) at non-singleton "
                                   "dimension %d" % (i, o, d))
            shape.append(max(i, o))
        self.shape = tuple(shape)
        self.device = input.device

    def contiguous(self):
        if not allow_materialization:
            raise RuntimeError("Trying to materialize a tensor of shape (%s,)" % (self.shape,))
        return self.input.add(self.other)

    def __getitem__(self, index):
        return self.input[index] + self.other[index]


class RepeatGraph(data.PackedGraph):

    def __init__(self, graph, repeats, **kwargs):
        if not isinstance(graph, data.PackedGraph):
            graph = graph.pack([graph])
        core._MetaContainer.__init__(self, **kwargs)
        self.input = graph
        self.repeats = repeats

        # data.PackedGraph
        self.num_nodes = graph.num_nodes.repeat(repeats)
        self.num_edges = graph.num_edges.repeat(repeats)
        self.num_cum_nodes = self.num_nodes.cumsum(0)
        self.num_cum_edges = self.num_edges.cumsum(0)

        # data.Graph
        self.num_node = graph.num_node * repeats
        self.num_edge = graph.num_edge * repeats
        self.num_relation = graph.num_relation

    @property
    def _offsets(self):
        return RepeatInterleave(self.num_cum_nodes - self.num_nodes, self.input.num_edge)

    @property
    def edge_list(self):
        offsets = self.num_cum_nodes - self.num_nodes
        offsets = torch.stack([offsets, offsets, torch.zeros_like(offsets)], dim=-1)
        offsets = RepeatInterleave(offsets, self.input.num_edge)
        return Add(Repeat(self.input.edge_list, self.repeats), offsets)

    @utils.cached_property
    def adjacency(self):
        return utils.sparse_coo_tensor(self.edge_list.t(), self.edge_weight.contiguous(), self.shape)

    def edge_mask(self, index, compact=False):
        index = self._standarize_index(index, self.num_edge)
        num_edges = bincount(self.edge2graph[index], minlength=self.batch_size)
        edge_list = self.edge_list[index]
        if compact:
            node_index = edge_list[:, :2].flatten()
            node_index, inverse = node_index.unique(return_inverse=True)
            num_nodes = bincount(self.node2graph[node_index], minlength=self.batch_size)
            edge_list[:, :2] = inverse.view(-1, 2)
            data_dict, meta_dict = self.data_mask(node_index, index)
        else:
            num_nodes = self.num_nodes
            data_dict, meta_dict = self.data_mask(edge_index=index)

        return type(self.input)(edge_list, edge_weight=self.edge_weight[index], num_nodes=num_nodes,
                                num_edges=num_edges, num_relation=self.num_relation, offsets=self._offsets[index],
                                meta_dict=meta_dict, **data_dict)

    @utils.cached_property
    def neighbor_inverted_index(self):
        node_in = self.input.edge_list[:, 0]
        node_in, order = node_in.sort()
        degree_in = bincount(node_in, minlength=self.input.num_node)
        ends = degree_in.cumsum(0)
        starts = ends - degree_in
        ranges = torch.stack([starts, ends], dim=-1)
        offsets = RepeatInterleave(self.num_cum_edges - self.num_edges, self.input.num_edge)
        order = Add(Repeat(order, self.repeats), offsets)
        offsets = (self.num_cum_edges - self.num_edges).unsqueeze(-1).expand(-1, 2)
        offsets = RepeatInterleave(offsets, self.input.num_node)
        ranges = Add(Repeat(ranges, self.repeats), offsets)
        return order, ranges

    def neighbors(self, index):
        order, ranges = self.neighbor_inverted_index
        starts, ends = ranges[index].t()
        num_neighbors = ends - starts
        offsets = num_neighbors.cumsum(0) - num_neighbors
        ranges = torch.arange(num_neighbors.sum(), device=self.device)
        ranges = ranges + (starts - offsets).repeat_interleave(num_neighbors)
        edge_index = order[ranges]
        node_out = self.edge_list[edge_index, 1]
        return edge_index, node_out

    def num_neighbors(self, index):
        order, ranges = self.neighbor_inverted_index
        starts, ends = ranges[index].t()
        num_neighbors = ends - starts
        return num_neighbors

    def personalized_pagerank(self, index, alpha=0.8, num_iteration=20):
        node_in, node_out = self.input.edge_list.t()[:2]
        edge_weight = self.input.edge_weight
        edge_weight = edge_weight / (self.input.degree_in[node_in] + 1e-10)

        init = torch.zeros(self.num_node, device=self.device)
        init[index] = 1
        init = init.view(self.repeats, -1).t()
        ppr = init
        index = torch.stack([node_out, node_in])
        for i in range(num_iteration):
            ppr = spmm(index, edge_weight, self.input.num_node, self.input.num_node, ppr)
            ppr = ppr * alpha + init * (1 - alpha)
        return ppr.t().flatten()

    @utils.cached_property
    def node2graph(self):
        range = Range(self.batch_size, device=self.device)
        return RepeatInterleave(range, self.input.num_node)

    @utils.cached_property
    def edge2graph(self):
        range = Range(self.batch_size, device=self.device)
        return RepeatInterleave(range, self.input.num_edge)

    def __getattr__(self, name):
        if "input" in self.__dict__:
            attr = getattr(self.__dict__["input"], name)
            if isinstance(attr, torch.Tensor):
                return Repeat(attr, self.repeats)
            return attr
        raise AttributeError("`RepeatGraph` object has no attribute `%s`" % name)
