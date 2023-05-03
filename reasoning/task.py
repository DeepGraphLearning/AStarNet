from ogb import linkproppred

import torch
from torch.utils import data as torch_data

from torchdrug import core, tasks
from torchdrug.core import Registry as R


@R.register("task.InductiveKnowledgeGraphCompletion")
class InductiveKnowledgeGraphCompletion(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, sample_weight=True,
                 full_batch_eval=False):
        super(InductiveKnowledgeGraphCompletion, self).__init__(
            model, criterion, metric, num_negative, margin, adversarial_temperature, strict_negative,
            sample_weight=sample_weight, filtered_ranking=True, full_batch_eval=full_batch_eval)

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("graph", dataset.input)
        self.register_buffer("fact_graph", dataset.fact_graph)
        self.register_buffer("inductive_graph", dataset.inductive_graph)
        self.register_buffer("inductive_fact_graph", dataset.inductive_fact_graph)

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)
        graph = self.inductive_fact_graph if self.split == "test" else self.fact_graph

        if all_loss is None:
            # test
            all_index = torch.arange(graph.num_node, device=self.device)
            num_negative = self.num_entity if self.full_batch_eval else self.num_negative
            t_preds = []
            h_preds = []
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        graph = self.inductive_graph if self.split == "test" else self.graph
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()


@R.register("task.KnowledgeGraphCompletionOGB")
class KnowledgeGraphCompletionOGB(tasks.KnowledgeGraphCompletion, core.Configurable):

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.evaluator = linkproppred.Evaluator(dataset.name)
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
        self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        return train_set, valid_set, test_set

    def predict(self, batch, all_loss=None, metric=None):
        batch_size = len(batch)

        if all_loss is None:
            # test
            h_index, t_index, r_index = batch.unbind(-1)
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            pos_h_index, pos_t_index, pos_r_index = batch.t()
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        target = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # in case of GPU OOM
        return target.cpu()

    def evaluate(self, pred, target):
        is_positive = torch.zeros(pred.shape, dtype=torch.bool)
        is_positive.scatter_(-1, target.unsqueeze(-1), 1)
        pos_pred = pred[is_positive]
        neg_pred = pred[~is_positive].view(len(pos_pred), -1)
        metric = self.evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})

        new_metric = {}
        for key in metric:
            new_key = key.split("_")[0]
            new_metric[new_key] = metric[key].mean()

        return new_metric