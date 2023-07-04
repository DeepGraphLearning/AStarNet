import os
import sys
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util


def load_vocab(dataset):
    name = dataset.config_dict()["class"]
    name = name.split(".")[-1].lower()
    path = os.path.dirname(os.path.dirname(__file__))
    vocabs = []
    for object in ["entity", "relation"]:
        vocab_file = os.path.join(path, "data", name, "%s.txt" % object)
        mapping = {}
        with open(vocab_file, "r") as fin:
            for line in fin:
                k, v = line.strip().split("\t")
                mapping[k] = v
        vocab = [mapping[t] for t in getattr(dataset, "%s_vocab" % object)]
        vocabs.append(vocab)

    return vocabs


def visualize(solver, sample, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    h_index, t_index, r_index = sample.unbind(-1)
    inverse = torch.stack([t_index, h_index, r_index + num_relation], dim=-1)
    batch = sample.unsqueeze(0)
    if sample.ndim == 1:
        vis_batch = torch.stack([sample, inverse])
    else:
        is_t_neg = (h_index == h_index[0]).all()
        vis_batch = sample[:1] if is_t_neg else inverse[:1]
    batch = batch.to(solver.device)
    vis_batch = vis_batch.to(solver.device)

    solver.model.eval()
    with torch.no_grad():
        pred, target = solver.model.predict_and_target(batch)
    if isinstance(target, tuple):
        mask, target = target
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
    else:
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1
    paths, weights, num_steps = solver.model.visualize(vis_batch)
    batch = batch.tolist()
    rankings = rankings.tolist()
    paths = paths.tolist()
    weights = weights.tolist()
    num_steps = num_steps.tolist()

    logger.warning("")
    for i in range(len(vis_batch)):
        h, t, r = vis_batch[i]
        h_token = entity_vocab[h]
        t_token = entity_vocab[t]
        r_token = relation_vocab[r % num_relation]
        if r >= num_relation:
            r_token += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_token, h_token, r_token, rankings[i]))

        for path, weight, num_step in zip(paths[i], weights[i], num_steps[i]):
            if weight == float("-inf"):
                break
            triplets = []
            for h, t, r in path[:num_step]:
                h_token = entity_vocab[h]
                t_token = entity_vocab[t]
                r_token = relation_vocab[r % num_relation]
                if r >= num_relation:
                    r_token += "^(-1)"
                triplets.append("<%s, %s, %s>" % (h_token, r_token, t_token))
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] not in ["FB15k237", "OGBLWikiKG2"]:
        raise ValueError("Visualization is not implemented for %s" % cfg.dataset["class"])

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_vocab, relation_vocab = load_vocab(dataset)

    g = torch.Generator()
    g.manual_seed(1024)
    index = torch.randperm(len(solver.test_set), generator=g)[:500].tolist()
    index = index[solver.rank::solver.world_size]
    for i in index:
        visualize(solver, solver.test_set[i], entity_vocab, relation_vocab)
