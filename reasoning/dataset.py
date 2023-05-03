import os
import csv
import glob
from tqdm import tqdm
from ogb import linkproppred

import torch
from torch.utils import data as torch_data

from torchdrug import data, datasets, utils
from torchdrug.core import Registry as R


class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    def load_inductive_tsvs(self, transductive_files, inductive_files, verbose=0):
        assert len(transductive_files) == len(inductive_files) == 3
        inv_transductive_vocab = {}
        inv_inductive_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in transductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_transductive_vocab:
                        inv_transductive_vocab[h_token] = len(inv_transductive_vocab)
                    h = inv_transductive_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_transductive_vocab:
                        inv_transductive_vocab[t_token] = len(inv_transductive_vocab)
                    t = inv_transductive_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in inductive_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_inductive_vocab:
                        inv_inductive_vocab[h_token] = len(inv_inductive_vocab)
                    h = inv_inductive_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_inductive_vocab:
                        inv_inductive_vocab[t_token] = len(inv_inductive_vocab)
                    t = inv_inductive_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        transductive_vocab, inv_transductive_vocab = self._standarize_vocab(None, inv_transductive_vocab)
        inductive_vocab, inv_inductive_vocab = self._standarize_vocab(None, inv_inductive_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.fact_graph = data.Graph(triplets[:num_samples[0]],
                                     num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        self.graph = data.Graph(triplets[:sum(num_samples[:3])],
                                num_node=len(transductive_vocab), num_relation=len(relation_vocab))
        self.inductive_fact_graph = data.Graph(triplets[sum(num_samples[:3]): sum(num_samples[:4])],
                                               num_node=len(inductive_vocab), num_relation=len(relation_vocab))
        self.inductive_graph = data.Graph(triplets[sum(num_samples[:3]):],
                                          num_node=len(inductive_vocab), num_relation=len(relation_vocab))
        self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:4]):])
        self.num_samples = num_samples[:2] + [sum(num_samples[4:])]
        self.transductive_vocab = transductive_vocab
        self.inductive_vocab = inductive_vocab
        self.relation_vocab = relation_vocab
        self.inv_transductive_vocab = inv_transductive_vocab
        self.inv_inductive_vocab = inv_inductive_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("dataset.FB15k237Inductive")
class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


@R.register("dataset.WN18RRInductive")
class WN18RRInductive(InductiveKnowledgeGraphDataset):

    transductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
    ]

    inductive_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        transductive_files = []
        for url in self.transductive_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            transductive_files.append(txt_file)
        inductive_files = []
        for url in self.inductive_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            inductive_files.append(txt_file)

        self.load_inductive_tsvs(transductive_files, inductive_files, verbose=verbose)


@R.register("dataset.OGBLWikiKG2")
class OGBLWikiKG2(data.KnowledgeGraphDataset):
    """
    OGBLWikiKG2(
        #entity: 2,500,604
        #relation: 535
        #triplet: 17,137,181
    )
    #train: 16,109,182, #valid: 858,912, #test: 1,197,086
    """

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        dataset = linkproppred.LinkPropPredDataset("ogbl-wikikg2", path)
        self.load_ogb(dataset, verbose=verbose)

    def load_ogb(self, dataset, verbose=1):
        inv_entity_vocab = {}
        inv_relation_vocab = {}

        zip_files = glob.glob(os.path.join(dataset.root, "mapping/*.csv.gz"))
        for zip_file in zip_files:
            csv_file = utils.extract(zip_file)
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin)
                if verbose:
                    reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
                fields = next(reader)
                if "reltype" in csv_file:
                    for index, token in reader:
                        inv_relation_vocab[token] = int(index)
                elif "nodeidx" in csv_file:
                    for index, token in reader:
                        inv_entity_vocab[token] = int(index)
                else:
                    raise RuntimeError("Unknown mapping file `%s`" % csv_file)

        edge_split = dataset.get_edge_split()
        triplets = []
        num_samples = []
        num_samples_with_neg = []
        negative_heads = []
        negative_tails = []
        for key in ["train", "valid", "test"]:
            split_dict = edge_split[key]
            h = torch.as_tensor(split_dict["head"])
            t = torch.as_tensor(split_dict["tail"])
            r = torch.as_tensor(split_dict["relation"])

            triplet = torch.stack([h, t, r], dim=-1)
            triplets.append(triplet)
            num_samples.append(len(triplet))
            if "head_neg" in split_dict:
                neg_h = torch.as_tensor(split_dict["head_neg"])
                neg_t = torch.as_tensor(split_dict["tail_neg"])
                negative_heads.append(neg_h)
                negative_tails.append(neg_t)
                num_samples_with_neg.append(len(neg_h))
            else:
                num_samples_with_neg.append(0)
        triplets = torch.cat(triplets)

        self.load_triplet(triplets, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab)
        self.num_samples = num_samples
        self.num_samples_with_neg = num_samples_with_neg
        self.negative_heads = torch.cat(negative_heads)
        self.negative_tails = torch.cat(negative_tails)
        self.name = dataset.name

    def split(self, test_negative=True):
        offset = 0
        neg_offset = 0
        splits = []
        for num_sample, num_sample_with_neg in zip(self.num_samples, self.num_samples_with_neg):
            if test_negative and num_sample_with_neg:
                triplets = self[offset: offset + num_sample]
                negative_heads = self.negative_heads[neg_offset: neg_offset + num_sample_with_neg]
                negative_tails = self.negative_tails[neg_offset: neg_offset + num_sample_with_neg]
                split = OGBLKGTest(triplets, negative_heads, negative_tails)
            else:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
            neg_offset += num_sample_with_neg
        return splits


class OGBLKGTest(torch_data.Dataset):

    def __init__(self, triplets, negative_heads, negative_tails):
        self.triplets = triplets
        self.negative_heads = negative_heads
        self.negative_tails = negative_tails
        self.num_negative = negative_heads.shape[-1]

    def __getitem__(self, index):
        assert isinstance(index, int)

        is_t_neg = index // len(self.triplets) == 0
        index = index % len(self.triplets)
        triplet = self.triplets[index]
        triplet = triplet.repeat(self.num_negative + 1, 1)
        if is_t_neg:
            triplet[1:, 1] = self.negative_tails[index]
        else:
            triplet[1:, 0] = self.negative_heads[index]
        return triplet

    def __len__(self):
        return len(self.triplets) * 2