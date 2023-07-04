import os
import re
import sys
import json
import pprint
import urllib
import warnings

import jinja2
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

import torch

from torchdrug import core, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from reasoning import dataset, layer, model, task, util


module = sys.modules[__name__]
warnings.filterwarnings("ignore")
openai.api_key = os.environ["OPENAI_API_KEY"]

functions = [
    {"name": "reasoning", "description": "Predict answers to a natural language question. "
                                         "Return answers ordered by their scores.",
     "parameters": {
         "type": "object",
         "properties": {
             "query": {"type": "string",
                       "description": "Natural language question. e.g. What is the job of Joe Biden?"}
         },
         "required": ["query"]
     },},
    {"name": "explain", "description": "Explain an answer to a natural language question. "
                                       "Return reasoning paths and reference links, ordered by their scores.",
     "parameters": {
         "type": "object",
         "properties": {
             "query": {"type": "string",
                       "description": "Natural language question. e.g. What is the job of Joe Biden?"},
             "answer": {"type": "string", "description": "Answer entity. e.g. president"}
         },
         "required": ["query", "answer"],
     },}
]


def load_wikidata_names():
    path = os.path.dirname(os.path.dirname(__file__))
    id2names = []
    for object in ["entity", "relation"]:
        vocab_file = os.path.join(path, "data/ogblwikikg2/%s.txt" % object)
        id2name = {}
        with open(vocab_file, "r") as fin:
            for line in fin:
                k, v = line.strip().split("\t")
                match = re.search("(.*) \([PQ]\d+\)", v)
                if match:
                    v = match.group(1)
                id2name[k] = v
        id2names.append(id2name)

    return id2names


def get_examples(dataset):
    examples = {}
    for h, t, r in dataset.graph.edge_list.tolist():
        h_id = dataset.entity_vocab[h]
        t_id = dataset.entity_vocab[t]
        r_id = dataset.relation_vocab[r]
        if r_id not in examples:
            examples[r_id] = (h_id, t_id)
            if len(examples) == dataset.num_relation:
                break
    return examples


def get_wikidata_id(name, type="entity"):
    url = "https://www.wikidata.org/w/api.php?%s"
    data = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "type": "item" if type == "entity" else "property"
    }
    data = urllib.parse.urlencode(data)
    url = url % data
    with urllib.request.urlopen(url) as response:
        obj = json.loads(response.read())
    if obj["search"]:
        return obj["search"][0]["id"]
    else:
        return None


def get_entity_types(e_id):
    url = "https://www.wikidata.org/w/api.php?%s"
    data = {
        "action": "wbgetclaims",
        "entity": e_id,
        "property": "P31",
        "format": "json"
    }
    data = urllib.parse.urlencode(data)
    url = url % data
    with urllib.request.urlopen(url) as response:
        obj = json.loads(response.read())
    return [x["mainsnak"]["datavalue"]["value"]["id"] for x in obj["claims"]["P31"]]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def chatgpt(messages, functions=None, temperature=0.7):
    if functions:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, functions=functions,
                                                temperature=temperature)
    else:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=temperature)
    return response["choices"][0]["message"]


def llm_function(prompt_file):
    def func(**kwargs):
        query = prompt.render(kwargs)
        query = [{"role": "user", "content": query}]
        answer = chatgpt(query, temperature=0)["content"]
        return utils.literal_eval(answer)

    prompt = open(prompt_file, "r").read()
    prompt = jinja2.Template(prompt)
    return func


named_entity_recognition = llm_function("prompt/ner.txt")
relation_extraction = llm_function("prompt/re.txt")
triplet_direction = llm_function("prompt/direction.txt")


def parse_query(query, answer=None):
    head = named_entity_recognition(input=query)
    relation = relation_extraction(input=query)
    h_id = get_wikidata_id(head, type="entity")
    r_id = get_wikidata_id(relation, type="relation")
    if not h_id:
        raise RuntimeError("Can't find entity `%s` in Wikidata" % head)
    if not r_id:
        raise RuntimeError("Can't find relation `%s` in Wikidata" % relation)
    is_same_type = []
    for e_id in examples[r_id]:
        h_types = get_entity_types(h_id)
        e_types = get_entity_types(e_id)
        is_same_type.append(bool(set(h_types).intersection(e_types)))
    if sum(is_same_type) == 1:
        is_inverse = is_same_type[1]
    else:
        is_inverse = triplet_direction(input=query, triplet=(head, relation, "X"))
    if answer:
        t_id = get_wikidata_id(answer, type="entity")
        return h_id, r_id, t_id, is_inverse
    else:
        return h_id, r_id, is_inverse


def reasoning(solver, dataset, examples, e_id2name, r_id2name, query):
    h_id, r_id, is_inverse = parse_query(query)
    if h_id not in dataset.inv_entity_vocab:
        raise RuntimeError("Can't find the entity in ogbl-wikikg2")
    if r_id not in dataset.inv_relation_vocab:
        raise RuntimeError("Can't find the relation in ogbl-wikikg2")
    h_index = dataset.inv_entity_vocab[h_id]
    r_index = dataset.inv_relation_vocab[r_id]
    h_index = torch.ones(dataset.num_entity, dtype=torch.long, device=solver.device) * h_index
    r_index = torch.ones(dataset.num_entity, dtype=torch.long, device=solver.device) * r_index
    t_index = torch.arange(dataset.num_entity, device=solver.device)
    if is_inverse:
        h_index, t_index = t_index, h_index
    sample = torch.stack([h_index, t_index, r_index], dim=-1)

    solver.model.eval()
    with torch.no_grad():
        pred = solver.model.predict(sample.unsqueeze(0))[0]
        prob, index = pred.topk(k=10)
    prob = prob.tolist()
    index = index.tolist()

    result = []
    for i, p in zip(index, prob):
        if p < 0:
            break
        h_id = dataset.entity_vocab[i]
        answer = e_id2name[h_id]
        result.append({"answer": answer, "score": p})
    return result


def explain(solver, dataset, examples, e_id2name, r_id2name, query, answer):
    h_id, r_id, t_id, is_inverse = parse_query(query, answer)
    if h_id not in dataset.inv_entity_vocab:
        raise RuntimeError("Can't find the entity in ogbl-wikikg2")
    if r_id not in dataset.inv_relation_vocab:
        raise RuntimeError("Can't find the relation in ogbl-wikikg2")
    num_relation = len(r_id2name)
    h_index = dataset.inv_entity_vocab[h_id]
    r_index = dataset.inv_relation_vocab[r_id]
    t_index = dataset.inv_entity_vocab[t_id]
    if is_inverse:
        h_index, t_index = t_index, h_index
        r_index += num_relation
    sample = torch.tensor([h_index, t_index, r_index], device=solver.device)

    solver.model.eval()
    paths, weights, num_steps = solver.model.visualize(sample.unsqueeze(0))
    paths = paths.squeeze(0).tolist()
    weights = weights.squeeze(0).tolist()
    num_steps = num_steps.squeeze(0).tolist()

    url = "https://www.wikidata.org/wiki/%s#%s"
    result = []
    for path, weight, num_step in zip(paths, weights, num_steps):
        if weight == float("-inf"):
            break
        triplets = []
        links = []
        for h, t, r in path[:num_step]:
            h_id = dataset.entity_vocab[h]
            t_id = dataset.entity_vocab[t]
            r_id = dataset.relation_vocab[r % num_relation]
            head = e_id2name[h_id]
            tail = e_id2name[t_id]
            relation = r_id2name[r_id]
            if r >= num_relation:
                relation += "^(-1)"
                links.append(url % (t_id, r_id))
            else:
                links.append(url % (h_id, r_id))
            triplets.append((head, relation, tail))
        result.append({"path": triplets, "score": weight, "reference links": links})
    return result


def chat(solver, dataset, examples, e_id2name, r_id2name):
    messages = [{"role": "system",
                 "content": "You are a helpful assistant equipped with A*Net to answer questions and explain answers. "
                            "A*Net is a path-based reasoning model trained on ogbl-wikikg2, "
                            "a subset of the Wikidata knowledge graph."}]
    while True:
        logger.warning("User:")
        query = input()
        logger.warning(query)
        query = {"role": "user", "content": query}
        messages.append(query)
        answer = chatgpt(messages, functions=functions)
        messages.append(answer)
        if "function_call" in answer:
            func = answer["function_call"]["name"]
            kwargs = json.loads(answer["function_call"]["arguments"])
            if hasattr(module, func):
                try:
                    result = getattr(module, func)(solver, dataset, examples, e_id2name, r_id2name, **kwargs)
                except RuntimeError as err:
                    result = str(err)
                result = {"role": "function", "name": func, "content": json.dumps(result)}
                messages.append(result)
                answer = chatgpt(messages)
                messages.append(answer)
        logger.warning("Bot:")
        logger.warning(answer["content"])


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] != "OGBLWikiKG2":
        raise ValueError("Chat is not implemented for %s" % cfg.dataset["class"])

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    e_id2name, r_id2name = load_wikidata_names()
    examples = get_examples(dataset)

    chat(solver, dataset, examples, e_id2name, r_id2name)
