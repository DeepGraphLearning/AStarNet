# A\*Net: A\* Networks #

This is the official codebase of the paper

[A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs][paper]

[Zhaocheng Zhu](https://kiddozhu.github.io)\*,
[Xinyu Yuan](https://github.com/KatarinaYuan)\*,
[Mikhail Galkin](https://migalkin.github.io),
[Sophie Xhonneux](https://github.com/lpxhonneux),
[Ming Zhang](http://net.pku.edu.cn/dlib/mzhang/),
[Maxime Gazeau](https://scholar.google.com/citations?user=LfmqBJsAAAAJ),
[Jian Tang](https://jian-tang.com)

[paper]: https://arxiv.org/pdf/2206.04798.pdf

## Overview ##

A\*Net is a scalable path-based method for knowledge graph reasoning. Inspired by
the classical A\* algorithm, A\*Net learns a neural priority function to select
important nodes and edges at each iteration, which significantly reduces time and
memory footprint for both training and inference.

A\*Net is the first path-based method that scales to ogbl-wikikg2 (2.5M entities,
16M triplets). It also enjoys the advantages of path-based methods such as
inductive capacity and interpretability.

https://github.com/DeepGraphLearning/AStarNet/assets/17213634/b521113e-1360-4082-af65-e2579bf01b29

This codebase contains implementation for A\*Net and its predecessor [NBFNet].

[NBFNet]: https://github.com/DeepGraphLearning/NBFNet

## Installation ##

The dependencies can be installed via either conda or pip. A\*Net is compatible
with 3.7 <= Python <= 3.10 and PyTorch >= 1.13.0.

### From Conda ###

```bash
conda install pytorch cudatoolkit torchdrug pytorch-sparse -c pytorch -c pyg -c milagraph
conda install ogb easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torch torchdrug torch-sparse
pip install ogb easydict pyyaml
```

## Usage ##

To run A\*Net, use the following command. The argument `-c` specifies the experiment
configuration file, which includes the dataset, model architecture, and
hyperparameters. You can find all configuration files in `config/.../*.yaml`.
All the datasets will be automatically downloaded in the code.

```bash
python script/run.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0]
```

For each experiment, you can specify the number of GPU via the argument `--gpus`.
You may use `--gpus null` to run A\*Net on a CPU, though it would be very slow.
To run A\*Net with multiple GPUs, launch the experiment with `torchrun`

```bash
torchrun --nproc_per_node=4 script/run.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0,1,2,3]
```

For the inductive setting, there are 4 different splits for each dataset. You need
to additionally specify the split version with `--version v1`.

## Visualization ##

A\*Net supports visualization of important paths for its predictions. With a trained
model, you can visualize the important paths with the following line. Please replace
the checkpoint with your own path.

```bash
python script/visualize.py -c config/knowledge_graph/fb15k237_astarnet_visualize.yaml --checkpoint /path/to/astarnet/experiment/model_epoch_20.pth
```

## Parameterize with your favourite GNNs ##

A\*Net is designed to be general frameworks for knowledge graph reasoning. This
means you can parameterize it with a broad range of message-passing GNNs. To do so,
just implement a convolution layer in `reasoning/layer.py` and register it with
`@R.register`. The GNN layer is expected to have the following member functions

```python
def message(self, graph, input):
    ...
    return message

def aggregate(self, graph, message):
    ...
    return update

def combine(self, input, update):
    ...
    return output
```

where the arguments and the return values are
- `graph` ([data.PackedGraph]): a batch of subgraphs selected by A*Net, with
  `graph.query` being the query embeddings of shape `(batch_size, input_dim)`.
- `input` (Tensor): node representations of shape `(graph.num_node, input_dim)`.
- `message` (Tensor): messages of shape `(graph.num_edge, input_dim)`.
- `update` (Tensor): aggregated messages of shape `(graph.num_node, *)`.
- `output` (Tensor): output representations of shape `(graph.num_node, output_dim)`.

To support the neural priority function in A\*Net, we need to additionally provide
an interface for computing messages

```python
def compute_message(self, node_input, edge_input):
   ...
   return msg_output
```

You may refer to the following tutorials of TorchDrug
- [Graph Data Structures](https://torchdrug.ai/docs/notes/graph.html)
- [Graph Neural Network Layers](https://torchdrug.ai/docs/notes/layer.html)

[data.PackedGraph]: https://torchdrug.ai/docs/api/data.html#packedgraph

## Frequently Asked Questions ##

1. **The code is stuck at the beginning of epoch 0.**

   This is probably because the JIT cache is broken.
   Try `rm -r ~/.cache/torch_extensions/*` and run the code again.

2. **The code is stuck when downloading dataset files.**

   This is probably because your machine is not connected to the Internet.
   You can manually download datasets with the following lines
   ```python
   from torchdrug import datasets
   from reasoning import dataset
   fb_transductive = datasets.FB15k237("~/datasets/knowledge_graphs")
   fb_inductive = dataset.FB15k237Inductive("~/datasets/knowledge_graphs")
   ```

## Citation ##

If you find this project useful, please consider citing the following paper

```bibtex
@article{zhu2022scalable,
  title={A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs},
  author={Zhu, Zhaocheng and Yuan, Xinyu and Galkin, Mikhail and Xhonneux, Sophie and Zhang, Ming and Gazeau, Maxime and Tang, Jian},
  journal={arXiv preprint arXiv:2206.04798},
  year={2022}
}
```
