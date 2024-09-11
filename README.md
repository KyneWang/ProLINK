<div align="center">

# LLM as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs #

</div>

Knowledge Graph (KG) inductive reasoning, which aims to infer missing facts from new KGs that are not seen during training, has been widely adopted in various applications. One critical challenge of KG inductive reasoning is handling low-resource scenarios with scarcity in both textual and structural aspects. In this paper, we attempt to address this challenge with Large Language Models (LLMs). Particularly, we utilize the state-of-the-art LLMs to generate a graph-structural prompt to enhance the pre-trained Graph Neural Networks (GNNs), which brings us new methodological insights into the KG inductive reasoning methods, as well as high generalizability in practice. On the methodological side, we introduce a novel pretraining and prompting framework ProLINK, designed for low-resource inductive reasoning across arbitrary KGs without requiring additional training. On the practical side, we experimentally evaluate our approach on 36 low-resource KG datasets and find that ProLINK outperforms previous methods in three-shot, one-shot, and zero-shot reasoning tasks, exhibiting average performance improvements by 20{\%}, 45{\%}, and 147{\%}, respectively. Furthermore, ProLINK demonstrates strong robustness for various LLM promptings as well as full-shot scenarios.

## Requirements ##

- pytorch  2.1.0+cu121
- torch_geometric  2.4.0
- torch_scatter 2.1.2+pt21cu121
- transformers 4.41.1

## For transductive reasoning

We can use the following commands to conduct the inference process of ProLINK based on pre-grained ultra_3g model on the InGram datasets under the inductive setting.

```
    python script/run_fewshot.py -c config/inductive/inference.yaml --gpus [0] --ckpt ckpts/ultra_3g.pth -d FBIngram:100,WKIngram:100,NLIngram:100
```


## Citation ##

If you find this codebase useful in your research, please cite the original papers.

The main ULTRA paper:

```bibtex
@inproceedings{wang-etal-2024-llm,
    title = "{LLM} as Prompter: Low-resource Inductive Reasoning on Arbitrary Knowledge Graphs",
    author = "Wang, Kai  and
      Xu, Yuwei  and
      Wu, Zhiyong  and
      Luo, Siqiang",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.224",
    pages = "3742--3759",
}
```

## Acknowledgement ##

We refer to the code of [ULTRA](https://github.com/DeepGraphLearning/ULTRA) and the datasets of [InGram](https://github.com/bdi-lab/InGram). Thanks for their contributions.

> Note: The basic Ultra model code are copied from ULTRA's repository.
