# adv-retreival-icl
Codebase for the paper [Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models](https://arxiv.org/abs/2405.15984)

![Main Figure](./figure/model-arch-4.png)


### 0. Installation
```bash
conda create -n adv-retrieval-icl python=3.8
pip install -r requirements.txt

pip install -e git+https://github.com/simonucl/TextAttack.git#egg=TextAttack
```
Following the above instructions with the installed egg package for running experiments

### 1. Running Main Experiments
```bash
MODEL=meta-llama/Llama-2-7b-hf
DATASET=rte # sst2|rte|mnli|cr|mr|trec
ATTACK=textfooler

# Vanilla ICL
bash scripts/icl/attack.sh $DATASET $MODEL icl $ATTACK

# kNN-ICL
bash scripts/knn_icl/attack.sh $DATASET $MODEL knn_icl $ATTACK

# Retrieval ICL
bash scripts/ralm/attack.sh $DATASET $MODEL retrieval_icl $ATTACK
```

### 2. Running Ablation Experiments
```bash
# Section 4.3: Ablation Study
bash scripts/scaling_model/scale_model_icl.sh
bash scripts/scaling_model/scale_model_ricl.sh

# Section 4.4: Transferable attack
bash scripts/transferable/transfer_llama_7b.sh
bash scripts/transferable/transfer_llama_70b.sh

bash scripts/transferable/transfer_mistral.sh
bash scripts/transferable/transfer_mistral_moe.sh
```

### 3. Citation
```bib
@misc{yu2024evaluatingadversarialrobustnessretrievalbased,
      title={Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models}, 
      author={Simon Chi Lok Yu and Jie He and Pasquale Minervini and Jeff Z. Pan},
      year={2024},
      eprint={2405.15984},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.15984}, 
}
```