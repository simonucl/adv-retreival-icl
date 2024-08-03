# ICL

## RTE
### textfooler
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl textfooler > ./logs/run_icl_rte_textfooler.log 2>&1 &

### textbugger
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl textbugger > ./logs/run_icl_rte_textbugger.log 2>&1 &

### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl textbugger

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh sst2 meta-llama/Llama-2-7b-hf icl icl_attack > ./logs/run_icl_sst2_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_sst2.sh sst2 meta-llama/Llama-2-7b-hf retrieval_icl icl_attack > ./logs/run_retrieval_icl_attack_sst2_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_trec.sh trec meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_trec_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_trec.sh trec meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_retrieval_icl_trec_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_trec.sh trec meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_trec_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_mr.sh mr meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_mr_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_mr.sh mr meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_ralm_mr_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_mnli.sh mnli meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_mnli_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_mnli.sh mnli meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_ralm_mnli_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_retrieval_icl_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/attack_swap_labels_7b.sh > ./logs/run_icl_7b_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/swap_labels_ablation/transfer_llama_7b_icl.sh > ./logs/run_icl_7b_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/swap_labels_ablation/transfer_llama_7b_ricl.sh > ./logs/run_ricl_7b_rte_swap_labels.log 2>&1 &


### swap_labels fix dist
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_fix_dist.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_rte_swap_labels.log 2>&1 &

### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/icl/attack_fix_dist.sh rte meta-llama/Llama-2-7b-hf icl_attack swap_orders > ./logs/run_icl_rte_swap_orders.log 2>&1 &


################################################
## SST2
### textfooler
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh sst2 meta-llama/Llama-2-7b-hf icl textfooler > ./logs/run_icl_sst2_textfooler.log 2>&1 &

### textbugger
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh sst2 meta-llama/Llama-2-7b-hf icl textbugger > ./logs/run_icl_sst2_textbugger.log 2>&1 &
### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh sst2 meta-llama/Llama-2-7b-hf icl_attack swap_labels > ./logs/run_icl_sst2_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh sst2 meta-llama/Llama-2-7b-hf retrieval_icl icl_attack > ./logs/run_retrieval_icl_sst2_icl_attack.log 2>&1 &
### swap_labels fix dist

### swap_orders

################################################
################################################
################################################

# KNN-ICL
## RTE
### textfooler
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl textfooler > ./logs/run_knn_icl_rte_textfooler.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack_whole.sh rte meta-llama/Llama-2-7b-hf knn_icl textfooler > ./logs/run_knn_icl_whole_rte_textfooler.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 bash scripts/knn_icl/attack_whole.sh rte meta-llama/Llama-2-7b-hf icl irrelevant_sample

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack.sh rte meta-llama/Llama-2-7b-hf icl irrelevant_sample > ./logs/run_icl_rte_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl irrelevant_sample > ./logs/run_ralm_rte_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack_whole.sh rte meta-llama/Llama-2-7b-hf knn_icl textbugger > ./logs/run_knn_icl_whole_rte_textbugger.log 2>&1 &

### textbugger
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl textbugger > ./logs/run_knn_icl_rte_textbugger.log 2>&1 &

### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl_attack swap_labels > ./logs/run_knn_icl_rte_swap_labels.log 2>&1 &

### swap_labels fix dist

### irrelevant_sample
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl_attack irrelevant_sample > ./logs/run_knn_icl_rte_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl irrelevant_sample > ./logs/run_knn_icl_rte_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/attack_icl_irrelevant_1.sh sst2 meta-llama/Llama-2-7b-hf icl irrelevant_sample > ./logs/run_icl_sst2_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/attack_ralm_irrelevant_1.sh sst2 meta-llama/Llama-2-7b-hf retrieval_icl irrelevant_sample > ./logs/run_ralm_sst2_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/attack_icl_irrelevant_2.sh mr meta-llama/Llama-2-7b-hf icl irrelevant_sample > ./logs/run_icl_mr_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/attack_ralm_irrelevant_2.sh mr meta-llama/Llama-2-7b-hf retrieval_icl irrelevant_sample > ./logs/run_ralm_mr_irrelevant_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh sst2 meta-llama/Llama-2-7b-hf knn_icl irrelevant_sample > ./logs/run_knn_icl_sst2_irrelevant_sample.log 2>&1 &

### textbugger
### icl_attack
### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/knn_icl/attack_quantized_fix_dist.sh rte meta-llama/Llama-2-7b-hf knn_icl swap_labels > ./logs/run_knn_icl_rte_swap_labels.log 2>&1 &

### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl swap_orders > ./logs/run_knn_icl_rte_swap_orders.log 2>&1 &

### irrelevant_sample
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh rte meta-llama/Llama-2-7b-hf knn_icl irrelevant_sample > ./logs/run_knn_icl_rte_irrelevant_sample.log 2>&1 &

################################################
################################################
################################################


## Retrieval-ICL

### RTE
### textbugger
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl textbugger > ./logs/run_retrieval_icl_rte_textbugger.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 bash scripts/ralm/attack_test.sh rte meta-llama/Llama-2-7b-hf retrieval_icl textfooler

CUDA_VISIBLE_DEVICES=1 bash scripts/ralm/attack_test.sh rte meta-llama/Llama-2-7b-hf retrieval_icl irrelevant_sample

### textfooler
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl textfooler > ./logs/run_retrieval_icl_rte_textfooler.log 2>&1 &

### icl_attack
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl icl_attack > ./logs/run_retrieval_icl_attack_rte_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_icl_attack.sh sst2 meta-llama/Llama-2-7b-hf icl icl_attack > ./logs/run_icl_sst2_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/icl/attack_icl_attack.sh rte meta-llama/Llama-2-7b-hf icl icl_attack > ./logs/run_icl_rte_icl_attack.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_1.sh rte meta-llama/Llama-2-7b-hf retrieval_icl icl_attack > ./logs/run_retrieval_icl_attack_rte_icl_attack_1.log 2>&1 &
### swap_labels
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack_quantized_fix_dist.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_retrieval_icl_rte_swap_labels.log 2>&1 &
### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_orders > ./logs/run_retrieval_icl_rte_swap_orders.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl swap_labels > ./logs/run_retrieval_icl_rte_swap_labels.log 2>&1 &
### irrelevant_sample

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl textfooler > ./logs/run_retrieval_icl_rte_textfooler.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl textbugger > ./logs/run_retrieval_icl_rte_textbugger.log 2>&1 &

################################################

## SST2
### textfooler
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ralm/attack.sh sst2 meta-llama/Llama-2-7b-hf retrieval_icl textfooler > ./logs/run_retrieval_icl_sst2_textfooler.log 2>&1 &
### textbugger
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh sst2 meta-llama/Llama-2-7b-hf retrieval_icl textbugger > ./logs/run_retrieval_icl_sst2_textbugger.log 2>&1 &

################################################
################################################


## Retrieval-ICL-attack
### textbugger
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_ralm_attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl_attack textbugger > ./logs/run_retrieval_icl_attack_rte_textbugger.log 2>&1 &
### icl_attack
### swap_labels
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack_quantized_fix_dist.sh rte meta-llama/Llama-2-7b-hf retrieval_icl_attack swap_labels > ./logs/run_retrieval_icl_attack_rte_swap_labels.log 2>&1 &
### swap_orders
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ralm/attack.sh rte meta-llama/Llama-2-7b-hf retrieval_icl_attack swap_orders > ./logs/run_retrieval_icl_attack_rte_swap_orders.log 2>&1 &

### irrelevant_sample

nohup bash scripts/scaling_model/scale_model_70b.sh > ./logs/run_icl_70b_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/scaling_model/scale_model_icl.sh > ./logs/run_icl_rte_swap_labels.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/scaling_model/scale_model_ricl.sh > ./logs/run_ricl_rte_swap_labels.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/transferable/transfer_mistral_moe.sh > ./logs/run_transfer_mistral_moe.log 2>&1 &