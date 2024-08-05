# DATASET=$1
# MODEL=$2
# MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
# TOTAL_BATCH=$4

# wait until this command is finished then next 
DATASET=rte
# MODELS=(meta-llama/Llama-2-13b-hf mistralai/Mistral-7B-v0.1 lmsys/vicuna-7b-v1.5 mistralai/Mistral-7B-Instruct-v0.1 google/gemma-7b mistralai/Mixtral-8x7B-v0.1 mistralai/Mixtral-8x7B-Instruct-v0.1)
TOTAL_BATCH=8
MODEL=meta-llama/Llama-2-70b-hf

# wait until this command is finished then next 
bash scripts/icl/attack_all_model.sh $DATASET $MODEL icl $TOTAL_BATCH > ./logs/run_icl_${DATASET}_${MODEL_NAME}_all_model.log 2>&1
bash scripts/ralm/attack_all_model.sh $DATASET $MODEL retrieval_icl $TOTAL_BATCH > ./logs/run_ralm_${DATASET}_${MODEL_NAME}_all_model.log 2>&1