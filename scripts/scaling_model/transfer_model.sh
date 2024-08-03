DATASET=$1
MODEL=$2
MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
TOTAL_BATCH=$4

# replace '/' with '_'
MODEL_NAME=${MODEL//\//_}

# wait until this command is finished then next 
nohup bash scripts/icl/attack_all_model.sh $DATASET $MODEL icl $TOTAL_BATCH > ./logs/run_icl_${DATASET}_${MODEL_NAME}_all_model.log 2>&1

nohup bash scripts/ralm/attack_all_model.sh $DATASET $MODEL retrieval_icl $TOTAL_BATCH > ./logs/run_ralm_${DATASET}_${MODEL_NAME}_all_model.log 2>&1
