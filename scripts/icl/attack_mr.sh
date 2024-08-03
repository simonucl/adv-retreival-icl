DATASET=$1
MODEL=$2
MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
ATTACK=$4 # [textfooler | textbugger | icl_attack | swap_labels | swap_orders | irrelevant_sample]

TEMPLATE_FILE=configs/templates_${DATASET}.yaml
VERBALIZER_FILE=configs/verbalizer_${DATASET}.yaml
if [[ $DATASET == "rte" ]]; then
    TOTAL_BATCH=32
elif [[ $DATASET == "mnli" ]]; then
    TOTAL_BATCH=32
else
    TOTAL_BATCH=64
fi

ATTACKS=(swap_labels bert_attack icl_attack)

SEEDS=(1 13 42)

SHOTS=(4 2 16)

for ATTACK in ${ATTACKS[@]};
do
    if [[ $ATTACK == "swap_labels" ]]; then
        SHOTS=(4 2 16)
    else
        SHOTS=(8 4 2 16)
    fi

    if [[ $ATTACK == "bert_attack" ]]; then
        MODEL_TYPE="icl"
    else
        MODEL_TYPE="icl_attack"
    fi

    if [[ $ATTACK == "textfooler" ]] || [[ $ATTACK == "textbugger" ]] || [[ $ATTACK == "icl_attack" ]] || [[ $ATTACK == "bert_attack" ]]; then
        ATTACK_PRECENT=0.15
    else
        if [[ $DATASET == "sst2" ]] || [[ $DATASET == "rte" ]] || [[ $DATASET == "mr" ]] || [[ $DATASET == "cr" ]]; then
            ATTACK_PRECENT=0.5
        elif [[ $DATASET == "mnli" ]]; then
            ATTACK_PRECENT=0.33
        else
            ATTACK_PRECENT=0.2
        fi
    fi

    if [[ $ATTACK == "swap_labels" ]]; then
        QUERY_BUDGET=250
    else
        QUERY_BUDGET=-1
    fi

    for SHOT in ${SHOTS[@]};
    do
        for SEED in ${SEEDS[@]};
        do 
            BATCH_SIZE=$((TOTAL_BATCH / SHOT))
            if [[ $SHOT -eq 2 ]]; then
                BATCH_SIZE=$((BATCH_SIZE / 2))
            fi
            
            echo $SEED+${SHOT}+${MODEL}+"mvp"
            MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
            MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

            DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

            mkdir -p ${MODELPATH}
            echo ${MODELPATH}

            nohup python3 main.py \
                --mode attack \
                --attack_name ${ATTACK} \
                --num_examples 1000 \
                --dataset ${DATASET} \
                --query_budget ${QUERY_BUDGET} \
                --batch_size ${BATCH_SIZE} \
                --model_type ${MODEL_TYPE} \
                --model ${MODEL} \
                --verbalizer_file ${VERBALIZER_FILE} \
                --template_file ${TEMPLATE_FILE} \
                --seed $SEED \
                --shot ${SHOT} \
                --max_percent_words ${ATTACK_PRECENT} \
                --model_dir ${MODELPATH} \
                    > ${MODELPATH}/logs_${ATTACK}.txt

            if [[ $ATTACK == "swap_labels" ]]; then
                    nohup python3 main.py \
                        --mode attack \
                        --attack_name ${ATTACK} \
                        --num_examples 1000 \
                        --dataset ${DATASET} \
                        --query_budget ${QUERY_BUDGET} \
                        --batch_size ${BATCH_SIZE} \
                        --model_type ${MODEL_TYPE} \
                        --model ${MODEL} \
                        --verbalizer_file ${VERBALIZER_FILE} \
                        --template_file ${TEMPLATE_FILE} \
                        --seed $SEED \
                        --shot ${SHOT} \
                        --max_percent_words 0.5 \
                        --model_dir ${MODELPATH} \
                        --fix_dist \
                        > ${MODELPATH}/logs_${ATTACK}_fix_dist.txt
                fi
        done
    done
done