MODELS=(meta-llama/Llama-2-13b-hf mistralai/Mistral-7B-v0.1 lmsys/vicuna-7b-v1.5 mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mixtral-8x7B-v0.1)
SEEDS=(1 13 42)
RETRIEVERS=(bm25 sbert instructor)
ATTACKS=(swap_labels swap_labels_fix_dist)
DATASETS=(sst2)

BASE_MODEL=meta-llama/Llama-2-7b-hf

for MODEL in ${MODELS[@]};
do
    if [[ $MODEL == "meta-llama/Llama-2-70b-hf" ]] || [[ $MODEL == "mistralai/Mixtral-8x7B-v0.1" ]]; then
        PRECISION=int4
    else
        PRECISION=bf16
    fi
    for DATASET in ${DATASETS[@]};
    do
        for ATTACK in ${ATTACKS[@]};
        do
            for SEED in ${SEEDS[@]};
            do
                echo model: $MODEL
                if [[ $ATTACK == "swap_labels_fix_dist" ]]; then
                    python3 src/transfer_attack.py \
                        --model $MODEL \
                        --csv_path checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/icl_attack-seed-${SEED}-shot-8/swap_labels_fix_dist_log.csv \
                        --attack $ATTACK \
                        --precision $PRECISION \
                        --dataset $DATASET
                else
                    echo csv_path: checkpoints/${DATASET}/${BASE_MODEL}/${ATTACK}/icl-seed-${SEED}-shot-8/${ATTACK}_log.csv
                    python3 src/transfer_attack.py \
                        --model $MODEL \
                        --csv_path checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/icl_attack-seed-${SEED}-shot-8/swap_labels_log.csv \
                        --attack $ATTACK \
                        --precision $PRECISION \
                        --dataset $DATASET
                fi
            done
        done
    done
done