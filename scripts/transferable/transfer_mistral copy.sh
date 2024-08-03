MODELS=(mistralai/Mistral-7B-Instruct-v0.2 mistralai/Mixtral-8x7B-v0.1)
SEEDS=(1 13 42)
ATTACKS=(swap_labels swap_labels_fix_dist)
DATASETS=(rte)

RETRIEVERS=(bm25 sbert instructor)

BASE_MODEL=mistralai/Mistral-7B-v0.1

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
                if [[ $ATTACK == "swap_labels_fix_dist" ]]; then
                    echo csv_path: checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/icl_attack-seed-${SEED}-shot-8/swap_labels_fix_dist_log.csv
                    python3 src/transfer_attack.py \
                        --model $MODEL \
                        --csv_path checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/icl_attack-seed-${SEED}-shot-8/swap_labels_fix_dist_log.csv \
                        --attack $ATTACK \
                        --precision $PRECISION \
                        --demonstration_path data/icl/${DATASET}-icl-seed-${SEED}-shot-8.pkl
                else
                    echo csv_path: checkpoints/${DATASET}/${BASE_MODEL}/${ATTACK}/icl_attack-seed-${SEED}-shot-8/${ATTACK}_log.csv
                    python3 src/transfer_attack.py \
                        --model $MODEL \
                        --csv_path checkpoints/${DATASET}/${BASE_MODEL}/${ATTACK}/icl_attack-seed-${SEED}-shot-8/${ATTACK}_log.csv \
                        --attack $ATTACK \
                        --demonstration_path data/icl/${DATASET}-icl-seed-${SEED}-shot-8.pkl \
                        --precision $PRECISION
                fi
            done

            # for RETRIEVER in ${RETRIEVERS[@]};
            # do
            #     echo model: $MODEL
            #     echo csv_path: checkpoints/${DATASET}/${BASE_MODEL}/${ATTACK}/icl-seed-${SEED}-shot-8/${ATTACK}_log.csv
            #     if [[ $ATTACK == "swap_labels_fix_dist" ]]; then
            #         python3 src/transfer_attack.py \
            #             --model $MODEL \
            #             --csv_path checkpoints/${DATASET}/${BASE_MODEL}/swap_labels/retrieval_icl-seed-1-shot-8_${RETRIEVER}_fix_dist/swap_labels_log.csv \
            #             --attack $ATTACK \
            #             --precision $PRECISION \
            #             --demonstration_path data/ralm/${DATASET}_${RETRIEVER}.pkl \
            #             --dataset $DATASET
            #     else
            #         python3 src/transfer_attack.py \
            #             --model $MODEL \
            #             --csv_path checkpoints/${DATASET}/${BASE_MODEL}/${ATTACK}/retrieval_icl-seed-1-shot-8_${RETRIEVER}/${ATTACK}_log.csv \
            #             --attack $ATTACK \
            #             --precision $PRECISION \
            #             --demonstration_path data/ralm/${DATASET}_${RETRIEVER}.pkl \
            #             --dataset $DATASET
            #     fi
            # done
        done
    done
done