SEEDS=(1 13 42)
ATTACKS=(textfooler textbugger bert_attack)
DATASETS=(rte)

RETRIEVERS=(bm25 sbert instructor)

BASE_MODEL=meta-llama/Llama-2-7b-hf


for DATASET in ${DATASETS[@]};
do
    for ATTACK in ${ATTACKS[@]};
    do
        for SEED in ${SEEDS[@]};
        do
            CUDA_VISIBLE_DEVICES=0 python3 src/transfer_attack.py \
                --model $BASE_MODEL \
                --csv_path checkpoints/rte/${BASE_MODEL}/${ATTACK}/icl-seed-${SEED}-shot-8/${ATTACK}_log.csv \
                --attack $ATTACK \
                --demonstration_path data/icl/${DATASET}-icl-seed-${SEED}-shot-8.pkl \
                --add_icl_examples_only
        done
    done
done

for DATASET in ${DATASETS[@]};
do
    for ATTACK in ${ATTACKS[@]};
    do
        for RETRIEVER in ${RETRIEVERS[@]};
        do
            CUDA_VISIBLE_DEVICES=0 python3 src/transfer_attack.py \
                --model $BASE_MODEL \
                --csv_path checkpoints/rte/${BASE_MODEL}/${ATTACK}/retrieval_icl-seed-1-shot-8_${RETRIEVER}/${ATTACK}_log.csv \
                --attack swap_labels \
                --demonstration_path data/ralm/${DATASET}_${RETRIEVER}.pkl \
                --add_icl_examples_only
        done
    done
done