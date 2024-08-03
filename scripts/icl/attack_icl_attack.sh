DATASET=$1
MODEL=$2
MODEL_TYPE=$3 # [icl | knn_icl | retrieval_icl | retrieval_icl_attack ]
ATTACK=$4 # [textfooler | textbugger | icl_attack | swap_labels | swap_orders | irrelevant_sample]

TEMPLATE_FILE=configs/templates_${DATASET}.yaml
VERBALIZER_FILE=configs/verbalizer_${DATASET}.yaml
SHOTS=(4 16)
SEEDS=(1 13 42)

if [[ $ATTACK == "textfooler" ]] || [[ $ATTACK == "textbugger" ]] || [[ $ATTACK == "icl_attack" ]]; then
    ATTACK_PRECENT=0.15
else
    ATTACK_PRECENT=0.5
fi

# source ~/.bashrc
# echo $PWD
# conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/software/spack/spack-rhel8-20210927/opt/spack/linux-centos8-zen2/gcc-9.4.0/cuda-11.4.0-3hnxhjt2jt4ruy75w2q4mnvkw7dty72l

for SHOT in ${SHOTS[@]};
do
    for SEED in ${SEEDS[@]};
    do 
        BATCH_SIZE=$((64 / SHOT))

        echo $SEED+${SHOT}+${MODEL}+"mvp"
        MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
        MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

        DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

        mkdir -p ${MODELPATH}
        echo ${MODELPATH}

        nohup python3 main.py \
            --mode attack \
            --attack_name ${ATTACK} \
            --num_examples 100 \
            --dataset ${DATASET} \
            --query_budget -1 \
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
                    --query_budget -1 \
                    --batch_size ${BATCH_SIZE} \
                    --model_type ${MODEL_TYPE} \
                    --model ${MODEL} \
                    --verbalizer_file ${VERBALIZER_FILE} \
                    --template_file ${TEMPLATE_FILE} \
                    --seed $SEED \
                    --shot ${SHOT} \
                    --max_percent_words ${ATTACK_PRECENT} \
                    --model_dir ${MODELPATH} \
                    --fix_dist \
                    > ${MODELPATH}/logs_${ATTACK}_fix_dist.txt
            fi
    done
done
