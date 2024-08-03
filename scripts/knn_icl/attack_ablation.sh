BATCH_SIZE=$1
DATASET=$2
MODEL=$3
MODEL_TYPE=$4
TEMPLATE_FILE=${5}
VERBALIZER_FILE=${6}
ATTACK=${7}
ADV=${8}


# source ~/.bashrc
# echo $PWD
# conda activate /home/co-huan1/rds/rds-qakg-2iBGk7DbOVc/jie/conda/multi

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/software/spack/spack-rhel8-20210927/opt/spack/linux-centos8-zen2/gcc-9.4.0/cuda-11.4.0-3hnxhjt2jt4ruy75w2q4mnvkw7dty72l

for SEED in 42;
do
    for SHOT in 32;
    do 
        BETA=0.8
        echo $SEED+${SHOT}+${MODEL}+"mvp"

        MODEL_ID=${MODEL_TYPE}-seed-${SEED}-shot-${SHOT}
        
        ATTACK=textfooler
        MODELPATH=./checkpoints/${DATASET}/${MODEL}/${ATTACK}/${MODEL_ID}

        DATASET_PATH=./data/${DATASET}/${SHOT}-$SEED

        mkdir -p ${MODELPATH}
        echo ${MODELPATH}

        if [[ $SHOT -eq 4 ]]; then
            Ms=(1 2)
            Ks=(4)
        elif [[ $SHOT -eq 8 ]]; then
            Ms=(1 2)
            Ks=(4 8)
        elif [[ $SHOT -eq 16 ]]; then
            Ms=(1 2 4)
            Ks=(4 8 16)
        elif [[ $SHOT -eq 32 ]]; then
            Ms=(1 2 4)
            Ks=(4 8 16)
        fi
        
        for M in ${Ms[@]};
        do
            for KNN in ${Ks[@]};
            do
                ATTACK=textfooler
                mkdir -p ${MODELPATH}/example-k-${K}-m-${M}
                echo ${MODELPATH}/example-k-${K}-m-${M}+${ATTACK}
                # MODEL_TYPE=knn_icl
                nohup python3 main.py --mode attack \
                                            --attack_name ${ATTACK} --num_examples 1000 --dataset ${DATASET} \
                                            --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                            --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                            --seed $SEED --shot ${SHOT} \
                                            --adv_augment $ADV --knn_k $KNN --examples_per_label ${M} --beta ${BETA} > ${MODELPATH}/example-k-${K}-m-${M}/logs_${ATTACK}.txt
                ATTACK=textbugger
                echo ${MODELPATH}/example-k-${K}-m-${M}+${ATTACK}
                nohup python3 main.py --mode attack \
                                            --attack_name ${ATTACK} --num_examples 1000 --dataset ${DATASET} \
                                            --query_budget -1 --batch_size ${BATCH_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} \
                                            --verbalizer_file ${VERBALIZER_FILE} --template_file ${TEMPLATE_FILE} \
                                            --seed $SEED --shot ${SHOT} \
                                            --adv_augment $ADV --knn_k $KNN --examples_per_label ${M} --beta ${BETA} > ${MODELPATH}/example-k-${K}-m-${M}/logs_${ATTACK}.txt
            done
        done

    done
done
