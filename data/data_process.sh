data_path=.
models=(bert-base-uncased bert-large-uncased roberta-base roberta-large)
# models=(roberta-base)
# for params in 'sst-2 2' 'ag_news 2' 'imdb 2' 'snli 3' 'yelp_polarity 2'
for params in 'sst-2 2'
do 
    set -- $params
    task=$1
    labels=$2
    # tindex=$3
    echo $params

    mkdir -p $data_path/$task/basic
    # mkdir -p $data_path/$task/prompt
    # mkdir -p $data_path/$task/prompt_with_demon
    
    # for seed in 100 13 42 87 21
    for seed in 13
    do
        for shot in 16 32 64
            do 
                for model in ${models[@]}
                    do
                    echo $seed
                    echo "prompt mode"
                    python3 data_process.py --train $data_path/$task/$shot-$seed/train.json \
                    --valid $data_path/$task/$shot-$seed/val.json \
                    --test $data_path/$task/$shot-$seed/test.json \
                    --num-label $labels \
                    --output $data_path/$task/basic/$model/$shot-$seed \
                    --task $task --seed 2017 --mode 2 --tindex 8 \
                    --num-demon-train 1 --num-demon-test 1 \
                    --model $model
                    done
            done 
    done
done 

