# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/knn_icl/attack.sh mnli meta-llama/Llama-2-7b-hf knn_icl_attack swap_labels > ./logs/run_knn_icl_mnli_swap_labels.log 2>&1 &
# wait
export CUDA_VISIBLE_DEVICES=0 

for i in textfooler textbugger icl_attack swap_labels swap_orders irrelevant_sample
do
    if [ "$i" == "textfooler" ] || [ "$i" == "textbugger" ]
    then
        nohup bash scripts/knn_icl/attack.sh mnli meta-llama/Llama-2-7b-hf knn_icl "$i" > ./logs/run_knn_icl_mnli_"$i".log 2>&1 &
        wait
    else
        nohup bash scripts/knn_icl/attack.sh mnli meta-llama/Llama-2-7b-hf knn_icl_attack "$i" > ./logs/run_knn_icl_mnli_"$i".log 2>&1 &
        wait
    fi
done
