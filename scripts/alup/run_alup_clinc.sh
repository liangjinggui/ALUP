
#!/usr/bin bash

method="alup"
prefix="outs"
pretrain_suffix="alup_pretrain"
suffix="alup"

for dataset in "clinc"
do
    for known_cls_ratio in 0.25 0.5 0.75
    do
        for seed in 0 1 2 3 4
        do 
            python run.py \
            --dataset $dataset \
            --method ${method} \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --config_file_name "./methods/${method}/configs/config_${dataset}_al_finetune.yaml" \
            --save_results \
            --log_dir "./${prefix}/${method}/logs/${dataset}_${suffix}" \
            --output_dir "./${prefix}/${method}" \
            --dataset_dir "datasets_${known_cls_ratio}/data_${known_cls_ratio}_${dataset}_${seed}.pkl" \
            --model_file_name "model_${known_cls_ratio}_${dataset}_${seed}_${suffix}.pt" \
            --pretrained_nidmodel_file_name "model_${known_cls_ratio}_${dataset}_${seed}_${pretrain_suffix}_epoch_best.pt" \
            --result_dir "./${prefix}/${method}/results_${dataset}_${suffix}_kcr${known_cls_ratio}" \
            --results_file_name "results_${known_cls_ratio}_${dataset}_${seed}_${suffix}.csv"  \
            --cl_loss_weight 1.0 \
            --semi_cl_loss_weight 1.0
        done
    done
done


