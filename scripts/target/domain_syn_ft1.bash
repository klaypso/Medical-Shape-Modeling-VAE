
python main_target.py domain_syn_ft1 \
    -G $1 \
    --method domain_adaptation \
    --load_prefix seg_nih \
    --load_prefix_vae vae_nih \
    --train_list SYN_train \
    --val_list SYN_val \
    --data_root <Your_SYN_data_path> \
    --val_data_root <Your_SYN_data_path> \
    --data_path data/Multi_all.json \
    --pan_index 11 \
    --lambda_vae 0.1 \
    --val_finetune 1 \
    --eval_epoch 1 \
    --save_epoch 100 \
    --max_epoch 50