python main_target.py domain_syn_dh_ft1 \
    -G $1 \
    --method domain_adaptation \
    --load_prefix seg_nih \
    --load_prefix_vae vae_nih \
    --train_list SYN_train \
    --val_list SYN_val \
    --data_root <Your_SYN_data_path> \
    --val_data_r