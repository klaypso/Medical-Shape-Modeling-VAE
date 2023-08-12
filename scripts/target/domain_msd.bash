python main_target.py domain_msd \
    -G $1 \
    --method domain_adaptation \
    --load_prefix seg_nih \
    --load_prefix_vae vae_nih \
    --train_list MSD_train \
    --val_list MSD_val \
    --data_root <Your_MSD_data_path> \
    --val_data_root <Your_MSD_data_path> \
    --data_path data/Multi_all.json \
    