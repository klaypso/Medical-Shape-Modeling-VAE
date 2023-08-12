
python main_source.py vae_nih \
    -G $1 \
    --method vae_train \
    --train_list NIH_train \
    --val_list NIH_val \
    --data_root <Your_data_path> \
    --val_data_root <Your_data_path> \
    --data_path data/Multi_all.json \
    --eval_epoch 20 \
    --save_epoch 800 \
    --max_epoch 4800 \