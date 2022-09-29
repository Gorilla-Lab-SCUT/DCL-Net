python -m ipdb tools/train_YCBV_stage1.py \
    --path_data datasets/YCB_Video_Dataset \
    --model DCL_Net \
    --config configs/config_YCBV_bs32.yaml \
    --exp_id 0 \
    --gpus 0