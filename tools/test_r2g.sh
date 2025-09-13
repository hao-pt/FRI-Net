# DATA=data/R2G_hr_dataset_processed_v1
# python eval_stru3d.py \
#     --img_folder ${DATA} \
#     --occ_folder ${DATA}/occ \
#     --ids_path null \
#     --input_channels 3 \
#     --checkpoint checkpoints/frinet_r2g/checkpoint_2.pth \
#     --phase 2 \
#     --batch_size 1 \
#     --num_workers 0 \
#     --save_folder eval_results/r2g_frinet_ckpt_last/

DATA=data/coco_cubicasa5k_nowalls_v4-1_refined
python eval_stru3d.py \
    --img_folder ${DATA} \
    --occ_folder ${DATA}/occ \
    --ids_path null \
    --input_channels 3 \
    --checkpoint checkpoints/frinet_r2g/checkpoint_2.pth \
    --phase 2 \
    --batch_size 4 \
    --num_workers 0 \
    --save_folder eval_results/r2g-cc5k_frinet_ckpt_last/







