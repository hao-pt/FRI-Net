python eval_stru3d.py \
    --img_folder data/coco_s3d_bw \
    --occ_folder data/stru3d/occ \
    --ids_path data/stru3d \
    --input_channels 3 \
    --checkpoint /home/htp26/FRI-Net/checkpoints/frinet_s3dbw/checkpoint_2.pth \
    --phase 2 \
    --batch_size 1 \
    --num_workers 0 \
    --save_folder eval_results/s3dbw_frinet_ckpt_last/

# python eval_stru3d.py \
#     --img_folder data/stru3d/input \
#     --occ_folder data/stru3d/occ \
#     --ids_path data/stru3d \
#     --input_channels 1 \
#     --save_folder org_results \
#     --checkpoint /home/htp26/FRI-Net/checkpoints/pretrained_ckpt.pth


