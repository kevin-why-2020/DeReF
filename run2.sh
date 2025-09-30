CUDA_VISIBLE_DEVICES=5 python main.py --which_splits 5foldcv \
                                      --dataset tcga_luad \
                                      --data_root_dir /data2/share/TCGA_LUAD/LUAD_patch/features \
                                      --modal coattn \
                                      --model cmta \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_mse \
                                      --lr 0.0005 \
                                      --optimizer Adam \
                                      --scheduler None \
                                      --alpha 1.0 \
                                      --seed 7

