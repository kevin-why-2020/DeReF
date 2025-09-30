CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv \
                                      --dataset tcga_blca \
                                      --data_root_dir /data3/share/TCGA/BLCA_feature \
                                      --modal coattn \
                                      --model DeReF \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_mse \
                                      --lr 0.0005 \
                                      --optimizer Adam \
                                      --scheduler None \
                                      --alpha 1.0 \
                                      --seed 7


