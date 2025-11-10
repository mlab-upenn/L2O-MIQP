#!/bin/bash
# supervised learning
# python test.py \
#     --w_obj 0.0 \
#     --w_slack 0.0 \
#     --w_con 0.0 \
#     --w_sup 1.0 \
#     --save_stats \
#     --TRAINING_EPOCHS 10 \
#     --filename robot_nav_sl
# self-supervised learning
# python test.py \
#     --w_obj 1e-5 \
#     --w_slack 1.0 \
#     --w_con 1.0 \
#     --w_sup 0.0 \
#     --save_stats \
#     --filename robot_nav_ssl \
#     --TRAINING_EPOCHS 20 \
# # hybrid learning
# python test.py \
#     --w_obj 0.0 \
#     --w_slack 0.0 \
#     --w_con 1.0 \
#     --w_sup 1e3 \
#     --save_stats \
#     --filename robot_nav_hl_1 \
#     --TRAINING_EPOCHS 10
# python test.py \
#     --w_obj 0.0 \
#     --w_slack 1.0 \
#     --w_con 1.0 \
#     --w_sup 1e3 \
#     --save_stats \
#     --filename robot_nav_hl_2 \
#     --TRAINING_EPOCHS 20
python train.py \
    --w_obj 1e0 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 1e4 \
    --save_stats \
    --filename robot_nav_hyl_3 \
    --TRAINING_EPOCHS 20    
