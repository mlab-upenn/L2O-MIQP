#!/bin/bash

# supervised learning
# python test.py \
#     --w_obj 0.0 \
#     --w_slack 0.0 \
#     --w_con 0.0 \
#     --w_sup 1.0 \
#     --save_stats \
#     ----TRAINING_EPOCHS 10 \
#     --filename robot_nav_sl.pt
# self-supervised learning
python test.py \
    --w_obj 1e-4 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 0.0 \
    --save_stats \
    --filename robot_nav_ssl.pt \
    ----TRAINING_EPOCHS 10
# hybrid learning
python test.py \
    --w_obj 0.0 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 1e3 \
    --save_stats \
    --filename robot_nav_hybrid_1.pt \
    ----TRAINING_EPOCHS 10
python test.py \
    --w_obj 0.0 \
    --w_slack 0.0 \
    --w_con 1.0 \
    --w_sup 1e3 \
    --save_stats \
    --filename robot_nav_hybrid_2.pt \
    --TRAINING_EPOCHS 10 
