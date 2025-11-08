#!/bin/bash
# self-supervised learning
python test.py \
    --w_obj 1e-5 \
    --w_slack 1.0 \
    --w_sup 0.0 \
    --save_stats \
    --filename energy_tank_ssl_1 \
    --TRAINING_EPOCHS 20 \
# hybrid learning
# python test.py \
#     --w_obj 1e-4 \
#     --w_slack 1.0 \
#     --w_sup 1e3 \
#     --save_stats \
#     --filename energy_tank_hybrid_2 \
#     --TRAINING_EPOCHS 20
# # supervised learning
# python test.py \
#     --w_obj 0.0 \
#     --w_slack 0.0 \
#     --w_sup 1.0 \
#     --save_stats \
#     --filename energy_tank_sl \
#     --TRAINING_EPOCHS 20