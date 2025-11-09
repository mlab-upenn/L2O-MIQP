#!/bin/bash
# supervised learning
python test.py \
    --w_obj 0.0 \
    --w_slack 0.0 \
    --w_con 0.0 \
    --w_sup 1.0 \
    --save_stats \
    --filename energy_tank_sl \
    --TRAINING_EPOCHS 20
# self-supervised learning
python test.py \
    --w_obj 1e-3 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 0.0 \
    --save_stats \
    --filename energy_tank_ssl \
    --TRAINING_EPOCHS 20 \
# hybrid learning
python test.py \
    --w_obj 0.0 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 1e2 \
    --save_stats \
    --filename energy_tank_hl_1 \
    --TRAINING_EPOCHS 20    
python test.py \
    --w_obj 0.0 \
    --w_slack 0.0 \
    --w_con 1.0 \
    --w_sup 1e1 \
    --save_stats \
    --filename energy_tank_hl_2 \
    --TRAINING_EPOCHS 20       
