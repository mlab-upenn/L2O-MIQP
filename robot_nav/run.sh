#!/bin/bash

# supervised learning
python test.py \
    --w_obj 0.0 \
    --w_slack 0.0 \
    --w_con 0.0 \
    --w_sup 1.0 \
    --save_stats \
    --filename robot_nav_sl  


# self-supervised learning
python test.py \
    --w_obj 1e-5 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 0.0 \
    --save_stats \
    --filename robot_nav_ssl

# hybrid learning
python test.py \
    --w_obj 0.0 \
    --w_slack 1.0 \
    --w_con 1.0 \
    --w_sup 1e3 \
    --save_stats \
    --filename robot_nav_hybrid