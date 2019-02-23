#!/bin/bash

if [ $# -eq 0 ]
then 
    script="params/param_train.py"
else
    script=$1
fi

script="params/evals/param_eval_01.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_02.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_03.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_04.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_05.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_06.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_07.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_08.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

script="params/evals/param_eval_09.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args

