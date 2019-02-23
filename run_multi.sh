#!/bin/bash

if [ $# -eq 0 ]
then 
    script="params/param_train.py"
else
    script=$1
fi

script="params/param_train.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

script="params/param_eval.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args


script="params/param_train1.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

script="params/param_eval1.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args


script="params/param_train2.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

script="params/param_eval2.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args


script="params/param_train3.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

script="params/param_eval3.py"
args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

#script="params/param_train4.py"
#args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

#script="params/param_eval4.py"
#args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args


#script="params/param_train5.py"
#args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

#script="params/param_eval5.py"
#args="$(python3.6 $script)"
#CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args

