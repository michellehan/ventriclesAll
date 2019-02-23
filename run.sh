#!/bin/bash

if [ $# -eq 0 ]
then 
    script="params/param_train.py"
else
    script=$1
fi

args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="2,3" python3.6 vent_main.py --parameters $script $args
#CUDA_VISIBLE_DEVICES="0,1,2" python3.6 vent_crossent.py --parameters $script $args

