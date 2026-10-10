#!/bin/bash
envs=("vascular" "capillary_test" "aml_final_video" "4squares" "racetrack" "maze_3_test")

for env in "${envs[@]}"
do
    echo "----------------------------------------------------"
    echo "Processing Environment: $env"
    echo "----------------------------------------------------"
    python scripts/play_dreamer_single.py "$env"
done
