#!/bin/bash
# R7 EP-TTA Research Experiment Launcher
# Purpose:
# GPU0: SSL-AASIST target evaluation preparation/run
# GPU1: Published TTA baseline preparation placeholder
# GPU2: Task-aware EP development placeholder
# GPU3: Result collection/report
#
# Run from TTA repository root.

set -e

ROOT=/media/dell/data/fakeAudioDection/TTA

cd ${ROOT}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

mkdir -p logs
mkdir -p outputs_v2/r7


echo "======================================"
echo "EP-TTA R7 experiment start"
date
echo "======================================"


########################################
# GPU0
# SSL-AASIST target evaluation
########################################

(
echo "[GPU0] SSL-AASIST evaluation start"
echo "[GPU0] SSL frozen/source resources already exist"

python -m eptta.cli run-tta \
 --config configs/local_v2/ssl_aasist_control_main/ep_tta.yaml

# Replace config below with generated SSL target configs
# after checking SSL frozen bundle path.
#
# Example:
# python -m eptta.cli run-tta \
#  --config configs/local_v2/ssl_aasist_control_main/ep_tta.yaml

echo "[GPU0] SSL preparation finished"

) > logs/r7_gpu0_ssl.log 2>&1 &



########################################
# GPU1
# TTA baseline development
########################################

(
echo "[GPU1] Baseline preparation"

# Current repository status:
# Tent/SAR/MEMO/EATA/T2A are NOT_RUN.
# Add implementations/configs before enabling.

echo "Baseline branch waiting for implementation"

) > logs/r7_gpu1_baseline.log 2>&1 &



########################################
# GPU2
# New task-aware EP development
########################################

(
echo "[GPU2] Task-aware EP development"

# Development branch:
# Add new objective/config here.
# Keep old EP results unchanged.

echo "Task-aware EP branch waiting for implementation"

) > logs/r7_gpu2_new_method.log 2>&1 &



########################################
# GPU3
# Existing result summary
########################################

(
echo "[GPU3] Generate existing report"

python -m eptta.cli report \
 --runs outputs_v2/aasist/target-runs \
 --out outputs_v2/r7/aasist_report.csv

echo "Report finished"

) > logs/r7_gpu3_report.log 2>&1 &



wait


echo "======================================"
echo "R7 experiment launcher finished"
date
echo "======================================"
