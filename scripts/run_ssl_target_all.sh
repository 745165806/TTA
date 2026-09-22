#!/bin/bash

set -e


cd /media/dell/data/fakeAudioDection/TTA

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta


mkdir -p logs


##################################
# GPU0
# ASV2021 LA
##################################

(
CUDA_VISIBLE_DEVICES=0 \
python -m eptta.cli run-tta \
--config configs/local_v2/ssl_aasist_target/ssl_asv2021_la_frozen.yaml


CUDA_VISIBLE_DEVICES=0 \
python -m eptta.cli run-tta \
--config configs/local_v2/ssl_aasist_target/ssl_asv2021_la_ep.yaml

) > logs/ssl_la.log 2>&1 &



##################################
# GPU1
# ASV2021 DF
##################################

(
CUDA_VISIBLE_DEVICES=1 \
python -m eptta.cli run-tta \
--config configs/local_v2/ssl_aasist_target/ssl_asv2021_df_frozen.yaml


CUDA_VISIBLE_DEVICES=1 \
python -m eptta.cli run-tta \
--config configs/local_v2/ssl_aasist_target/ssl_asv2021_df_ep.yaml

) > logs/ssl_df.log 2>&1 &




##################################
# GPU2
# ITW
##################################

(
CUDA_VISIBLE_DEVICES=2 \
python -m eptta.cli run-tta \
--config configs/local_v2/ssl_aasist_target/ssl_itw_frozen.yaml


CUDA_VISIBLE_DEVICES=2 \
python -m eptta.cli run-tta \
--config configs/local_v2/ssl_aasist_target/ssl_itw_ep.yaml

) > logs/ssl_itw.log 2>&1 &



wait


echo "SSL target scoring finished"
