#!/bin/bash


cd /media/dell/data/fakeAudioDection/TTA

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta


mkdir -p outputs_v2/paper


python -m eptta.cli report \
--runs outputs_v2/aasist/target-runs \
--out outputs_v2/paper/aasist_report.csv



python -m eptta.cli report \
--runs outputs_v2/ssl_aasist/target-runs \
--out outputs_v2/paper/ssl_report.csv



echo "paper reports generated"

