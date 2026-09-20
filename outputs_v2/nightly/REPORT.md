# Nightly TTA Report

_regenerated at 2026-09-20 18:58:36_

## CPU pytest
- status: PASS (115 passed in 3.78s)

## In-the-Wild
- status: PASS, sample_count=31779, class_counts(0=bonafide/1=spoof)={'0': 19963, '1': 11816}, speakers=54, audio_missing=0, header_problems=0

## Model: aasist
- pipeline state: DONE
- freeze: OK bundle.json
- cache+resources: OK
- source selection: OK
- source-select candidate verify: checked=11 ok=True problems=[]
- source-select EER (on select):
  - ep_tta: EER=1.0735%
  - frozen: EER=1.0735%
  - multiview_mean: EER=1.1561%
- selected epoch: 66, source_val EER=0.0000%
- target metrics (EER ratio & %, AUROC, fixed-threshold FPR/FNR):
  - asv2019_control_test (role=control_test):
    - EP: EER=0.0208 (2.0802%), AUROC=0.997629, FPR=0.032767, FNR=0.007530, threshold=-6.133879
    - Frozen: EER=0.0208 (2.0802%), AUROC=0.997629, FPR=0.032767, FNR=0.007530, threshold=-6.133879
    - Multiview Mean: EER=0.0231 (2.3058%), AUROC=0.997208, FPR=0.039565, FNR=0.007185, threshold=-6.133879
  - asv2021_df_eval (role=target_test):
    - EP: EER=0.2318 (23.1825%), AUROC=0.837220, FPR=0.467752, FNR=0.027132, threshold=-6.133879
    - Frozen: EER=0.2318 (23.1825%), AUROC=0.837220, FPR=0.467752, FNR=0.027138, threshold=-6.133879
    - Multiview Mean: EER=0.2340 (23.4044%), AUROC=0.835680, FPR=0.489744, FNR=0.018626, threshold=-6.133879
  - asv2021_la_eval (role=target_test):
    - EP: EER=0.1280 (12.7970%), AUROC=0.922921, FPR=0.414754, FNR=0.002452, threshold=-6.133879
    - Frozen: EER=0.1280 (12.8037%), AUROC=0.922877, FPR=0.414822, FNR=0.002430, threshold=-6.133879
    - Multiview Mean: EER=0.1328 (13.2829%), AUROC=0.918338, FPR=0.418602, FNR=0.002310, threshold=-6.133879
  - in_the_wild (role=target_test):
    - EP: EER=0.3921 (39.2095%), AUROC=0.659692, FPR=0.889295, FNR=0.002624, threshold=-6.133879
    - Frozen: EER=0.3921 (39.2095%), AUROC=0.659693, FPR=0.889295, FNR=0.002624, threshold=-6.133879
    - Multiview Mean: EER=0.3971 (39.7089%), AUROC=0.652990, FPR=0.893703, FNR=0.002370, threshold=-6.133879

## Model: ssl_aasist
- pipeline state: CACHE_RESOURCES
- freeze: OK bundle.json
- selected epoch: 7, source_val EER=0.0000%

## Notes
- EER is reported both as ratio and percent. Score direction: larger = spoof. Labels: 0=bonafide, 1=spoof.
- tDCF/minDCF are unavailable (no official ASV scores/protocol inputs); not fabricated.

