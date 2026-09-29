# RawNet2: fp32 vs the canonical int8 device artifact on DF21 and ITW (sol1, 2026-09-29)

- **int8** = the D39 canonical artifact: recipe r09, calibration seed 2, **int8-mixed (K=16)**,
  sha256 `3fd6523aa4097fdf...`, checked against the ledger before scoring
  (`conversion_sensitivity/rawnet2/r09_seed2/scores_{itw,df21_verified}.csv`; ORT CPU, 8 shards,
  launcher `sol1:xfer/canonical_rawnet2_scoring.sh`).
- **fp32** = PyTorch on CUDA with TF32 off, the reproduction checkpoint
  (`scores_fp32_pytorch_{itw,df21_verified}.csv` here; `sol1:xfer/fp32_rawnet2_scoring.sh`).
- DF21 = the D34 verified surface (455,508 trials). Both runs decoded exactly 149,983 class-A
  files through the MD5-verified fallback. The fresh fp32 scores equal the D33 gate scores on
  the 305,525 clean trials (max |d| 4.8e-6).
- All thresholds are the frozen 19LA-dev taus (`tau_rawnet2.json`); nothing is re-calibrated.
  Output of `python analysis/rawnet2_transfer.py`:

```

[ITW] fp32 file: fp32/pytorch  int8 file: int8/ort

## ITW: n=31779 (bona fide 19963)
EER fp32 49.001%  int8 51.106%  dEER +2.105 pp
tau_eer  fp32 HTER  51.41% (FRR  33.75 FAR  69.07) TTG  +5.44 | int8 HTER  52.30% (FRR  34.95 FAR  69.66) TTG  +2.55 | flips 5345
tau_frr1 fp32 HTER  50.35% (FRR  42.78 FAR  57.91) TTG  +4.38 | int8 HTER  51.57% (FRR  43.27 FAR  59.87) TTG  +1.82 | flips 6221
tau_frr5 fp32 HTER  46.86% (FRR  83.49 FAR  10.23) TTG  +0.89 | int8 HTER  50.64% (FRR  78.85 FAR  22.43) TTG  +0.89 | flips 5505

[DF21 verified] fp32 file: fp32/pytorch  int8 file: int8/ort

## DF21 verified: n=455508 (bona fide 12067)
EER fp32 19.655%  int8 20.898%  dEER +1.243 pp
tau_eer  fp32 HTER  19.64% (FRR  33.68 FAR   5.60) TTG  +0.58 | int8 HTER  20.38% (FRR  33.99 FAR   6.76) TTG  +0.04 | flips 12022
tau_frr1 fp32 HTER  19.73% (FRR  35.17 FAR   4.29) TTG  +0.67 | int8 HTER  20.40% (FRR  35.38 FAR   5.42) TTG  +0.06 | flips 10109
tau_frr5 fp32 HTER  21.21% (FRR  41.00 FAR   1.43) TTG  +2.15 | int8 HTER  21.01% (FRR  39.74 FAR   2.29) TTG  +0.68 | flips 5482
fresh fp32 vs D33 gate scores on the 305525 clean trials: max |d| 4.77e-06

## DF21 clean only (sensitivity): n=305525 (bona fide 6991)
EER fp32 15.064%  int8 16.440%  dEER +1.377 pp
tau_eer  fp32 HTER  15.58% (FRR  26.48 FAR   4.67) TTG  +0.92 | int8 HTER  16.50% (FRR  26.88 FAR   6.13) TTG  +0.19 | flips 7953
tau_frr1 fp32 HTER  15.66% (FRR  27.72 FAR   3.61) TTG  +1.01 | int8 HTER  16.52% (FRR  28.11 FAR   4.94) TTG  +0.21 | flips 6679
tau_frr5 fp32 HTER  17.52% (FRR  33.86 FAR   1.18) TTG  +2.86 | int8 HTER  17.20% (FRR  32.28 FAR   2.12) TTG  +0.89 | flips 3662

per codec (verified surface): EER fp32 -> int8, HTER@tau_eer fp32 -> int8
  high_m4a   n= 46671  EER  24.53 ->  25.97  HTER  27.98 ->  28.49
  high_mp3   n= 46483  EER  24.55 ->  26.19  HTER  27.57 ->  28.25
  high_ogg   n= 57108  EER  16.01 ->  17.00  HTER  14.92 ->  15.66
  low_m4a    n= 46267  EER  24.06 ->  25.77  HTER  27.24 ->  27.86
  low_mp3    n= 46082  EER  24.21 ->  26.11  HTER  27.22 ->  28.07
  low_ogg    n= 55040  EER  16.12 ->  17.14  HTER  15.05 ->  15.92
  mp3m4a     n= 55997  EER  16.11 ->  17.32  HTER  14.89 ->  15.67
  nocodec    n= 46995  EER  24.42 ->  25.11  HTER  26.74 ->  27.45
  oggm4a     n= 54865  EER  15.89 ->  17.47  HTER  15.06 ->  15.92
```

**ITW is at chance for this checkpoint (fp32 EER 49.0%, AUC 0.510).** Harness checks: labels
19,963 / 11,816 (the published counts), all audio 16 kHz, and 31,489 distinct scores (not
constant). The model rates nearly all ITW audio as spoof-like (bona fide mean score -4.5 vs
-0.09 on 19LA eval). Müller et al. report ~34% for a RawNet2 that *they* trained with their
own pipeline. This is the ASVspoof 2021 DF baseline checkpoint (D33), so the figures are not
directly comparable. Treat this as provisional until a second model has been scored on ITW.
The ITW loader now refuses unknown labels instead of mapping them to spoof (none occur in
meta.csv).

**Update (same day):** AASIST and AASIST-L fp32 on ITW, through the same loader and scorer,
come out above chance: EER 43.02% (AUC 0.601) and 44.46% (AUC 0.590). This supports reading
RawNet2's 49.0% as a property of this checkpoint rather than a pipeline fault
(`python analysis/itw_fp32_descriptives.py`).
