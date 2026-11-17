# Benchmark Results

Tasks: asr, sid, er, asv

Seeds: 0,1,2


| Upstream | ASR | SID | ER | ASV |
|---|---|---|---|---|
| XLS-R 300M (vanilla) | — | — | — | — |
| mHuBERT-147 | — | — | — | — |
| WavLM Large | — | — | — | — |
| XLS-R 300M (Si/Ta 200h, raw-trained, norm eval) | 0.904 ± 0.094 | 0.587 ± 0.003 | 0.341 ± 0.016 | 0.266 ± 0.060 |
| XLS-R 300M (Si/Ta 200h, raw-trained, raw eval) | 0.906 ± 0.097 | 0.569 ± 0.006 | 0.319 ± 0.016 | 0.317 ± 0.084 |
| XLS-R 300M (Si/Ta 200h) | 0.898 ± 0.098 | 0.564 ± 0.002 | 0.358 ± 0.010 | 0.263 ± 0.056 |
| XLS-R 300M (Si/Ta 200h, language-homogeneous updates) | 0.703 ± 0.095 | 0.872 ± 0.005 | 0.396 ± 0.023 | 0.137 ± 0.008 |


## Change in the primary metric (mean over languages and seeds)

Signed differences in each metric's own units (WER/EER: lower is better; accuracy: higher is better). Blank where either side has not been benchmarked.

| Upstream | Adaptation mode | Compared with | Δ ASR | Δ SID | Δ ER | Δ ASV |
|---|---|---|---|---|---|---|
| XLS-R 300M (Si/Ta 200h, raw-trained, norm eval) | joint_200h_mixed_batch_raw_trained | joint 200h mixed batches | +0.005 | +0.022 | -0.017 | +0.003 |
| XLS-R 300M (Si/Ta 200h, raw-trained, raw eval) | joint_200h_mixed_batch_raw_trained | joint 200h mixed batches | +0.008 | +0.005 | -0.039 | +0.054 |
| XLS-R 300M (Si/Ta 200h, language-homogeneous updates) | joint_200h_language_homogeneous | joint 200h mixed batches | -0.196 | +0.308 | +0.038 | -0.126 |

