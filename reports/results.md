# Benchmark Results

Tasks: asr, sid, er, sd, asv, ic

Seeds: 0,1,2


| Upstream | Protocol | ASR | SID | ER | SD | ASV | IC |
|---|---|---|---|---|---|---|---|
| XLS-R 300M (Si/Ta 200h, raw-trained, norm eval) | unknown | 0.904 ± 0.094 | 0.587 ± 0.003 | 0.341 ± 0.016 | — | 0.266 ± 0.060 | — |
| XLS-R 300M (Si/Ta 200h, raw-trained, raw eval) | unknown | 0.906 ± 0.097 | 0.569 ± 0.006 | 0.319 ± 0.016 | — | 0.317 ± 0.084 | — |
| HuBERT Large (Si/Ta 200h), SLSB v0.3 | unknown | 0.739 ± 0.087 | 0.998 ± 0.001 | 0.399 ± 0.006 | 0.066 ± 0.020 | 0.178 ± 0.008 | 0.848 ± 0.159 |
| mHuBERT-147 (vanilla), SLSB v0.3 | unknown | 0.714 ± 0.096 | 0.997 ± 0.001 | 0.406 ± 0.004 | 0.127 ± 0.067 | 0.196 ± 0.009 | 0.828 ± 0.170 |
| mHuBERT-147 (Si/Ta 200h), SLSB v0.3 | unknown | 0.685 ± 0.111 | 0.994 ± 0.002 | 0.397 ± 0.017 | 0.148 ± 0.042 | 0.201 ± 0.003 | 0.887 ± 0.109 |

Not benchmarked yet: xlsr300m-copt200h-norm, xlsr300m, hubert-large-copt200h, hubert-large, mhubert147-copt200h, mhubert147, wav2vec2-large-lv60-copt200h, wav2vec2-large-lv60, wavlm-large-copt200h, wavlm-large
