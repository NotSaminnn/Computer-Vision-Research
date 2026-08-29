# transition_training

Source        : transphy3d/test @ `3b023eb84592efd3de63d40f7738fb4f9b3768f7`
Licence       : Apache-2.0, as declared on the Hugging Face dataset card
Device        : NVIDIA GeForce RTX 5070 (sm_120)
Supervision   : 11,264,000 samples from 2750 frame pairs, stride 5
Split         : held out 6/25 sequences: ['11', '13', '19', '2', '28', '6']

| quantity | value |
|---|---|
| final train MSE | 0.0185228 |
| final val MSE | 0.0189134 |
| rigid (H_D) baseline val MSE | 0.0263162 |
| pooled ratio vs rigid (NOT the result) | 0.718695 |
| train seconds | 123.83 |

## The number that matters

| subset | rows | rigid baseline | model | ratio |
|---|---|---|---|---|
| consistent | 2,416,960 (97.9%) | 7.73723e-07 | 0.00228769 | **2956.727924** |
| occlusion_affected | 52,928 (2.1%) | 1.22801 | 0.778126 | **0.633646** |

Occlusion boundaries carry orders of magnitude more residual than optics does,
so the pooled ratio is dominated by them. **`consistent` is the subset that
speaks to the optics claim**; a ratio above 1 there means the model is worse
than the rigid H_D hypothesis exactly where transparency and reflection live,
however good the pooled number looks.
