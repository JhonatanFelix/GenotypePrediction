# GenotypePrediction

Independent Python implementations of classical genomic prediction models.

## Implemented models

| Model | Status | Notes |
|---|---|---|
| Ridge/SNP-BLUP | working | primal ridge baseline |
| GBLUP | partial | dual kernel method under development |
| BayesC fixed-q | prototype | Gibbs sampler |
| BayesCπ | prototype | learns global inclusion q |
| BayesR | planned | mixture-of-normals prior |
| BayesRC | planned | annotation-stratified mixture prior |

## Installation

```bash
git clone ...
cd GenotypePrediction
pdm install