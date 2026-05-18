"""Global BayesR with a four-class mixture prior on SNP effects."""

from __future__ import annotations

import pandas as pd
import numpy as np

from genotypeprediction.data.preprocessing import GenotypeStandardizer
from genotypeprediction.evaluation.metrics import r2

