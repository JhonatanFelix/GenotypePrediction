"""SNP-BLUP implementation / Ridge"""

# from __future__ import annotations # Really necessary??

import numpy as np

from GPtools.data.preprocessing import GenotypeStandardizer
from GPtools.data.splitting import train_test_split_genomic
from GPtools.evaluation.metrics import mse, r2
