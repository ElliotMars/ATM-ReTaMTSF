from pathlib import Path
from pivot_stage2.common import ROOT, sha, digest, read_json, write_json, verify, seal, new_directory, versions
DATA = ROOT / 'dataset/PIVOT/stage2_v2'
ART = ROOT / 'artifacts/pivot_stage2/v2'
HOUR = 3600 * 10**9
TZS = ('fixed_cet', 'berlin', 'utc')
HORIZONS = (24, 72, 168, 336)
