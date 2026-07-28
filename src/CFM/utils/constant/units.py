# constants/units.py
import numpy as np

# Power conversions
DBM_TO_WATT = 1e-3
WATT_TO_DBM_FACTOR = 10 / np.log(10)

# Length conversions
KM_TO_M = 1000.0
NM_TO_M = 1e-9

# Time conversions
PS_TO_S = 1e-12
PS2_TO_S2 = 1e-24
PS3_TO_S3 = 1e-36
PS4_TO_S4 = 1e-48

# Frequency conversions
THZ_TO_HZ = 1e12
GHZ_TO_HZ = 1e9

# Utility conversions (optional)
def dbm_to_watt(dbm):
    return 1e-3 * 10 ** (dbm / 10)

def watt_to_dbm(watt):
    return 10 * np.log10(watt / 1e-3)
