# constants/model.py
import numpy as np

# --- GN model coefficients ---
GN_COEFF_A = np.array([
    +1.0436e0, -1.1878e0, +1.0573e0, -1.8309e+1,
    +1.6665e0, -1.0020e0, +9.0933e0, +6.6420e-3,
    +8.4481e-1, -1.8530e0, +9.4539e-1, -1.5421e+1,
    +1.0229e0, -1.1440e0, +1.1393e-2, +3.8070e+5,
    +1.4785e+3, -2.2593e0, -6.7997e-1, +2.0215e0,
    -2.9781e-1, +5.5130e-1, -3.6718e-1, +1.1486e0
])

M_CONST = 10

# Common GN factors
SCI_FACTOR = 16/27
XCI_FACTOR = 2/np.pi

# Default system parameters
DEFAULT_ROLL_OFF = 0.1
DEFAULT_CHANNEL_BAUD = 64e9
