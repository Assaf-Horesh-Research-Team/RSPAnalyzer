# __init__.py

# Importing submodules
import pandas as pd
import numpy as np
import typing
from typing import Any,Tuple,List
import os
import glob  # Add this line to import the 'glob' module
from matplotlib import pyplot as plt
import scipy.optimize
import scipy.special as sf
import scipy.signal as sp
import scipy.linalg
from scipy.integrate import quad
from scipy.constants import electron_mass, Boltzmann, Planck
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import minimize, differential_evolution, approx_fprime
import emcee
import time
from multiprocessing import Pool
import corner
import math
import tqdm
from emcee import EnsembleSampler
from numpy.typing import NDArra

#constatns
C1 = 6.27e18 #cgs units - too big. maybe units err
# C1 = 6.27
C5 = 7.52e-24 #for gamma=3, cgs
C6 = 7.97e-41 #for gamma=3, cgs
F= 0.5 #emission filling factor of an approximate spherical volume with outter radius r
c = 3e10
Me = 9.1e-28
EL = Me * c**2 #cm*g/s
EPS_B = 0.1
EPS_E = 0.1
mJY = 1e26
MPC = 3.08e24
DAYS = 86400
Ghz = 1e9
eps_b=0.1
alpha = 1
m_electron = 9e-28 
q_electron = 4.8*1e-10
b=0.5
m_proton = 1.6726231*1e-24 #g
light_speed = 2.9*1e10

# Package-level constants
VERSION = '1.0'

# Initialization code
print(f"Package {__name__} initialized.")
