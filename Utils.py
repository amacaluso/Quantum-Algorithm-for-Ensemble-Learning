# Classical packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

from numpy import dot
from numpy.linalg import norm
from numpy.random import uniform

from scipy.stats import ttest_ind

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute, IBMQ, Aer
from qiskit.compiler import transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit import Gate
from qiskit.quantum_info.operators import Operator




import warnings
warnings.filterwarnings('ignore')

def create_dir(path):
    if not os.path.exists(path):
        print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print('The directory', path, ' already exists')


def normalize_custom(x, C=1):
    M = x[0] ** 2 + x[1] ** 2

    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed


def pdf(url):
    return HTML('<embed src="%s" type="application/pdf" width="100%%" height="600px" />' % url)


def retrieve_proba(r):
    try:
        p0 = r['0'] / (r['0'] + r['1'])
        p1 = 1 - p0
    except:
        if list(r.keys())[0] == '0':
            p0 = 1
            p1 = 0
        elif list(r.keys())[0] == '1':
            p0 = 0
            p1 = 1
    return [p0, p1]


def exec_simulator(qc, n_shots = 8192):
    # QASM simulation
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots = n_shots)
    results = job.result()
    answer = results.get_counts(qc)
    return answer




def run_real_device(qc, backend, shots=8192):
    job = execute(qc, backend, shots=shots)
    results = job.result()
    r = results.get_counts(qc)
    return r