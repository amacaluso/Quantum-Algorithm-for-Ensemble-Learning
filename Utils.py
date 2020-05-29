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




def state_prep(x):
    backend = Aer.get_backend('unitary_simulator')

    x = normalize_custom(x)

    qreg = QuantumRegister(1)
    qc = QuantumCircuit(qreg)
    # Run the quantum circuit on a unitary simulator backend
    qc.initialize(x,[qreg])
    job = execute(qc, backend)
    result = job.result()
    
    U = result.get_unitary(qc)
    S = Operator(U)
    return S




def classic_swap_test(x,y):
    sim = dot(x, y)/(norm(x)*norm(y))
    out = 1/2 + (sim**2)/2
    return out
    
def classic_ensemble(x1, x2, x_test):
    avg = np.mean([classic_swap_test(x1, x_test), classic_swap_test(x2, x_test)])       
    return avg



def quantum_swap_test(a,b):
    a = normalize_custom(a)
    b = normalize_custom(b)
    
    ancilla = QuantumRegister(1)
    v1 = QuantumRegister(1, 'a')
    v2 = QuantumRegister(1, 'b')
    
    c = ClassicalRegister(1, 'c')
    
    qc = QuantumCircuit(v1, v2, ancilla, c)
    
    S1 = state_prep(a)
    qc.unitary(S1, [0], label='$S_{a}$')

    S2 = state_prep(b)
    qc.unitary(S2, [1], label='$S_{b}$')

    qc.barrier()
    
    qc.h(ancilla[0])
    qc.cswap(ancilla[0], v1[0], v2[0])
    qc.h(ancilla[0])
    qc.measure(ancilla[0], c)
    return qc

    

def quantum_ensemble(x1, x2, x_test):
    n_obs = 2
    d=1
    
    control = QuantumRegister(d, 'd')
    data = QuantumRegister(n_obs, 'x')
    temp = QuantumRegister(1, 'temp')
    data_test = QuantumRegister(1, 'x^{test}')
    avg = QuantumRegister(1, 'f_{i}')
    c = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(control, data, temp, data_test, avg, c)

    S1 = state_prep(x1)
    qc.unitary(S1, [1], label='$S_x$')
    
    S2 = state_prep(x2)
    qc.unitary(S2, [2], label='$S_x$')
    
    S3 = state_prep(x_test)
    qc.unitary(S3, [4], label='$S_{x}$')
    
    qc.h(control)

    qc.barrier()

    qc.cswap(control[0], data[0], temp[0])

    qc.x(control[0])
    qc.cswap(control[0], data[1], temp[0])
    qc.barrier()
    
    
    qc.h(avg[0])
    qc.cswap(avg[0], temp[0], data_test[0])
    qc.h(avg[0])
    qc.measure(avg[0], c)
    return qc



def plot_multiple_experiments(runs, avg, ens, clas, filename='Simulator'):
    
    x = np.arange(runs)

    ax = plt.subplot()
    ax.plot(x, ens, color='orange', label='qEnsemble', zorder=1, linewidth=5)
    ax.plot(x, avg, color='steelblue', label='qAVG')
    ax.scatter(x, clas, label='cAVG', color='sienna', zorder=2, linewidth=.5)

    #ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-.2, 1.05)
    ax.grid(alpha=0.3)
    #ax.set_xticks(np.round(np.arange(-1, 1.1, .4), 1).tolist())
    ax.set_title('Comparison', size=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.15),
              ncol=4, fancybox=True, shadow=True, fontsize = 12)
    plt.savefig('output/'+ filename+'.png', dpi = 300, bbox_inches='tight')