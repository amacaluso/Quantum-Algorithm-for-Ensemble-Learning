# Quantum-Algorithm-for-Ensemble-Learning

This repository contains the code to reproduce the results in the paper *Quantum Algorithm for Ensemble Learning*, that will be published in the proceedings at the [21st Italian Conference on Theoretical Computer Science (ICTCS 2020)](https://sites.google.com/view/ictcs-2020/home/accepted-papers),
14-16 September 2020, Ischia, Italy. The code for the implementation of the quantum circuits uses the [IBM Qiskit](https://qiskit.org/) environment.
The three notebooks also cover all the technical details omitted in the paper.

## Description

The code is organised as follows:
- *Quantum Ensemble of Swap Test.ipynb* contains the details about the implementation of the ensemble of two swap tests. Also, it explains all equations omitted in the paper.
- *Multiple Experiments for Quantum Ensemble (Simulator).ipynb* uses the quantum algorithm to produce the ensemble of two swap tests. It generates 20 small datasets and compare the results of quantum ensemble with the ensemble computed classically. 
- *Quantum Swap Test.ipynb* explains in detail the swap test by performing also simulation considering a small dataset.



The script *Utils.py* contains the import of the needed packages and all the custom routines for the circuit generation.

The script *Visualization.py* contains the custom routines for plot the results as reported in the paper.

The script *run_all.py* implements the experiments of 20 random generated dataset in quantum simulator and real device.

## Issues

For any issues or questions related to the code, open a new git issue or send a mail to antonio.macaluso2@unibo.it
