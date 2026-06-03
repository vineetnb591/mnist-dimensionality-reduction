# Dimensionality Reduction and Classification on MNIST Digits

## Overview
A comprehensive comparative study of classification and clustering strategies on a 
subset of the MNIST handwritten digits dataset (digits 1, 2, 5, 6, 7, 8 — ~29,000 
training samples). The project benchmarks linear vs nonlinear approaches across 
supervised and unsupervised learning tasks.

## Key Results
| Task | Method | Performance |
|---|---|---|
| Binary Classification (6 vs 8) | Gaussian SVM | **99.19% accuracy** |
| Binary Classification (6 vs 8) | Linear SVM | 98.34% accuracy |
| Multi-class (6 digits) | One-vs-One SVM | **95.41% precision** |
| Multi-class (6 digits) | One-vs-All SVM | 94.35% precision |
| Kernel PCA + Classification | Linear SVM | ~99.2% accuracy |
| Laplacian Eigenmaps + Classification | Linear SVM | 99.01% accuracy |
| Hierarchical Clustering (L1) | Ward Linkage | **88.5% purity** |
| K-Means (best) | K=8 | 63% purity |

## Project Structure
