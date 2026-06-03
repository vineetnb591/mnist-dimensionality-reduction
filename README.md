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
- `Q1_binary_svm.m` — Linear vs Gaussian SVM (digits 6 vs 8)
- `Q2_multiclass_svm.m` — One-vs-All and One-vs-One classification
- `Q3_kmeans.m` — K-Means clustering (K = 4, 5, 6, 8)
- `Q4_hierarchical.m` — Hierarchical clustering (Euclidean vs L1)
- `Q5_kernel_pca.m` — Kernel PCA + all tasks
- `Q6_laplacian_eigenmaps.m` — Laplacian Eigenmaps + all tasks
- `compute_multiclass_metrics.m` — Helper: precision, recall, specificity

## Methodology

### 1. Binary Classification — Gaussian vs Linear SVM (Digits 6 vs 8)
- 80/20 stratified train-test split, pixel values normalized to [0,1]
- Gaussian RBF kernel outperformed linear kernel across all metrics
- Gaussian kernel better captured nonlinear decision boundaries between 
  visually similar digits

### 2. Multi-Class Classification — OvA vs OvO
- 6-class problem using MATLAB's `fitcecoc` with linear SVM template
- One-vs-One achieved higher precision (95.41%) and specificity (99.10%)
  vs One-vs-All (94.35%, 98.90%)
- OvO's pairwise classifiers better handled subtle inter-digit differences

### 3. K-Means Clustering
- Tested K = 4, 5, 6, 8 using Euclidean distance
- Cluster quality improved with K; best silhouette score (0.1319) at K=8
- Digits with distinct shapes (1, 7) formed cleaner clusters than 
  visually similar pairs (5 vs 6)

### 4. Hierarchical Clustering
- Compared Euclidean vs L1 (Manhattan) distance with average and Ward linkage
- L1 + Ward linkage achieved best purity (~88.5%) at 5 clusters
- L1 distance proved more robust to noise in high-dimensional pixel space

### 5. Kernel PCA
- Gaussian kernel, σ=5, retained 4,565 components for 95% variance
- Used 200 components manually for computational balance
- Boosted classification accuracy to ~99.2% in downstream SVM tasks

### 6. Laplacian Eigenmaps
- k=20 nearest neighbours, 50 retained dimensions
- Preserved local geometric structure of digit manifolds
- Matched Kernel PCA on classification (99.01%) and achieved best 
  hierarchical clustering purity (88.5%)

## Tools & Technologies
**Language:** MATLAB R2024b  
**Techniques:** SVM (Linear, RBF), K-Means, Hierarchical Clustering, 
Kernel PCA, Laplacian Eigenmaps  
**Dataset:** MNIST (mnist_all.mat) — 70,000 samples, 28×28 grayscale images  
**Metrics:** Accuracy, Precision, Recall, F1, Specificity, Cluster Purity, 
Silhouette Score  

## Video Walkthrough
[YouTube Presentation](https://youtu.be/LE3bM48zvWo)
