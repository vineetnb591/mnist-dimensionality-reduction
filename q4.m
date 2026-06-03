clear; clc;

load('mnist_all.mat');

% Combine digits
digits = [1, 2, 5, 6, 7, 8];
X = [];
for d = digits
    varname = sprintf('train%d', d);
    if exist(varname, 'var')
        digit_data = eval(varname);
        X = [X; digit_data];
    end
end

% Normalize
X = double(X) / 255;

% Sample 100 points for visualization (since HC is O(n^2))
rng(42);
sample_idx = randperm(size(X, 1), 100);
X_sample = X(sample_idx, :);

% Compute pairwise distances
D_euclidean = pdist(X_sample, 'euclidean');
D_L1 = pdist(X_sample, 'cityblock');

% Create linkage trees
Z_euclidean = linkage(D_euclidean, 'average');
Z_L1 = linkage(D_L1, 'average');

% Plot dendrograms
figure;
subplot(1, 2, 1);
dendrogram(Z_euclidean, 0);
title('Hierarchical Clustering (Euclidean Distance)');

subplot(1, 2, 2);
dendrogram(Z_L1, 0);
title('Hierarchical Clustering (L1 / Manhattan Distance)');