load('mnist_all.mat');
digits = [1, 2, 5, 6, 7, 8];
X = []; y = [];

for d = digits
    varname = sprintf('train%d', d);
    digit_data = eval(varname);
    X = [X; double(digit_data) / 255];
    y = [y; d * ones(size(digit_data, 1), 1)];
end

% Subsample
rng(42);
sample_idx = randperm(size(X, 1), 8000);
X_small = X(sample_idx, :);
y_small = y(sample_idx);

% Apply Laplacian Eigenmaps
no_dims = 50;
k = 20;
X_le = laplacian_eigenmap(X_small, no_dims, k);

% Q1: Binary Classification (6 vs 8)
mask_bin = (y_small == 6 | y_small == 8);
X_bin = X_le(mask_bin, :);
y_bin = y_small(mask_bin);
y_bin(y_bin == 6) = 0;
y_bin(y_bin == 8) = 1;

model_q1 = fitcsvm(X_bin, y_bin, 'KernelFunction', 'linear');
pred_q1 = predict(model_q1, X_bin);
conf_q1 = confusionmat(y_bin, pred_q1);
disp('Confusion matrix (SVM 6 vs 8, Laplacian Eigenmaps):');
disp(conf_q1);

% Q2: Multi-class OvA and OvO
template = templateSVM('KernelFunction', 'linear');

model_ova = fitcecoc(X_le, y_small, 'Coding', 'onevsall', 'Learners', template);
pred_ova = predict(model_ova, X_le);
conf_ova = confusionmat(y_small, pred_ova);
disp('Confusion matrix (OvA, Laplacian Eigenmaps):');
disp(conf_ova);

model_ovo = fitcecoc(X_le, y_small, 'Coding', 'onevsone', 'Learners', template);
pred_ovo = predict(model_ovo, X_le);
conf_ovo = confusionmat(y_small, pred_ovo);
disp('Confusion matrix (OvO, Laplacian Eigenmaps):');
disp(conf_ovo);

% Q3: K-Means Clustering
Ks = [4, 5, 6, 8];
figure;
for i = 1:length(Ks)
    K = Ks(i);
    [idx, C] = kmeans(X_le, K);
    for j = 1:K
        subplot(length(Ks), max(Ks), (i-1)*max(Ks)+j);
        bar(C(j,:)); axis tight;
        title(sprintf('K = %d\\newlineCluster %d', K, j));
    end
end

% Q4: Hierarchical Clustering
sample_idx = randperm(size(X_le, 1), 100);
X_sample = X_le(sample_idx, :);

D_euclidean = pdist(X_sample, 'euclidean');
D_L1 = pdist(X_sample, 'cityblock');

Z_euc = linkage(D_euclidean, 'average');
Z_l1 = linkage(D_L1, 'average');

figure;
subplot(1,2,1); dendrogram(Z_euc); title('Euclidean (Laplacian Eigenmaps)');
subplot(1,2,2); dendrogram(Z_l1); title('L1 (Laplacian Eigenmaps)');
