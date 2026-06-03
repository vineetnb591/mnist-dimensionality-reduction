load('mnist_all.mat');

digits = [1, 2, 5, 6, 7, 8];
X = [];
y = [];

for d = digits
    varname = sprintf('train%d', d);
    if exist(varname, 'var')
        digit_data = eval(varname);
        X = [X; digit_data];
        y = [y; d * ones(size(digit_data, 1), 1)];
    end
end

X = double(X) / 255;

% Subsample 8000 points to reduce kernel size
rng(42);
sample_idx = randperm(size(X, 1), 8000);
X_small = X(sample_idx, :);
y_small = y(sample_idx);

% Continue with Kernel PCA using X_small
X_centered = X_small - mean(X_small);

% Apply Gaussian kernel PCA
sigma = 5;  
K = pdist2(X_centered, X_centered, 'euclidean');
K = exp(-K.^2 / (2 * sigma^2));

% Center the kernel matrix
N = size(K, 1);
oneN = ones(N, N) / N;
K_centered = K - oneN * K - K * oneN + oneN * K * oneN;

% Eigen-decomposition
[eigvecs, eigvals_matrix] = eig(K_centered);
eigvals = diag(eigvals_matrix);

% Sort in descending order
[eigvals_sorted, idx] = sort(eigvals, 'descend');
eigvecs_sorted = eigvecs(:, idx);

% Choose number of components
eigvals_total = sum(eigvals_sorted);
eigvals_cumsum = cumsum(eigvals_sorted);
num_components = find(eigvals_cumsum >= 0.95 * eigvals_total, 1);


plot(cumsum(eigvals_sorted) / sum(eigvals_sorted));
xlabel('Number of Components');
ylabel('Cumulative Variance');
title('Kernel PCA Explained Variance');
grid on;


fprintf('Selected %d components for ~95%% variance\n', num_components);
num_components = 200; % Manually chosen for balance
X_kpca = K_centered * eigvecs_sorted(:, 1:num_components);

% Q1: Binary Classification for digits 6 vs 8 using KPCA
mask = (y_small == 6) | (y_small == 8);
X_q1 = X_kpca(mask, :);
y_q1 = y_small(mask);
y_q1(y_q1 == 6) = 0;
y_q1(y_q1 == 8) = 1;

model_q1 = fitcsvm(X_q1, y_q1, 'KernelFunction', 'linear');
pred_q1 = predict(model_q1, X_q1);

conf_q1 = confusionmat(y_q1, pred_q1);
disp('Confusion matrix (SVM 6 vs 8, KPCA):');
disp(conf_q1);

% Q2: OvA and OvO SVM Classification using KPCA
template = templateSVM('KernelFunction','linear','BoxConstraint',1);

model_ova = fitcecoc(X_kpca, y_small, 'Learners', template, 'Coding', 'onevsall');
model_ovo = fitcecoc(X_kpca, y_small, 'Learners', template, 'Coding', 'onevsone');

pred_ova = predict(model_ova, X_kpca);
pred_ovo = predict(model_ovo, X_kpca);

conf_ova = confusionmat(y_small, pred_ova);
conf_ovo = confusionmat(y_small, pred_ovo);

disp('Confusion matrix (OvA):');
disp(conf_ova);

disp('Confusion matrix (OvO):');
disp(conf_ovo);

% Q3: K-Means clustering on KPCA data
Ks = [4, 5, 6, 8];
figure;

for i = 1:length(Ks)
    K = Ks(i);
    fprintf('\nRunning K-Means for K = %d\n', K);
    
    [idx, C] = kmeans(X_kpca, K, 'Distance', 'sqeuclidean', 'MaxIter', 300, 'Replicates', 5);
    
    for j = 1:K
        subplot(length(Ks), max(Ks), (i - 1) * max(Ks) + j);
        bar(C(j, :)); axis tight;
        title(sprintf('K = %d\\newlineCluster %d', K, j));
    end
end

% Q4: Hierarchical Clustering (Euclidean and L1) on KPCA
sample_idx = randperm(size(X_kpca, 1), 100);
X_sample = X_kpca(sample_idx, :);

D_euclidean = pdist(X_sample, 'euclidean');
D_L1 = pdist(X_sample, 'cityblock');

Z_euc = linkage(D_euclidean, 'average');
Z_l1 = linkage(D_L1, 'average');

figure;
subplot(1, 2, 1); dendrogram(Z_euc, 0); title('Euclidean Distance (KPCA)');
subplot(1, 2, 2); dendrogram(Z_l1, 0); title('L1 Distance (KPCA)');
