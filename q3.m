clear; clc;

load('mnist_all.mat');

% Digits assigned
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

% Apply K-Means for K = 4, 5, 6, 8
Ks = [4, 5, 6, 8];
figure;
for i = 1:length(Ks)
    K = Ks(i);
    fprintf('\nRunning K-Means for K = %d\n', K);
    
    % Run K-Means
    [idx, C] = kmeans(X, K, 'Distance', 'sqeuclidean', 'MaxIter', 300, 'Replicates', 5);
    
    % Reshape cluster centers and show as images
    for j = 1:K
        subplot(length(Ks), max(Ks), (i - 1) * max(Ks) + j);
        imagesc(reshape(C(j, :), 28, 28));
        colormap gray;
        axis off;
        title(sprintf('K=%d\\newlineCluster %d', K, j));
    end
end

