function mappedX = laplacian_eigenmap(X, no_dims, k)
    D = squareform(pdist(X));
    n = size(D, 1);
    sigma = median(D(:));
    W = zeros(n);
    [~, idx] = sort(D, 2);
    for i = 1:n
        neighbors = idx(i, 2:k+1);
        W(i, neighbors) = exp(-D(i, neighbors).^2 / (2 * sigma^2));
    end
    W = max(W, W');
    Dg = diag(sum(W, 2));
    L = Dg - W;
    [eigvec, eigval] = eig(L, Dg);
    eigval = diag(eigval);
    [~, ind] = sort(eigval);
    eigvec = eigvec(:, ind);
    mappedX = eigvec(:, 2:no_dims+1);
    
    for i = 1:size(mappedX, 2)
        mappedX(:,i) = mappedX(:,i) / norm(mappedX(:,i));
    end
end
