load('mnist_all.mat');  

% Concatenate data
X = [train6; train8];
y = [6*ones(size(train6, 1), 1); 8*ones(size(train8, 1), 1)];

% Relabel: 6 -> 0, 8 -> 1
y(y == 6) = 0;
y(y == 8) = 1;

% Normalize pixel values
X = double(X) / 255;

cv = cvpartition(y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Train with linear kernel
linearModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
linearPred = predict(linearModel, X_test);

% Train with RBF kernel
rbfModel = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', 'auto', 'Standardize', true);
rbfPred = predict(rbfModel, X_test);

% Confusion matrices
cm_linear = confusionmat(y_test, linearPred);
cm_rbf = confusionmat(y_test, rbfPred);

% Accuracy
acc_linear = sum(linearPred == y_test) / length(y_test);
acc_rbf = sum(rbfPred == y_test) / length(y_test);

% Precision, Recall, F1
precision = @(cm) cm(2,2) / (cm(2,2) + cm(1,2));
recall = @(cm) cm(2,2) / (cm(2,2) + cm(2,1));
f1 = @(p, r) 2 * (p * r) / (p + r);

% Linear metrics
p_lin = precision(cm_linear);
r_lin = recall(cm_linear);
f1_lin = f1(p_lin, r_lin);

% RBF metrics
p_rbf = precision(cm_rbf);
r_rbf = recall(cm_rbf);
f1_rbf = f1(p_rbf, r_rbf);

% Display
fprintf('Linear SVM:\n Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n', acc_linear, p_lin, r_lin, f1_lin);
fprintf('RBF SVM:\n Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n', acc_rbf, p_rbf, r_rbf, f1_rbf);
