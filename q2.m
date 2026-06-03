clear; clc;

load('mnist_all.mat');  

% Digits to use
digits = [1, 2, 5, 6, 7, 8];

% Initialize data
X = [];
y = [];

% Combine selected digit datasets
for d = digits
    varname = sprintf('train%d', d);
    if exist(varname, 'var')
        digit_data = eval(varname);  
        X = [X; digit_data];
        y = [y; d * ones(size(digit_data, 1), 1)];
    end
end

% Normalize
X = double(X) / 255;

% Proceed with filtered version
labels_to_use = [1,2,5,6,7,8];
mask = ismember(y, labels_to_use);
X_filtered = X(mask, :);
y_filtered = y(mask);

% Stratified split with 5-fold
cv = cvpartition(y_filtered, 'KFold', 5); 
fold = 1;

X_train = X_filtered(training(cv, fold), :);
y_train = y_filtered(training(cv, fold));
X_test = X_filtered(test(cv, fold), :);
y_test = y_filtered(test(cv, fold));

% Check class balance
disp('Train labels:');
tabulate(y_train)
disp('Test labels:');
tabulate(y_test)

% One-vs-All SVM
template = templateSVM('KernelFunction','linear','BoxConstraint',1);
model_ova = fitcecoc(X_train, y_train, 'Learners', template, 'Coding', 'onevsall');
pred_ova = predict(model_ova, X_test);

% One-vs-One SVM
model_ovo = fitcecoc(X_train, y_train, 'Learners', template, 'Coding', 'onevsone');
pred_ovo = predict(model_ovo, X_test);

% Confusion matrices
conf_ova = confusionmat(y_test, pred_ova);
conf_ovo = confusionmat(y_test, pred_ovo);

% Compute multi-class precision, recall, specificity
[p_ova, r_ova, s_ova] = compute_multiclass_metrics(conf_ova);
[p_ovo, r_ovo, s_ovo] = compute_multiclass_metrics(conf_ovo);

% Display results
fprintf('\n--- One-vs-All (OvA) Results ---\n');
fprintf('Precision: %.4f | Recall (Sensitivity): %.4f | Specificity: %.4f\n', p_ova, r_ova, s_ova);

fprintf('\n--- One-vs-One (OvO) Results ---\n');
fprintf('Precision: %.4f | Recall (Sensitivity): %.4f | Specificity: %.4f\n', p_ovo, r_ovo, s_ovo);
