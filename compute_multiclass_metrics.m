function [precision, recall, specificity] = compute_multiclass_metrics(C)
    K = size(C, 1); 
    precision_all = zeros(K, 1);
    recall_all = zeros(K, 1);
    specificity_all = zeros(K, 1);
    
    for i = 1:K
        TP = C(i, i);
        FP = sum(C(:, i)) - TP;
        FN = sum(C(i, :)) - TP;
        TN = sum(C(:)) - TP - FP - FN;
        
        precision_all(i) = TP / (TP + FP + eps);
        recall_all(i) = TP / (TP + FN + eps);
        specificity_all(i) = TN / (TN + FP + eps);
    end
    
    precision = mean(precision_all);
    recall = mean(recall_all);
    specificity = mean(specificity_all);
end