function [W, b] = train_lda(class1, class2)
%     Trains the LDA algorithm.
%     arguments:
%         class1 - An array (features x trials) for class 1
%         class2 - An array (features x trails) for class 2)
%     returns:
%         The projection matrix W
%         The offset b
    m1 = mean(class1');
    m2 = mean(class2');

    W = (m2 - m1) / (cov(class1') + cov(class2'));
    b = (m1 + m2) * W' / 2;
end

