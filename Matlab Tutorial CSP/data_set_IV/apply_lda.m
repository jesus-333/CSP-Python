function prediction = apply_lda(test, W, b)
%     Applies a previously trained LDA to new data.
%     arguments:
%         test - An array (features x trials) containing the data
%         W    - The project matrix W as calculated by train_lda()
%         b    - The offsets b as calculated by train_lda()
%     returns:
%         A list containing a classlabel for each trial
    ntrials = size(test, 2);

    prediction = [];
    for i = 1:ntrials
        % The line below is a generalization for:
        % result = W(0) * test(0,i) + W(1) * test(1,i) - b
        result = W * test(:,i) - b;
        if result <= 0
            prediction = [prediction 1];
        else
            prediction = [prediction 2];
        end
    end
end