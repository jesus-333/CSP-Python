function trials_csp = apply_mix(W, trials)
    % Apply a mixing matrix to each trial (basically multiply W with the
    % EEG signal matrix)
    ntrials = size(trials, 3);
    trials_csp = zeros(size(trials));

    for i = 1:ntrials
        trials_csp(:,:,i) = W' * squeeze(trials(:,:,i));
    end
end