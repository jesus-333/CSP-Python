function X = trial_cov(trials)
    % Calculate the covariance for each trial and return their average
    [nchannels, nsamples, ntrials] = size(trials);
    covs = zeros(nchannels, nchannels, ntrials);
    for i = 1:ntrials
        covs(:,:,i) = (squeeze(trials(:,:,i)) * squeeze(trials(:,:,i))') / nsamples;
    end

    X = mean(covs, 3);
end