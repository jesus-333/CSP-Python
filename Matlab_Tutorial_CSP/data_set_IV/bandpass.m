function trials_filt = bandpass(trials, lo, hi, sample_rate)
%     Designs and applies a bandpass filter to the signal.
%     arguments:
%         trials      - An array (channels x samples x trials) containing the signal
%         lo          - Lower frequency bound (in Hz)
%         hi          - Upper frequency bound (in Hz)
%         sample_rate - Sample rate of the signal (in Hz)
%     returns:
%         An array (channels x samples x trials) containing the bandpassed
%         signal
    ntrials = size(trials, 3);

    % The butter() function takes the filter order: higher numbers mean a
    % sharper frequency cutoff, but the resulting signal might be shifted
    % in time, lower numbers mean a soft frequency cutoff, but the
    % resulting signal less distorted in time. It also takes the lower and
    % upper frequency bounds to pass, divided by the niquist frequency,
    % which is the sample rate divided by 2:
    [a, b] = butter(3, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)]);

    % Applying the filter to each trial.
    % filtfilt operates across the first non-singleton dimension, so we
    % permute the trials array so that the samples are on the first
    % dimension
    trials = permute(trials, [2, 1, 3]);

    trials_filt = zeros(size(trials));
    for i = 1:ntrials
        trials_filt(:,:,i) = filtfilt(a, b, trials(:,:,i));
    end

    % Undo the permutation performed earlier
    trials_filt = permute(trials_filt, [2, 1, 3]);
end
