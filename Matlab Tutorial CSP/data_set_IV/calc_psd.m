function [trials_PSD, freqs] = calc_psd(trials, sample_rate)
%     adaptado por gpires
%     Calculates for each trial the Power Spectral Density (PSD).
%     arguments:
%         trials - An array (channels x samples x trials) containing the signal.
%     returns:
%         An array (channels x PSD x trials) containing the PSD for each trial.
%         A list containing the frequencies for which the PSD was computed (useful for plotting later)

    ntrials = size(trials, 3);
    nchannels = size(trials, 1); %gpires
    nsamples = size(trials, 2); %gpires
    trials_PSD = zeros(nchannels, 101, ntrials);

    % Iterate over trials and channels
    for trial = 1:ntrials
        for ch = 1:nchannels
            % Calculate the PSD
            [PSD, freqs] = pwelch(squeeze(trials(ch,:,trial)), nsamples, 0, nsamples, sample_rate);
            trials_PSD(ch, :, trial) = PSD;
        end
    end
end
