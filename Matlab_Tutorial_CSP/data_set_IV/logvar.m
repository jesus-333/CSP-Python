function x = logvar(trials)
%     Calculate the log-var of each channel.
%     arguments:
%         trials - An array (channels x samples x trials) containing the signals.
%     returns:
%         An array (channels x trials) containing for each channel the logvar of the signal

    % var operates along the first non singleton dimension, permute the
    % array to align the samples along the first dimension
    trials = permute(trials, [2, 1, 3]);

    % Calculate the log-var
    x = log(squeeze(var(trials)));
end
