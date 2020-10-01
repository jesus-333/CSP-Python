function plot_psds(trials_PSD, freqs, chan_ind, chan_lab, maxy, nclasses)
%     adaptado por gpires
%     Plots PSD data calculated with psd().
%     arguments:
%         trials   - The PSD data, as returned by psd()
%         freqs    - The frequencies for which the PSD is defined, as returned by psd()
%         chan_ind - List of indices of the channels to plot
%         chan_lab - (optional) List of names for each channel
%         maxy     - (optional) Limit the y-axis to this value
%         nclasses - número de classes - gpires
    figure('Position', [100, 100, 1000, 300]);

    nchans = length(chan_ind);

    % Maximum of 3 plots per row
    nrows = ceil(nchans / 3);
    ncols = min(3, nchans);

    % Enumerate over the channels
    for i = 1:length(chan_ind)
        ch = chan_ind(i);

        % Figure out which subplot to draw to
        subplot(nrows,ncols,i);

        % Plot the PSD for each class
        hold on;
        colors = {'b', 'r'};
        cl_lab=fieldnames(trials_PSD); % gpires
        for j = 1:nclasses
            cl = cl_lab{j};
            plot(freqs, squeeze(mean(trials_PSD.(cl)(ch,:,:), 3)), colors{j});
        end
        hold off;

        % All plot decoration below...
        xlim([1, 30]);
        ylim([0, maxy]);

        grid()

        xlabel('Frequency (Hz)')
        title(chan_lab{i});
        legend(cl_lab);
    end
end
