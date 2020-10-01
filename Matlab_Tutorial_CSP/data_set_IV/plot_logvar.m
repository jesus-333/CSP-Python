function plot_logvar(trials)
%     Plots the log-var of each channel/component.
%     arguments:
%         trials - Dictionary containing the trials (log-vars x trials) for 2 classes.
    figure('Position', [100, 100, 1200, 400]);

      
    cl_lab=fieldnames(trials); % gpires
    cl_1=cl_lab{1};            % gpires 
    cl_2=cl_lab{2};            % gpires 
    nchannels=size(trials.(cl_1),1); % gpires
    
    x0 = (1:nchannels);
    x1 = (1:nchannels) + 0.4;

    y0 = mean(trials.(cl_1), 2); % gpires
    y1 = mean(trials.(cl_2), 2); % gpires

    hold on;
    bar(x0, y0, 0.5, 'b');
    bar(x1, y1, 0.4, 'r');
    hold off;

    xlim([-0.5, nchannels+0.5]);

    grid();
    title('log-var of each channel/component');
    xlabel('channels/components');
    ylabel('log-var');
    legend({'right', 'foot'});
end