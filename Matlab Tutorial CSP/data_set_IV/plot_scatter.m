function plot_scatter(right, foot)
    figure();
    hold on;
    scatter(right(1,:), right(end,:), 'b');
    scatter(foot(1,:), foot(end,:), 'r');
    hold off;
    xlabel('Last component');
    ylabel('First component');
    title('Right hand versus foot movement');
    legend({'right', 'foot'});
end

