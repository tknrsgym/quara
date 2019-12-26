clear;

% Settings
list_k    = [1, 3, 5, 7, 9, 11, 13];
list_Nrep = [100, 1000, 10000, 100000];
Nave      = 100;

num_k    = numel(list_k);
num_Nrep = numel(list_Nrep);

% Data

diff_HSgb_G_target_prepared = 3.999479590474600e-04;

filename = './OutputFiles/191226_squared_error_GSgb_G_uniformWeight.mat';
data = load(filename);
error = data.squared_error_HSgb_G;

% Analysis

% 1. Sample Mean & Sample Variance
mean_error     = zeros(num_k, num_Nrep);
variance_error = zeros(num_k, num_Nrep);
for i_k = 1:num_k
    for i_Nrep = 1:num_Nrep
        sum1 = 0.0;
        sum2 = 0.0;
        for i_ave = 1:Nave
            x = error(i_k, i_Nrep, i_ave);
            sum1 = sum1 + x;
            sum2 = sum2 + x * x;
        end
        sum1 = sum1 ./ Nave;
        mean_error(i_k, i_Nrep) = sum1;
        
        sum2 = sum2 ./ (Nave -1);
        variance_error(i_k, i_Nrep) = sum2 - Nave .* sum1 * sum1 ./ (Nave -1);
    end
end
sdev_error = sqrt(variance_error);


% Output of Analysis Results
x = list_Nrep;
y0 = diff_HSgb_G_target_prepared .* ones(1, num_Nrep);
z0 = zeros(1, num_Nrep);
plot(x, y0, '-k', 'LineWidth', 1.5);
hold on;

str_legend = {'Prepared-Target'};
list_k_check = [1,13];
i_count = 0;
for i_k = 1:num_k
    k = list_k(i_k);
    if ismember(k, list_k_check)
        i_count = i_count + 1;
        str_add = strcat('k=', num2str(k));
        str_legend{i_count +1} = str_add;
        
        y = mean_error(i_k, :);
        z = sdev_error(i_k, :);
        
        if k == 1
            linespec = '-ok';
        else 
            linespec = '--xk';
        end

        errorbar(x,y,z,linespec, 'LineWidth', 1);
        hold on;
    end
end
legend(str_legend)
set(gca, 'XScale','log', 'YScale','log')
grid on

filename_plot = './OutputFiles/191226_plot_uniformWeight.pdf';
saveas(gcf, filename_plot)
