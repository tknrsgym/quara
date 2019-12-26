clear;
format long
char_precision = '%.15e';

path_importFolder = '/Users/sugiyamac3/GitHub/quara/tests/';
path_outputFolder = '/Users/sugiyamac3/GitHub/quara/pseudoData/191226_test/';

list_k    = [1, 3, 5, 7, 9, 11, 13];
list_Nrep = [100, 1000, 10000, 100000];
Nave      = 200;

num_k    = numel(list_k);
num_Nrep = numel(list_Nrep);

list_probDist = load(strcat(path_outputFolder, 'probDist_prepared.mat'));
%list_probDist.full_list_probDist_prepared;

list_empiDist = load(strcat(path_outputFolder, 'empiDist.mat'));
%list_empiDist.full_list_empiDist;

%% Check of Empirical Distributions

for i_k = 1:num_k
    probDist(:,:) = list_probDist.full_list_probDist_prepared(i_k, :, :);
    for i_Nrep = 1:num_Nrep
        sum = 0.0;
        for i_ave = 1:Nave
            empiDist(:,:) = list_empiDist.full_list_empiDist(i_k, i_ave, i_Nrep, :, :);
           
            for i_schedule = 1:size(probDist, 1)
                for i_omega = 1:size(probDist, 2)
                    dif = probDist(i_schedule, i_omega) - empiDist(i_schedule, i_omega); 
                    sum = sum + dif * dif;
                end
            end
        end
        error_empiDist(i_k, i_Nrep) = sum ./ Nave;
   
    end 
end
error_empiDist

%% Check of Sample Variance


for i_k = 1:num_k
    probDist(:,:) = list_probDist.full_list_probDist_prepared(i_k, :, :);
    for i_schedule = 1:size(probDist, 1)
        for i_omega = 1:size(probDist, 2)
             variance(i_k, i_schedule, i_omega) = probDist(i_schedule, i_omega) * (1.0 - probDist(i_schedule, i_omega));  
        end
    end
    for i_Nrep = 1:num_Nrep
        Nrep = list_Nrep(i_Nrep);
        sum1 = 0.0;
        sum2 = 0.0;
        sum3 = 0.0;
        for i_ave = 1:Nave
            empiDist(:,:) = list_empiDist.full_list_empiDist(i_k, i_ave, i_Nrep, :, :);
            weight_case1 = list_weight_from_list_empiDist_case1(empiDist, Nrep);
            weight_case2 = list_weight_from_list_empiDist_case2(empiDist, Nrep); 
            for i_schedule = 1:size(probDist, 1)
                for i_omega = 1:size(probDist, 2)
                    var0 = variance(i_k, i_schedule, i_omega);
                    var1 = 1.0 / weight_case1(i_schedule).mat(i_omega, i_omega);
                    var2 = 1.0 / weight_case2(i_schedule).mat(i_omega, i_omega);
                    var3 = empiDist(i_schedule, i_omega) * (1.0 - empiDist(i_schedule, i_omega));
                    dif1 = var1 - var0;
                    dif2 = var2 - var0;
                    dif3 = var3 - var0;
                    sum1 = sum1 + dif1 * dif1;
                    sum2 = sum2 + dif2 * dif2;
                    sum3 = sum3 + dif3 * dif3;
                end
            end
        end
        error_variance_1(i_k, i_Nrep) = sum1 ./ Nave;
        error_variance_2(i_k, i_Nrep) = sum2 ./ Nave;
        error_variance_3(i_k, i_Nrep) = sum3 ./ Nave;
    
    end 
end
error_variance_1
error_variance_2
error_variance_3

% for i_k = 1:num_k
%     variance(i_k, :,1)
% end




