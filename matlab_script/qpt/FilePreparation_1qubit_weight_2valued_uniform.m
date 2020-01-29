function [] = FilePreparation_1qubit_weight_2valued_uniform( filename_weight, filename_schedule, char_precision )
%FILEPREPARATION_1QUBIT_WEIGHT Summary of this function goes here
%   Detailed explanation goes here
    list_schedule = csvread(filename_schedule);
    num_schedule = numel(list_schedule)/2;
    
    weight = eye(2);% becase of 2-valued.
    list = weight;
    for id = 2:num_schedule
        list = vertcat(list, weight);
    end
    
    dlmwrite(filename_weight, list, 'precision', char_precision, 'delimiter', ',');
end

