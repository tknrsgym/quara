function list_weight = FileImport_weight( filename_weight, num_outcome )
%FILEIMPORT_WEIGHT 
    list = csvread(filename_weight);
    num_element = numel(list);
    num_schedule = num_element / (num_outcome * num_outcome);
    
    for ix = 1:num_schedule
        row = num_outcome*(ix -1) +1;
        list_weight(ix).mat = list(row:row+num_outcome -1, 1:num_outcome);
    end
  
end

