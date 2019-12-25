function num_schedule = FilePreparation_2qubit_schedule( filename, num_state, num_povm )
%FILEPREPARATION_ID_1QUBIT Summary of this function goes here
%   Detailed explanation goes here
   id = 0; 
   for i_state = 1:num_state
       for j_povm = 1:num_povm
           id = id + 1;
           list(id,1) = i_state;
           list(id,2) = j_povm;
       end
   end
   csvwrite(filename, list); 
   num_schedule = id;
end

