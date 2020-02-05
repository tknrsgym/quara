function list_state = FileImport_state( filename, dim, num_state )
%FILEIMPORT_1QUBIT_STATE 
    list = csvread(filename);
    
    for ix = 1:num_state
        row = dim*(ix -1) +1;
        list_state(ix).mat = list(row:row+dim -1, 1:dim);
    end

end

