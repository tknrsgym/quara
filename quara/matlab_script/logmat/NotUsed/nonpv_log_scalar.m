function output = nonpv_log_scalar(y_mod, y0)
%NONPV_LOG_SCALAR returns y
%   
    assert(-pi < y_mod | y_mod <= pi);
    
    y0_mod = imag(log(exp(1i .* y0)));

    dy = 0;
    if y0_mod >= 0
       if -pi < y_mod & y_mod < y0_mod - pi 
           dy = y_mod + 2 .* pi - y0_mod;
       else
           dy = y_mod - y0_mod;
       end
    else
        if pi + y0_mod < y_mod & y_mod <= pi
           dy = y_mod - 2 .* pi - y0_mod; 
        else
           dy = y_mod - y0_mod; 
        end
    end

    n_branch = branch_log_imaginary(y0);
    y = 2.0 .* pi .* n_branch + y0_mod + dy;

    output = y;
end

