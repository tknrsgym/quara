function [output1, output2] = mod_2pi_type2(input)
%MODE_2PI is a function for calculating n and Delta satisfying 
% input = 2.0 * pi * n + Delta.
% period: -pi < <= +pi.
    b = input + pi;
    c = 2.0 * pi;
    
    if (b <= pi)
        n = floor(b./c);
    else
        n = ceil(b./c) -1;
    end
    %
    %n = round(b./c);
    %
    %n = fix(b./c);
    %if (b <= 0.0)
    %    n = n -1;
    %end
    Delta = b - c.*n - pi;
    
    output1 = n;
    output2 = Delta;
end

