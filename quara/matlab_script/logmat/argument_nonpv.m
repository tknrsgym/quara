function [output1, output2, output3] = argument_nonpv(theta_k_mod, theta0_k, eps1, eps2)
%ARGUMENT_NONPV returns non-principal argument closest to a reference.
%  - Basic equations: theta = theta0 + delta_theta
%                     k * theta = k * theta0 + k * delta_theta
%  - theta: unknown argument
%  - theta0: a reference, known.
%  - delta_theta = theta - theta0, the difference, unknown.
%  - k: a positive integer, known.
%  - Assumption: abs( k * delta_theta ) < pi.
%  - theta_k_mod = mod(theta * k, 2 * pi), a real value in (- pi, + pi].
%  - theta0_k = theta0 * k, a reference, a real value.
%  - eps1: a numerical tolerance parameter around the boundaries, -pi and
%          +pi. eps1 >= 0.
%  - eps2: a numerical tolerance parameter around 
%  - output = theta_k = k * theta.
    assert(eps1 >= 0);
    assert(eps2 >= 0);
    assert(- pi <= theta_k_mod & theta_k_mod <= pi);

    theta0_k_mod = imag(log(exp(1i.*theta0_k)));
    delta_theta_k = 0;
    case_theta0_k_mod = -1;
    case_theta_k_mod  = -1;
    % (1)
    if -pi + eps1 <= theta0_k_mod & theta0_k_mod < 0
        case_theta0_k_mod = 1;
        
        % (i)
        if -pi <= theta_k_mod & theta_k_mod < theta0_k_mod + pi - eps2
            case_theta_k_mod = 1;
            delta_theta_k = theta_k_mod - theta0_k_mod;
        % (ii)    
        elseif theta0_k_mod + pi + eps2 < theta_k_mod & theta_k_mod <= pi
            case_theta_k_mod = 2;
            delta_theta_k = theta_k_mod - theta0_k_mod - 2.0 .* pi;
        % (iii)    
        else
            case_theta_k_mod = 3; 
        end
    
    % (2)    
    elseif 0 <= theta0_k_mod & theta0_k_mod < pi - eps1
        case_theta0_k_mod = 2;
        
        % (i)
        if -pi <= theta_k_mod & theta_k_mod < theta0_k_mod - pi - eps2
            case_theta_k_mod = 1;
            delta_theta_k = 2.0 .* pi - theta0_k_mod + theta_k_mod;
        % (ii)    
        elseif theta0_k_mod - pi + eps2 < theta_k_mod & theta_k_mod <= pi
            case_theta_k_mod = 2;
            delta_theta_k = theta_k_mod - theta0_k_mod;
        % (iii)    
        else
            case_theta_k_mod = 3;             
        end
         
    % (3)    
    else
        case_theta0_k_mod = 3;
        
        % (i)
        if -pi <= theta_k_mod & theta_k_mod < -eps2
            case_theta_k_mod = 1;
            delta_theta_k = theta_k_mod + pi;
        % (ii)
        elseif eps2 < theta_k_mod & theta_k_mod <= pi
            case_theta_k_mod = 2;
            delta_theta_k = theta_k_mod - pi;
        % (iii)
        else
            case_theta_k_mod = 3;
        end
    end
    theta_k = theta0_k + delta_theta_k;
    
    output1 = theta_k;
    output2 = case_theta0_k_mod;
    output3 = case_theta_k_mod;
end

