function [output1, output2, output3] = check_argument_recovered(theta0, delta_theta, k)
%CHECK_ARGUMENT_RECOVERED 
%   
    theta       = theta0 + delta_theta;

    theta0_k    = k * theta0;
    theta_k     = k * theta;
    theta_k_mod = imag(log(exp(1i .* theta_k)));
%     if theta_k_mod <= - pi
%         theta_k_mod = -pi + 10^(-10);
%     elseif theta_k_mod > +pi
%         theta_k_mod = +pi
%     end
       
    eps1 = 10^(-12);
    eps2 = 10^(-12);
    [theta_k_recovered, case_theta0_k_mod, case_theta_k_mod] = argument_nonpv(theta_k_mod, theta0_k, eps1, eps2);
    theta_recovered = theta_k_recovered ./ k;
    diff = abs(theta_recovered - theta);
    
    output1 = diff;
    output2 = case_theta0_k_mod;
    output3 = case_theta_k_mod;
end

