format long;

%% 
% 
% theta0      = pi/2;
% delta_theta = 0.1;
% theta       = theta0 + delta_theta
% k           = 19;
% 
% theta0_k    = k * theta0;
% theta_k     = k * theta;
% theta_k_mod = imag(log(exp(1i .* theta_k)));
% 
% eps1 = 10^(-12);
% eps2 = 10^(-12);
% [theta_k_recovered, case_theta0_k_mod, case_theta_k_mod] = argument_nonpv(theta_k_mod, theta0_k, eps1, eps2);
% 
% abs(theta_k_recovered - theta_k)
% 
% theta_recovered = theta_k_recovered ./ k
% abs(theta_recovered - theta)
%case_theta0_k_mod
%case_theta_k_mod


%% Test of argument_nonpv.m

kmax = 100;
A = 20;

list_theta0 = [-2.0 .* pi, -1.5.*pi, -pi, -0.5.*pi, -0.25.*pi, 0, 0.25 * pi, 0.5*pi, 0.75*pi, pi, 1.5*pi, 2*pi];
num_theta0 = size(list_theta0, 2);
for i_theta0 = 1:num_theta0
    theta0 = list_theta0(i_theta0);
    for k = 1:kmax
        delta_theta = [-A:+1:+A] ./A;
        delta_theta = (pi ./ k - 10^(-15)).* delta_theta;
        num_delta_theta = size(delta_theta, 2);
        for i_delta_theta = 1:num_delta_theta
            dth = delta_theta(i_delta_theta);
            [diff(i_delta_theta), case_theta0_k_mod(i_delta_theta), case_theta_k_mod(i_delta_theta)]  = check_argument_recovered(theta0, dth, k);
        end
        [diff_max, index] = max(diff);
        if diff_max > 10^(-10) & case_theta_k_mod(index) ~= 3
            disp('Recover of argument failed!')
            theta0
            k
            del_theta = delta_theta(index)
            del_theta_k = del_theta .* k
            %abs(del_theta_k - pi)
            case_theta0_k_mod_ = case_theta0_k_mod(index)
            case_theta_k_mod_ = case_theta_k_mod(index)
            diff_max
            break;
        end
    end
end
%plot(delta_theta, diff)