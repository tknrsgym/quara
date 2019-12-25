%% k=5, Positive imaginary part
k = 5;
target = 0.5 * pi;
error = 0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi_type2(a)
a_out = 2.0 * pi * n_out + residual_out

%% k=5, negative imaginary part
k = 5;
target = -0.5 * pi;
error = 0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi_type2(a)
a_out = 2.0 * pi * n_out + residual_out

%% k=5, Positive imaginary part
k = 5;
target = 0.5 * pi;
error = -0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi_type2(a)
a_out = 2.0 * pi * n_out + residual_out




%% k=5, negative imaginary part
k = 5;
target = -0.5 * pi;
error = -0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi_type2(a)
a_out = 2.0 * pi * n_out + residual_out


%% k=7, Positive imaginary part
k = 7;
target = 0.5 * pi;
error = 0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi(a)
a_out = 2.0 * pi * n_out + residual_out
a_out - a



%% k=7, Positive imaginary part
k = 7;
target = 0.5 * pi;
error = -0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi(a)
a_out = 2.0 * pi * n_out + residual_out
a_out - a


%% k=7, Negative imaginary part
k = 7;
target = -0.5 * pi;
error = 0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi(a)
a_out = 2.0 * pi * n_out + residual_out
a_out - a


%% k=7, Negative imaginary part
k = 7;
target = -0.5 * pi;
error = -0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi(a)
a_out = 2.0 * pi * n_out + residual_out
a_out - a


%% k=4, Negative imaginary part
k = 4;
target = -0.5 * pi;
error = -0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi(a)
a_out = 2.0 * pi * n_out + residual_out
a_out - a


%% k=100, Negative imaginary part
k = 100;
target = -0.5 * pi;
error = -0.01;
a = k * (target + error)

[n_out, residual_out] = mod_2pi(a)
a_out = 2.0 * pi * n_out + residual_out
a_out - a

a_out ./ k - (target + error)
residual_out ./ k