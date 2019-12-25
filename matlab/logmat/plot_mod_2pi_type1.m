% x = 5*(-pi):pi/100:5*pi;
% y0 = x;
% y1 = imag(log(exp(i*x)));
% [n, Delta] = mod_2pi(x);
% y2 = 2.0 * pi * n + Delta;
% 
% 
% figure
% plot(x,y0, x,y1)
% 
% plot(x,y2)
% 
% plot(x, n, x, Delta)
% 
% a = -3.0;
% [n_a, Delta_a] = mod_2pi(a)

num = 100000;
x_max = 13.0*pi;
x_min = -13.0*pi;
for index = 0:num
    x(index+1)  = x_min  + index * (x_max - x_min)./num;
    y0(index+1) = x(index+1);
    y1(index+1) = imag(log(exp(1.0i*x(index+1))));
    [n(index+1), Delta(index+1)] = mod_2pi_type1(x(index+1)); 
    y2(index+1) = 2.0*pi*n(index+1) + Delta(index+1);
    del_y(index+1) = y2(index+1) - y1(index+1);
end

plot(x,y0, x,y1, x, y2, x,n, x, Delta)
%plot(x,y1)
%plot(x,n, x,Delta)
%plot(x, y2-x)
xticks([-13*pi -12*pi -11*pi -10*pi -9*pi -8*pi -7*pi -6*pi -5*pi -4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi 5*pi 6*pi 7*pi 8*pi 9*pi 10*pi 11*pi 12*pi 13*pi])
xticklabels({'-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi'})
xticklabels({'-13\pi', '-12\pi', '-11\pi', '-10\pi', '-9\pi', '-8\pi', '-7\pi', '-6\pi', '-5\pi', '-4\pi', '-3\pi', '-2\pi', '-pi', '0', 'pi', '2\pi', '3\pi', '4\pi', '5\pi', '6\pi', '7\pi', '8\pi', '9\pi', '10\pi', '11\pi', '12\pi', '13\pi'})
grid on
