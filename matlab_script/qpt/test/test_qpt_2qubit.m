%% Analysis of Marginal distibution constraints in Quantum Tomography
%  Test by 2-qubit case
%  Takanori Sugiyama
%  2016/06/05 Sun. - 
%
%% 

tic

%% Setting: optimizer
opt = sdpsettings;
opt.solver = 'sedumi';



%% Setting 
% rho: prepared input state
% E: an element of prepared POVM
% Gate: prepared gates
    
    matI = eye(2);% Identity matrix on 1-qubit
    matX = [0, 1; 1, 0];% Pauli X
    matY = [0, -1i; 1i, 0];% Pauli Y
    matZ = [1, 0; 0, -1];% Pauli Z
    
        
    Pauli(1).mat = matI;
    Pauli(2).mat = matX;
    Pauli(3).mat = matY;
    Pauli(4).mat = matZ;
    
    % Input state
    % Maximally entangled state + completely mixed state
    state_Bell = 0.5.*[1, 0, 0, 1; 0, 0, 0, 0; 0, 0, 0, 0; 1, 0, 0, 1];
    state_CMix = 0.25.*eye(4);
    ratio_mix  = 1;
    rho_mat    = (1 - ratio_mix).*state_Bell + ratio_mix.*state_CMix;
    rho_vec    = reshape(rho_mat,[16,1]);
  
    
    
    % Rotation for projective measurements along with X and Y axes
    mat_Z2X = [1, 1; 1, -1]./sqrt(2);% Hadamard gate
    mat_Z2Y = [1, 1; 1i, -1i]./sqrt(2);
    
    % Measurement POVM (Z-direction)
    E(1).mat = [1, 0; 0, 0];% POVM element 1
    E(2).mat = [0, 0; 0, 1];% POVM element 2
   
    POVM_2qubit(2,2).mat = zeros(4);
    for i_x1 = 1:2
        for i_x2 = 1:2
            POVM_2qubit(i_x1,i_x2).mat = kron(E(i_x1).mat, E(i_x2).mat); 
        end
    end
    
    
    %
    % Hilbert-Schmidt representation
    for i_x1 = 1:2
        for i_x2 = 1:2
            POVM_2qubit(i_x1,i_x2).vec = reshape(POVM_2qubit(i_x1,i_x2).mat.', [16,1]);
        end
    end

 
  
    %
    % Sugiyama's setting:
    Gate_1qubit(1).mat = mat_Z2X';% ' is conjugate & transpose
    Gate_1qubit(2).mat = mat_Z2Y';
    Gate_1qubit(3).mat = matI;
  
    
    for I1 = 1:3
        for I2 = 1:3
            Gate_2qubit(I1,I2).mat = kron(Gate_1qubit(I1).mat, Gate_1qubit(I2).mat);
            Gate_2qubit(I1,I2).HS  = kron(Gate_2qubit(I1,I2).mat, conj(Gate_2qubit(I1,I2).mat));
        end
    end
    
  


%% Probability Distributions (Setting 1)
% Calculation of 
%      p_2qubit(x1,x2|i1,i2) = <<E_{x1,x2}| G_{i1,i2} |rho>>
   
     % Allocation of p1, p2, and p3:
     % - Setting 1:
     p_2qubit(1).prob = zeros(2,2,3,3);% p_2qubit(x1,x2|i1,i2) = <<E_{x1,x2}| G_{i1,i2} |rho>>, i1,i2=1~X,2~Y,3~Z, x1,x2 = 1,2
     
     % Calculation:
     for I1 = 1:3
         for I2 = 1:3
             for i_x1 = 1:2
                 for i_x2 = 1:2
                     p_2qubit(1).prob(i_x1,i_x2,I1,I2) = POVM_2qubit(i_x1,i_x2).vec'*Gate_2qubit(I1,I2).HS*rho_vec; 
                 end% i_x2
             end% i_x1
         end% I2
     end% I1    
     
     % vec_s
     vec_s = zeros(4,4);
     
     vec_s(1,1) = trace(rho_mat);
     
     f_change(1) = 1;
     f_change(2) = -1;
     for I1 = 1:3
         for I2 = 1:3
             for i_x1 = 1:2
                 for i_x2 = 1:2
                     vec_s(I1+1,I2+1) = vec_s(I1+1,I2+1) + f_change(i_x1).*f_change(i_x2).*p_2qubit(1).prob(i_x1,i_x2,I1,I2);
                 end% i_x2
             end% i_x1
         end% I2
     end% I1
     
     for I1 = 1:3
         for I2 = 1:3
            for i_x1 = 1:2
                for i_x2 = 1:2
                     vec_s(I1,1) = vec_s(I1,1) + f_change(i_x1).*p_2qubit(1).prob(i_x1,i_x2,I1,I2)./3;
                     vec_s(1,I2) = vec_s(1,I2) + f_change(i_x2).*p_2qubit(1).prob(i_x1,i_x2,I2,I2)./3;
                 end
            end
         end
     end
     
     
 
%     %% Averaging
     num_average = 100;

     num_per_unit = 10000;
     num_unit = 20;

     squarederror_1 = zeros(2,num_unit);
%     
     for m = 1:num_average
 
         %% Sampling
 
         res_sampling_1 = zeros(2,2,3,3);
         ratio_1        = zeros(2,2,3,3);
 
         for n = 1:num_unit
 
             for n_in_unit = 1:num_per_unit
                 for I1 = 1:3
                     for I2 = 1:3
                         rn = rand();   
                         if rn < p_2qubit(1).prob(1,1,I1,I2)
                             res_sampling_1(1,1,I1,I2) = res_sampling_1(1,1,I1,I2) + 1;
                         elseif rn < p_2qubit(1).prob(1,1,I1,I2) + p_2qubit(1).prob(1,2,I1,I2)    
                             res_sampling_1(1,2,I1,I2) = res_sampling_1(1,2,I1,I2) + 1;  
                         elseif rn < p_2qubit(1).prob(1,1,I1,I2) + p_2qubit(1).prob(1,2,I1,I2) + p_2qubit(1).prob(2,1,I1,I2)   
                             res_sampling_1(2,1,I1,I2) = res_sampling_1(2,1,I1,I2) + 1;    
                         else
                             res_sampling_1(2,2,I1,I2) = res_sampling_1(2,2,I1,I2) + 1;
                         end
                     end% I2
                 end% I1
             end% n_in_unit
 
             ratio_1(:,:,:,:) = res_sampling_1(:,:,:,:)./(num_per_unit*n);
 
             %% Estimation
 
             % Linear estimate
             vec_s_L_1 = zeros(4,4);
             vec_s_L_1(1,1) = 1;
             
             f_change(1) = 1;
             f_change(2) = -1;
             for I1 = 1:3
                 for I2 = 1:3
                     for i_x1 = 1:2
                         for I_x2 = 1:2
                             vec_s_L_1(I1 +1,I2 +1) = vec_s_L_1(I1 +1,I2 +1) + f_change(i_x1).*f_change(i_x2).*ratio_1(i_x1,i_x2,I1,I2);
                         end
                     end
                 end
             end
             
             for I1 = 1:3
                 for I2 = 1:3
                     if (~(I1==1) | ~(I2==1))
                        for i_x1 = 1:2
                            for i_x2 = 1:2
                                 vec_s_L_1(I1 +1,1) = vec_s_L_1(I1 +1,1) + f_change(i_x1).*ratio_1(i_x1,i_x2,I1,I2)./3;
                                 vec_s_L_1(1,I2 +1) = vec_s_L_1(1,I2 +1) + f_change(i_x2).*ratio_1(i_x1,i_x2,I2,I2)./3;
                             end
                        end
                     end% if
                 end
             end
 
             squarederror_1(1,n) = squarederror_1(1,n) + norm(vec_s_L_1 - vec_s, 2)^2;
 
             % Constraint least squares estimate
             dim = 4;
             rho_CLS = sdpvar(dim, dim, 'hermitian', 'complex');
             F = [trace(rho_CLS)==1, rho_CLS >=0];

%              obj = [];
%              for I1 = 1:4
%                 for I2 = 1:4
%                     if (~(I1==1) | ~(I2==1))
%                         obj = obj + (trace(rho_CLS* kron(Pauli(I1).mat,Pauli(I2).mat)) - vec_s_L(I1,I2))^2;
%                     end% if
%                 end% I2
%              end% I1

             obj =  (trace(rho_CLS* kron(Pauli(1).mat,Pauli(2).mat)) - vec_s_L_1(1,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(1).mat,Pauli(3).mat)) - vec_s_L_1(1,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(1).mat,Pauli(4).mat)) - vec_s_L_1(1,4))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(1).mat)) - vec_s_L_1(2,1))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(2).mat)) - vec_s_L_1(2,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(3).mat)) - vec_s_L_1(2,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(4).mat)) - vec_s_L_1(2,4))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(1).mat)) - vec_s_L_1(3,1))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(2).mat)) - vec_s_L_1(3,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(3).mat)) - vec_s_L_1(3,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(4).mat)) - vec_s_L_1(3,4))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(1).mat)) - vec_s_L_1(4,1))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(2).mat)) - vec_s_L_1(4,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(3).mat)) - vec_s_L_1(4,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(4).mat)) - vec_s_L_1(4,4))^2;
               
             
             % 
             optimize(F, obj, opt);
             optobj = value(obj);
             optrho = value(rho_CLS);
% 
             vec_s_CLS_1 = zeros(4,4);
             for I1 = 1:4
                 for I2 = 1:4              
                     vec_s_CLS_1(I1,I2) = trace(rho_CLS* kron(Pauli(I1).mat, Pauli(I2).mat)); 
                 end
             end
             squarederror_1(2,n) = squarederror_1(2,n) + norm(vec_s_CLS_1 - vec_s, 2)^2;
         end
     
     end
     
     squarederror_1 = squarederror_1./num_average;
    
     ax1 = subplot(1,2,1);
     plot([1:num_unit]*9*num_per_unit, squarederror_1(1,:), [1:num_unit]*9*num_per_unit, squarederror_1(2,:))
     legend('Linear (XYZ)', 'CLS (XYZ)')
     
     A = [[1:num_unit]*9*num_per_unit; squarederror_1(1,:); squarederror_1(2,:)];
     fileID_1 = fopen('output_Mixed_setting1.txt','w');
     fprintf(fileID_1, '%d %f %f\n',A);
     fclose(fileID_1);
    
    
 
%% Probability Distributions (Setting 2)
% Calculation of 
%      p_2qubit(x1,x2|i1,i2) = <<E_{x1,x2}| G_{i1,i2} |rho>>
   
     % Allocation of p1, p2, and p3:
     % - Setting 2:
     p_2qubit(2).prob = zeros(2,2,4,4);% p_2qubit(x1,x2|i1,i2) = <<E_{x1,x2}| G_{i1,i2} |rho>>, i1,i2=1~X,2~Y,3~Z,4~I x1,x2 = 1,2
     
     % Calculation:
     for I1 = 1:3
         for I2 = 1:3
             for i_x1 = 1:2
                 for i_x2 = 1:2
                     p_2qubit(2).prob(i_x1,i_x2,I1,I2) = POVM_2qubit(i_x1,i_x2).vec'*Gate_2qubit(I1,I2).HS*rho_vec; 
                 end% i_x2
             end% i_x1
         end% I2
     end% I1    
     
     for I = 1:3
         for i_x = 1:2
            p_2qubit(2).prob(i_x,1,I,4) = 0;
            p_2qubit(2).prob(1,i_x,4,I) = 0;
            for i_y = 1:2
                p_2qubit(2).prob(i_x,1,I,4) = p_2qubit(2).prob(i_x,1,I,4) + p_2qubit(2).prob(i_x,i_y,I,1);
                p_2qubit(2).prob(1,i_x,4,I) = p_2qubit(2).prob(1,i_x,4,I) + p_2qubit(2).prob(i_y,i_x,1,I);
            end
         end
     end
     
 
%     %% Averaging
     num_average = 100;

     num_per_unit = 10000;
     num_unit = 12;

     squarederror_2 = zeros(2,num_unit);
%     
     for m = 1:num_average
 
         %% Sampling
 
         res_sampling_2 = zeros(2,2,4,4);
         ratio_2        = zeros(2,2,4,4);
 
         for n = 1:num_unit
 
             for n_in_unit = 1:num_per_unit
                 % Pi^{I1}_{x1} \otimes Pi^{I2}_{x2}
                 for I1 = 1:3
                     for I2 = 1:3
                         rn = rand();   
                         if rn < p_2qubit(2).prob(1,1,I1,I2)
                             res_sampling_2(1,1,I1,I2) = res_sampling_2(1,1,I1,I2) + 1;
                         elseif rn < p_2qubit(2).prob(1,1,I1,I2) + p_2qubit(2).prob(1,2,I1,I2)    
                             res_sampling_2(1,2,I1,I2) = res_sampling_2(1,2,I1,I2) + 1;  
                         elseif rn < p_2qubit(2).prob(1,1,I1,I2) + p_2qubit(2).prob(1,2,I1,I2) + p_2qubit(2).prob(2,1,I1,I2)   
                             res_sampling_2(2,1,I1,I2) = res_sampling_2(2,1,I1,I2) + 1;    
                         else
                             res_sampling_2(2,2,I1,I2) = res_sampling_2(2,2,I1,I2) + 1;
                         end
                     end% I2
                 end% I1
                 % Pi^{I1}_{x1} \otimes I
                 for I1 = 1:3
                    rn = rand();
                    if rn < p_2qubit(2).prob(1,1,I1,4)
                        res_sampling_2(1,1,I1,4) = res_sampling_2(1,1,I1,4) +1;
                    else
                        res_sampling_2(2,1,I1,4) = res_sampling_2(2,1,I1,4) +1;   
                    end
                 end
                 % I \otimes Pi^{I2}_{x2}
                 for I2 = 1:3
                    rn = rand();
                    if rn < p_2qubit(2).prob(1,1,4,I2)
                        res_sampling_2(1,1,4,I2) = res_sampling_2(1,1,4,I2) +1;
                    else
                        res_sampling_2(1,2,4,I2) = res_sampling_2(1,2,4,I2) +1;   
                    end
                 end
                 %   
             end% n_in_unit
 
             ratio_2(:,:,:,:) = res_sampling_2(:,:,:,:)./(num_per_unit*n);
 
             %% Estimation
 
             % Linear estimate
             vec_s_L_2 = zeros(4,4);
             vec_s_L_2(1,1) = 1;
             
             f_change(1) = 1;
             f_change(2) = -1;
             for I1 = 1:3
                 for I2 = 1:3
                     for i_x1 = 1:2
                         for I_x2 = 1:2
                             vec_s_L_2(I1 +1,I2 +1) = vec_s_L_2(I1 +1,I2 +1) + f_change(i_x1).*f_change(i_x2).*ratio_2(i_x1,i_x2,I1,I2);
                         end
                     end
                 end
             end
             
             for I = 1:3
                 for i_x = 1:2
                     vec_s_L_2(I+1,1) = vec_s_L_2(I+1,1) + f_change(i_x).*ratio_2(i_x,   1, I, 4);
                     vec_s_L_2(1,I+1) = vec_s_L_2(1,I+1) + f_change(i_x).*ratio_2(  1, i_x, 4, I);
                 end
                 
                 %vec_s_L_2(I+1,1) = f_change(1).*ratio_2(1, 1, I, 4) + f_change(2).*ratio_2(2,   1, I, 4);
                 %vec_s_L_2(1,I+1) = f_change(1).*ratio_2(1, 1, 4, I) + f_change(2).*ratio_2(1,   2, 4, I);
        
             end
 
             squarederror_2(1,n) = squarederror_2(1,n) + norm(vec_s_L_2 - vec_s, 2)^2;
 
             % Constraint least squares estimate
             dim = 4;
             rho_CLS = sdpvar(dim, dim, 'hermitian', 'complex');
             F = [trace(rho_CLS)==1, rho_CLS >=0];

%              obj = [];
%              for I1 = 1:4
%                 for I2 = 1:4
%                     if (~(I1==1) | ~(I2==1))
%                         obj = obj + (trace(rho_CLS* kron(Pauli(I1).mat,Pauli(I2).mat)) - vec_s_L(I1,I2))^2;
%                     end% if
%                 end% I2
%              end% I1

             obj =  (trace(rho_CLS* kron(Pauli(1).mat,Pauli(2).mat)) - vec_s_L_2(1,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(1).mat,Pauli(3).mat)) - vec_s_L_2(1,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(1).mat,Pauli(4).mat)) - vec_s_L_2(1,4))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(1).mat)) - vec_s_L_2(2,1))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(2).mat)) - vec_s_L_2(2,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(3).mat)) - vec_s_L_2(2,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(2).mat,Pauli(4).mat)) - vec_s_L_2(2,4))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(1).mat)) - vec_s_L_2(3,1))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(2).mat)) - vec_s_L_2(3,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(3).mat)) - vec_s_L_2(3,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(3).mat,Pauli(4).mat)) - vec_s_L_2(3,4))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(1).mat)) - vec_s_L_2(4,1))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(2).mat)) - vec_s_L_2(4,2))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(3).mat)) - vec_s_L_2(4,3))^2 ...
                    + (trace(rho_CLS* kron(Pauli(4).mat,Pauli(4).mat)) - vec_s_L_2(4,4))^2;
               
             
             % 
             optimize(F, obj, opt);
             optobj = value(obj);
             optrho = value(rho_CLS);
% 
             vec_s_CLS_2 = zeros(4,4);
             for I1 = 1:4
                 for I2 = 1:4              
                     vec_s_CLS_2(I1,I2) = trace(rho_CLS* kron(Pauli(I1).mat, Pauli(I2).mat)); 
                 end
             end
             squarederror_2(2,n) = squarederror_2(2,n) + norm(vec_s_CLS_2 - vec_s, 2)^2;
         end
     
     end
     
     squarederror_2 = squarederror_2./num_average;
    
     ax2 = subplot(1,2,2);
     plot([1:num_unit]*15*num_per_unit, squarederror_2(1,:), [1:num_unit]*15*num_per_unit, squarederror_2(2,:))
     legend('Linear (IXYZ)', 'CLS (IXYZ)')   
     
     B = [[1:num_unit]*15*num_per_unit; squarederror_2(1,:); squarederror_2(2,:)];
     fileID_2 = fopen('output_Mixed_setting2.txt','w');
     fprintf(fileID_2, '%d %f %f\n',B);
     fclose(fileID_2);
    
%     ax3 = subplot(2,2,3);
%      plot([1:num_unit]*9*num_per_unit, squarederror_1(1,:), [1:num_unit]*15*num_per_unit, squarederror_2(1,:))
%      legend('Linear (XYZ)', 'Linear (IXYZ)')
%      
%       ax4 = subplot(2,2,4);
%      plot([1:num_unit]*9*num_per_unit, squarederror_1(2,:), [1:num_unit]*15*num_per_unit, squarederror_2(2,:))
%      legend('CLS (XYZ)', 'CLS (IXYZ)')   
%      
     
     toc
