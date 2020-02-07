function probDist = ProbDist_QPT( Choi, state, povm )
%PROBDIST_QPT 
    num_outcome = size(povm,2);
    
    for i_omega = 1:num_outcome
        povm_element = povm(i_omega).mat;
        p = Prob_QPT(Choi, state, povm_element);
        probDist(i_omega) = p;
    end

end

