state = [0.5,0.5i; -0.5i, 0.5]
povm(1).mat = [1,0; 0,0]
povm(2).mat = [0, 0; 0, 1]
Choi = [1, 0, 0, 1; 0, 0, 0, 0; 0, 0, 0, 0; 1, 0, 0, 1]

probDist = ProbDist_QPT(Choi, state, povm)
