matX = [0, 1; 1, 0]
matA = logm(matX)
matB = expm(matA)

matHS_X90 = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, -1, 0; 0, 0, 0, -1]
matA = logm(matHS_X90)
matB = expm(matA)