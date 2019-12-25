matA = [1,2,3,4;5,6,7,8;9,10,11,12;13,14,15,16]
matA(:)

vec1 = reshape(matA, [16, 1])
vec2 = reshape(matA, [1, 16])

vec3 = reshape(matA.', [16, 1])