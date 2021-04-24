// One work-item, one row, one work-group in total
__kernel void matvec(__global float* mat, int size, __global float* inVector, __global float* outVector)
{
    int row = get_global_id(0);
    outVector[row] = 0.0;
    for(int column = 0; column < size; column++)
        outVector[row] += mat[row*size + column]*inVector[column];
}