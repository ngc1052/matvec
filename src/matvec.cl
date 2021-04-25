// One work-item per row, one work-group in total
__kernel void matvec_v1(
    __global const float* mat, 
             const int    size, 
    __global const float* inVector, 
    __global       float* outVector)
{
    int row = get_global_id(0);
    float result = 0.0;
    for(int column = 0; column < size; column++)
        result += mat[row*size + column]*inVector[column];
    outVector[row] = result;
}

// Multiple work-items per row, they form a work-group
__kernel void matvec_v2(
    __global const float* mat, 
             const int    size, 
    __global const float* inVector, 
    __global       float* outVector, 
    __local        float* rowPart) // groupSize+1 floats
{
    int row         = get_group_id(0);         // row of outVector to calculate
    int groupSize   = get_local_size(0)        // number of work-items the row is divided into
    int innerSize   = size/groupSize;          // number of rows (columns) to process in inVector (mat)
    int workItemID  = get_local_id(0);
    int firstColumn = partID*innerSize;        // first row (column) of inVector (mat) to process
    int lastColumn  = firstColumn + innerSize; // last row (column) of inVector (mat) to process

    float result = 0.0f;
    for(int column = firstColumn; column < lastColumn; column++)
        result += mat[row*size + column]*inVector[column];

    rowPart[workItemID] = result;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(workItemID == 0)
    {
        rowPart[groupSize] = 0.0f;
        for(; workItemID < groupSize; workItemID++)
            rowPart[groupSize] += rowPart[workItemID];
        outVector[row] = rowPart[groupSize];
    }
}