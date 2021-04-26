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
    int groupSize   = get_local_size(0);       // number of work-items the row is divided into
    int innerSize   = size/groupSize;          // number of rows (columns) to process in inVector (mat)
    int workItemID  = get_local_id(0);
    int firstColumn = workItemID*innerSize;        // first row (column) of inVector (mat) to process
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
#define ROW 0
#define COL 1

// Work-groups are now responsible for multiple rows, and one row is still processed by multiple work-items
__kernel void matvec_v3(
    __global const float* mat, 
             const int    size, 
    __global const float* inVector, 
    __global       float* outVector, 
    __local        float* workArray)
{
    int2 groupSize;
    groupSize[ROW] = get_local_size(ROW);
    groupSize[COL] = get_local_size(COL);

    int groupID = get_group_id(ROW);

    int2 localID;
    localID[ROW] = get_local_id(ROW);
    localID[COL] = get_local_id(COL);

    int columnsPerItem = size / groupSize[COL];
    int row = groupSize[ROW] * groupID + localID[ROW];
    int firstColumn = columnsPerItem * localID[COL];
    int lastColumn = firstColumn + columnsPerItem;

    float result = 0.0f;
    for(int column = firstColumn; column < lastColumn; column++)
        result += mat[row*size + column]*inVector[column];

    int2 workArraySize;
    workArraySize[COL] = groupSize[COL] + 1;
    workArraySize[ROW] = groupSize[ROW];

    workArray[localID[ROW] * workArraySize[COL] + localID[COL]] = result;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(localID[COL] == 0)
    {
        int workArraySumIndex = localID[ROW]*workArraySize[COL] + workArraySize[COL] - 1;
        workArray[workArraySumIndex] = 0.0f;
        for(int localColumn = 0; localColumn < groupSize[COL]; localColumn++)
            workArray[workArraySumIndex] += workArray[localID[ROW]*workArraySize[COL] + localColumn];

        outVector[row] = workArray[workArraySumIndex];
    }
}
