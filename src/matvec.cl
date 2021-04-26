// One work-item per row, one work-group in total
kernel void matvec_v1(global const float* mat,                                    
                             const int   size,
                      global const float* inVector,
                      global       float* outVector) 
{
    const int row = get_global_id(0);
    float result = 0.0;
    for(int column = 0; column < size; column++)
        result += mat[row * size + column] * inVector[column];
    outVector[row] = result;
}

// Multiple work-items per row, they form a work-group
kernel void matvec_v2(global const float* mat, 
                             const int    size,
                      global const float* inVector,
                      global       float* outVector, 
                      local        float* workArray) 
{
    const int row         = get_group_id  (0);         // row of outVector to calculate
    const int groupSize   = get_local_size(0);
          int localID     = get_local_id  (0);
    const int itemRowSize = size / groupSize;          // number of rows to process in inVector
    const int firstColumn = localID * itemRowSize;     // first column of mat to process
    const int lastColumn  = firstColumn + itemRowSize; // last column of mat to process

    float result = 0.0f;
    for (int column = firstColumn; column < lastColumn; column++)
        result += mat[row * size + column] * inVector[column];

    workArray[localID] = result;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0) 
    {
        workArray[groupSize] = 0.0f;
        for(; localID < groupSize; localID++)
            workArray[groupSize] += workArray[localID];
        outVector[row] = workArray[groupSize];
    }
}

#define ROW 0
#define COL 1
// Work-groups are now responsible for multiple rows, and one row is still processed by multiple work-items
kernel void matvec_v3(global const float* mat, 
                             const int    size,
                      global const float* inVector,
                      global       float* outVector, 
                      local        float* workArray) 
{

    // Initializing auxiliary variables
    const int2 groupSize    = (int2)(get_local_size(ROW), get_local_size(COL));
    const int2 localID      = (int2)(get_local_id  (ROW), get_local_id  (COL));

    const int  groupID      = get_group_id(ROW);

    const int  itemRowSize  = size / groupSize[COL];
    const int  row          = groupSize[ROW] * groupID + localID[ROW];
    const int  firstColumn  = itemRowSize * localID[COL];
    const int  lastColumn   = firstColumn + itemRowSize;

    // Calculatind the dot product for a part of the rows
    float result = 0.0f;
    for(int column = firstColumn; column < lastColumn; column++)
        result += mat[row * size + column] * inVector[column];

    // Copying the result into the workArray
    workArray[localID[ROW] * groupSize[COL] + localID[COL]] = result;
    barrier(CLK_LOCAL_MEM_FENCE);

    // First item in the group sums up the elements in workArray and copies the result to outVector
    if(localID[COL] == 0) 
    {
        const int workArraySumIndex = localID[ROW] * groupSize[COL]; // First column
        for(int localColumn = 1; localColumn < groupSize[COL]; localColumn++)
            workArray[workArraySumIndex] += workArray[workArraySumIndex + localColumn];

        outVector[row] = workArray[workArraySumIndex];
    }
}
