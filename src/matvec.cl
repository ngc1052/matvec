// One work-item per matrixRow, one work-group in total
kernel void matvec_v1(global const float* matrix,                                    
                             const int    size,
                      global const float* inVector,
                      global       float* outVector) 
{
    const int matrixRow = get_global_id(0);
    float sum = 0.0f;
    for(int column = 0; column < size; column++)
        sum += matrix[matrixRow * size + column] * inVector[column];
    outVector[matrixRow] = sum;
}

// Multiple work-items per matrixRow, they form a work-group
kernel void matvec_v2(global const float* matrix, 
                             const int    size,
                      global const float* inVector,
                      global       float* outVector, 
                      local        float* work) 
{
    const int matrixRow       = get_group_id  (0);             // row of matrix the item processes
    const int groupSize       = get_local_size(0);
          int localID         = get_local_id  (0);
    const int elementsPerItem = size / groupSize;              // number of rows to process in inVector
    const int firstColumn     = localID * elementsPerItem;     // first column of matrix to process
    const int lastColumn      = firstColumn + elementsPerItem; // last column of matrix to process

    float sum = 0.0f;
    for (int column = firstColumn; column < lastColumn; column++)
        sum += matrix[matrixRow * size + column] * inVector[column];

    work[localID] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0) 
    {
        work[groupSize] = 0.0f;
        for(; localID < groupSize; localID++)
            work[groupSize] += work[localID];
        outVector[matrixRow] = work[groupSize];
    }
}

#define ROW 0
#define COL 1
// Work-groups are now responsible for multiple rows, and one matrixRow is still processed by multiple work-items
kernel void matvec_v3(global const float* matrix, 
                             const int    size,
                      global const float* inVector,
                      global       float* outVector, 
                      local        float* work) 
{

    // Initializing auxiliary variables
    const int2 groupSize        = (int2)(get_local_size(ROW), get_local_size(COL));
    const int2 localID          = (int2)(get_local_id  (ROW), get_local_id  (COL));

    const int  groupID          = get_group_id(ROW);

    const int  elementsPerItem  = size / groupSize[COL];
    const int  matrixRow        = groupSize[ROW] * groupID + localID[ROW];
    const int  firstColumn      = elementsPerItem * localID[COL];
    const int  lastColumn       = firstColumn + elementsPerItem;

    // Calculatind the dot product for a part of the rows
    float sum = 0.0f;
    for(int column = firstColumn; column < lastColumn; column++)
        sum += matrix[matrixRow * size + column] * inVector[column];

    // Copying the sum into the work
    work[localID[ROW] * groupSize[COL] + localID[COL]] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // First item in the group sums up the elements in work and copies the sum to outVector
    if(localID[COL] == 0) 
    {
        const int workSumIndex = localID[ROW] * groupSize[COL]; // First column
        sum = work[workSumIndex];
        for(int localColumn = 1; localColumn < groupSize[COL]; localColumn++)
            sum += work[workSumIndex + localColumn];

        outVector[matrixRow] = sum;
    }
}