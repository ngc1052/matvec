#define ROW 0
#define COL 1

float dotProduct(const global float* vec1, const global float* vec2, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += vec1[i] * vec2[i];
    return sum;
}

float dotProductFromLocal(const global float* vec1, const local float* vec2, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += vec1[i] * vec2[i];
    return sum;
}

// One work-item per matrixRow, one work-group in total
/*
Problem: Each element of inVector is read at every row, and it is stored in the global memory.
*/
kernel void matvec_v1(global const float* matrix,                                    
                             const int    size,
                      global const float* inVector,
                      global       float* outVector) 
{
    const int matrixRow = get_global_id(ROW);
    outVector[matrixRow] = dotProduct(matrix + (matrixRow * size), inVector, size);
}

// Multiple work-items per matrixRow, they form a work-group
/*

/              \ 
|              | 
|--------------| 
|  |  |  |  |  | 
|--------------| 
|              | 
\              / 

Here we introduce the work array that is stored in the local memory. Each item
calculates its own part of the dot product and stores the result in the work
array. After synchronization, the elements of the work array need to be summed.
This is done by choosing a single item, this is slow as it could be
paralellized.
*/
kernel void matvec_v2(global const float* matrix, 
                             const int    size,
                      global const float* inVector,
                      global       float* outVector, 
                      local        float* work) 
{
    const int matrixRow       = get_group_id  (ROW);            // row of matrix the item processes
    const int groupSize       = get_local_size(ROW);
          int localID         = get_local_id  (ROW);
    const int elementsPerItem = size / groupSize;              // number of rows to process in inVector
    const int firstColumn     = localID * elementsPerItem;     // first column of matrix to process

    global const float* shiftedMatrix   = matrix + matrixRow * size + firstColumn;
    global const float* shiftedInVector = inVector + firstColumn;
    work[localID] = dotProduct(shiftedMatrix, shiftedInVector, elementsPerItem);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Naive reduction
    if(localID == 0) 
    {
        work[groupSize] = 0.0f;
        for(; localID < groupSize; localID++)
            work[groupSize] += work[localID];
        outVector[matrixRow] = work[groupSize];
    }
}


// Work-groups are now responsible for multiple rows, and one matrixRow is still processed by multiple work-items
/*

/              \ 
|--------------| 
|  |  |  |  |  | 
|--------------| 
|  |  |  |  |  | 
|--------------| 
\              / 
We now use a 2D computational grid. The principle is the same as before, but
here we extend the scope of a work-group, therefore, the work array is now two
dimensional. Its elements in a row are still summed up by a single item.
*/
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
    const int  matrixRow        = get_global_id(ROW);
    const int  firstColumn      = elementsPerItem * localID[COL];

    // Calculatind the dot product for a part of the rows
    global const float* shiftedMatrix   = matrix + matrixRow * size + firstColumn;
    global const float* shiftedInVector = inVector + firstColumn;
    work[localID[ROW] * groupSize[COL] + localID[COL]] = dotProduct(shiftedMatrix, shiftedInVector, elementsPerItem);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Naive reduction
    if(localID[COL] == 0) 
    {
        const int workSumIndex = localID[ROW] * groupSize[COL]; // First column
        float sum = work[workSumIndex];
        for(int localColumn = 1; localColumn < groupSize[COL]; localColumn++)
            sum += work[workSumIndex + localColumn];

        outVector[matrixRow] = sum;
    }
}

// Work-groups do not process one complete row but have a limited size
/*

/      ----    \ 
|      |  |    | 
|      ----    | 
|      |  |    | 
|      ----    | 
|              | 
\              / 
Here we avoid reading the same data from global memory multiple times. From now
on, the work array is not used to store partial dot products, but to store the
relevant part of inVector that is read multiple times. The dot products are
therefore calculated by reading the vector from local memory. Given its limited
size, work-group now cannot compute full rows of the matrix, they have to be
restrictied. Partial results are stored in the global memory, and a separate
kernel is called to reduce its rows. Further optimization is now done by
refining the reduction algorithm.
*/
kernel void matvec_v4(global const float* matrix, 
                             const int    size,
                      global const float* inVector,
                      global       float* extendedOutVector, 
                      local        float* work) 
{

    // Initializing auxiliary variables
    const int2 groupSize        = (int2)(get_local_size (ROW), get_local_size (COL));
    const int2 globalSize       = (int2)(get_global_size(ROW), get_global_size(COL));
    const int  localID          = get_local_id(ROW);

    const int2 groupID          = (int2)(get_group_id(ROW), get_group_id(COL));

    const int  elementsPerItem  = size / globalSize[COL];
    const int  matrixRow        = get_global_id(ROW);
    const int  firstColumn      = elementsPerItem * groupID[COL];

    // Copying part of inVector into local memory
    const int readByOneItem = elementsPerItem / groupSize[ROW];
    const int startIndex = readByOneItem * localID;
    for(int workIndex = startIndex; workIndex < startIndex + readByOneItem; workIndex++)
    {
        const int vectorIndex = workIndex + firstColumn;
        work[workIndex] = inVector[vectorIndex];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculating the dot product for a part of the rows
    global const float* shiftedMatrix  = matrix + matrixRow * size + firstColumn;
    extendedOutVector[matrixRow*globalSize[COL] + groupID[COL]] = dotProductFromLocal(shiftedMatrix, work, elementsPerItem);
}

kernel void reduceRows(global float* extendedOutVector, const int columns)
{
    //Naive reudction
    const int sumIndex = get_global_id(ROW) * columns;
    for(int column = 1; column < columns; column++)
        extendedOutVector[sumIndex] += extendedOutVector[sumIndex + column];
}