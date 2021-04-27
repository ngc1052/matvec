import os, sys
import matplotlib.pyplot as plt

dim = 256
repeat_count = 10

def benchmark_method_v1():
    times = []
    for i in range(repeat_count):
        return_string = os.popen("./build/matvec 1 " + str(dim) + " --profile").read()
        time = int(return_string.split(' ')[2])
        times.append(time)
    return times

def measure_v1(dim):
    return_string = os.popen("./build/matvec 1 " + str(dim) + " --profile").read()
    time = int(return_string.split(' ')[2])
    return time

def find_fastest_param_for_v2(dim):
    rowBlockSizes = [2**n for n in range(1, 7)]
    times = []
    min_params = 0
    min_time = 0

    for rowBlockSize in rowBlockSizes:
        return_string = os.popen("./build/matvec 2 " + str(dim) + " " + str(rowBlockSize) + " --profile").read()
        time = int(return_string.split(' ')[2])
        times.append(time)
        if min_time == 0 or min_time > time:
            min_time = time
            min_params = rowBlockSize
    return min_time, min_params

def find_fastest_param_for_v3(dim):
    rowBlockSizes = [2**n for n in range(1, 7)]
    colBlockSizes = [2**n for n in range(1, 7)]
    times = []
    min_params = 0
    min_time = 0

    for rowBlockSize in rowBlockSizes:
        for colBlockSize in colBlockSizes:
            if rowBlockSize*colBlockSize < 256:
                return_string = os.popen("./build/matvec 3 " + str(dim) + " " + str(rowBlockSize) + " " + str(colBlockSize) + " --profile").read()
                time = int(return_string.split(' ')[2])
                times.append(time)
                if min_time == 0 or min_time > time:
                    min_time = time
                    min_params = [rowBlockSize, colBlockSize]
        return min_time, min_params

print("v1:")
print(measure_v1(dim))

print("v2:")
print(find_fastest_param_for_v2(dim))

print("v3:")
print(find_fastest_param_for_v3(dim))