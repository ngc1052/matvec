import os, sys
import matplotlib.pyplot as plt

dim = 1024
repeat_count = 10
launch_cmd = "./build/matvec"

def measure_time(version, dim, arg1 = None, arg2 = None):
    return_string = os.popen(f"{launch_cmd} {version} {dim} {'' if arg1 == None else arg1} {'' if arg1 == None else arg2} --profile").read()
    time = int(return_string.split(' ')[2])
    return time

def benchmark_cpu_version():
    times = []
    for i in range(repeat_count):
        times.append(measure_time(0, dim))
    return min(times)

def benchmark_method_v1():
    times = []
    for i in range(repeat_count):
        times.append(measure_time(1, dim))
    return min(times)

def find_fastest_param_for_v2():
    numItemsInGroupRow = [2**n for n in range(1, 7)]
    times = []
    min_param = 0
    min_time = 0

    for p in numItemsInGroupRow:
        time = measure_time(2, dim, p)
        times.append(time)
        if min_time == 0 or min_time > time:
            min_time = time
            min_param = p
    return min_time, min_param#, times

def find_fastest_param_for_v3():
    numItemsInGroupRow = [2**n for n in range(1, 7)]
    numItemsInGroupColumn = [2**n for n in range(1, 7)]
    times = []
    min_params = 0
    min_time = 0

    for p1 in numItemsInGroupRow:
        for p2 in numItemsInGroupColumn:
            if p1*p2 < 256:
                time = measure_time(3, dim, p1, p2)
                times.append(time)
                if min_time == 0 or min_time > time:
                    min_time = time
                    min_params = [p1, p2]
        return min_time, min_params#, times

def find_fastest_param_for_v4():
    numItemsInGroupColumn = [2**n for n in range(1, 8)]
    numItemsInMatrixRow = [2**n for n in range(1, 8)]
    times = []
    min_params = 0
    min_time = 0

    for p1 in numItemsInGroupColumn:
        for p2 in numItemsInMatrixRow:
            if dim > p1 * p2:
                time = measure_time(4, dim, p1, p2)
                times.append(time)
                if min_time == 0 or min_time > time:
                    min_time = time
                    min_params = [p1, p2]
        return min_time, min_params#, times

print("Times are measured in microseconds.")
print(f"Dimension: {dim}\n")

print("CPU version: ")
print(benchmark_cpu_version())

print("\nv1:")
print(benchmark_method_v1())

print("\nv2 (time, parameter):")
print(find_fastest_param_for_v2())

print("\nv3 (time, parameters):")
print(find_fastest_param_for_v3())

print("\nv4 (time, parameters):")
print(find_fastest_param_for_v4())