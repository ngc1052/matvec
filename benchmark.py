import os, sys
import matplotlib.pyplot as plt

dims = [10, 20, 50, 100, 200, 500, 1000, 5000, 10000]
max_times = []
min_times = []
avg_times = []
with open(sys.argv[1] + '.out', 'w') as f:
    f.write("dim;min_time[us];max_time[us];avg_time[us]\n")
    for dim in dims:
        print(dim)
        times = []
        for i in range(10):
            times.append(int(os.popen("./build/matvec " + str(dim)).read()[:-2]))
        max_times.append(max(times))
        min_times.append(min(times))
        avg_times.append(sum(times)/len(times))
        f.write(str(dim) + ";" + str(min_times[-1]) + ";" + str(max_times[-1]) + ";" + str(avg_times[-1]) + "\n")

plt.plot(dims, max_times, label="max_times", marker='o')
plt.plot(dims, min_times, label="min_times", marker='o')
plt.plot(dims, avg_times, label="avg_times", marker='o')
plt.xlabel('dims')
plt.ylabel('time [us]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(sys.argv[1] + '.png')