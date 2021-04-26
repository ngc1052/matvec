import os, sys
import matplotlib.pyplot as plt

dims = [2**n for n in range(4, 12)]
numRowParts = [2**n for n in range(1, 9)]

max_times = []
min_times = []
avg_times = []
with open(sys.argv[1] + '.out', 'w') as f:
    #f.write("dim;min_time[us];max_time[us];avg_time[us]\n")
    f.write("numRowParts;min_time[us];max_time[us];avg_time[us]\n")
    dim = 1024
    for n in numRowParts:
        print(n)
        times = []
        for i in range(10):
            times.append(int(os.popen("./build/matvec 1 1024 " + str(n)).read()[:-2]))
        max_times.append(max(times))
        min_times.append(min(times))
        avg_times.append(sum(times)/len(times))
        f.write(str(numRowParts) + ";" + str(min_times[-1]) + ";" + str(max_times[-1]) + ";" + str(avg_times[-1]) + "\n")

plt.plot(numRowParts, max_times, label="max_times", marker='o')
plt.plot(numRowParts, min_times, label="min_times", marker='o')
plt.plot(numRowParts, avg_times, label="avg_times", marker='o')
plt.xlabel('numRowParts')
plt.ylabel('time [us]')
plt.xticks(ticks=numRowParts)
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig(sys.argv[1] + '.png')