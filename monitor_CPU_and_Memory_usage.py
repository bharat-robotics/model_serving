import os
import signal
import subprocess
import time

# 1. Start monitoring by using "docker stats ..."
# Retrive percentages of CPU and Memory usages of all running containers
# Assume that each running container is a model server.
# Otherwise, model servers' container IDs should be identified first.
cmd = 'docker stats --all --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}"'
pro = subprocess.Popen("exec "+cmd, stdout=subprocess.PIPE,
                       shell=True)

# 2. [TBD] need to identify starting and ending points of a window for
# measuring the energy consumption in the duration.
#start=time.time()
#end=time.time()
duration=10 # For debugging, set the testing window to be about 10 seconds
time.sleep(duration)
pro.kill()


# 3. Process each line of the stdout of "docker stats ..."
# to compute averaged cpu and memory usage percents in the
# duration by firstly accumulating usage percents in each
# sampled result and then divding by the total number of samples.
cpu_power=65 #watt. check the power of your CPU
mem_power=1.485 #watt. check the power of your memory
total_cpu_usage_percent=0.0
total_mem_usage_percent=0.0

lines = pro.stdout.readlines()
cnt=0
for line in lines:
    line=line.strip()
    if "CONTAINER" not in line:
        cnt+=1
        print(line)
        parts=line.split()
        total_cpu_usage_percent+=float(parts[1].strip('%'))
        total_mem_usage_percent+=float(parts[2].strip('%'))

ave_cpu_usage_percent=total_cpu_usage_percent/cnt
ave_mem_usage_percent=total_mem_usage_percent/cnt

# 4. Calculate the energy consumption of cpu and memory in the duration
total_cpu_energy_consumption=cpu_power*duration*ave_cpu_usage_percent
total_mem_energy_consumption=mem_power*duration*ave_mem_usage_percent
print("Duration: {} seconds.\nCPU usage percent/energy cost: {}/{}.\nMemory usage percent/energy cost: {}/{}.".
        format(
            duration,
            ave_cpu_usage_percent,
            total_cpu_energy_consumption,
            ave_mem_usage_percent,
            total_mem_energy_consumption))
