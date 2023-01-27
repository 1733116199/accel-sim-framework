import sys
import re
import functools
import matplotlib.pyplot as plt

GIGA = 10 ** 9
FREQUENCY = 1.2 * GIGA
CYCLE = "gpu_tot_sim_cycle"
FP_COUNT = "total_fp_count"
BYTE_COUNT = "total_global_ldst_size"
STATS_FIELD = [CYCLE, FP_COUNT, BYTE_COUNT]

def read_stats(filename):
    delim = "----------------------------------------------------------------------------------------------------,"
    regex = "trace\_(.*)\.sh\/.*\-\-final_kernel,(.*)"
    result = {}
    with open(filename) as f:
        text = f.read()
        for chunk in text.split(delim):
            for s in STATS_FIELD:
                if s in chunk:
                    findResult = re.findall(regex, chunk)
                    for r in findResult:
                        if r[0] not in result:
                            result[r[0]] = {}
                        result[r[0]][s] = int(r[1])
    return result

def calc_gflops_per_sec(stats):
    for k in stats:
        stats[k]["gflops_per_sec"] = stats[k][FP_COUNT] / (stats[k][CYCLE] / FREQUENCY) / GIGA

def calc_flops_per_byte(stats):
    for k in stats:
        stats[k]["flops_per_byte"] = stats[k][FP_COUNT] / (stats[k][BYTE_COUNT])

def calc_gb_per_sec(stats):
    for k in stats:
        stats[k]["gb_per_sec"] = stats[k][BYTE_COUNT] / (stats[k][CYCLE] / FREQUENCY) / GIGA

def frange(start, stop, step=1.0):
    f = start
    while f < stop:
        f += step
        yield f
         
if __name__ == '__main__':
    assert(len(sys.argv) <= 2)
    filename = sys.argv[1]
    
    # read and process stats
    stats = read_stats(filename)
    calc_gflops_per_sec(stats)
    calc_flops_per_byte(stats)
    calc_gb_per_sec(stats)
    
    # print stats
    print(stats)
    
    # plot data points
    labels = [k for k in stats]
    x = [stats[k]["flops_per_byte"] for k in stats]
    y = [stats[k]["gflops_per_sec"] for k in stats]
    figure = plt.figure()
    ax = figure.add_subplot()
    fig = ax.scatter(x, y)
    ax.set_ylabel("Performance (FLOPs/second)")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_title("DNN Models on TITAN V")
    for i, l in enumerate(labels):
        ax.annotate(l, (x[i], y[i] - 500))
    
    if True:
        # plot roofline
        bw = 651.3
        max_flops = 14900
        xticks = [2.**i for i in range(-4, 10)]
        x = list(frange(min(xticks), max(xticks), 0.01))
        ax.plot(x, [min(bw*x, float(max_flops)) for x in x])
    
    ax.set_xlim(0, 30)
    
    # save figure
    plt.savefig('roofline.png')
    
    
    