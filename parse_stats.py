import sys
import re
import functools
import matplotlib.pyplot as plt

GIGA = 10 ** 9
FREQUENCY = 1.2 * GIGA
CYCLE = "gpu_tot_sim_cycle"
FP_COUNT = "total_fp_count"
BYTE_COUNT = "offchip_total_bytes"
L1_ACCESSES = "L1D_total_cache_accesses"
L1_MISSES = "L1D_total_cache_misses"
STATS_FIELD = [CYCLE, FP_COUNT, BYTE_COUNT, L1_ACCESSES, L1_MISSES]

def read_stats(filename, label=0, regex=None):
    delim = "----------------------------------------------------------------------------------------------------,"
    regex = "trace\_(.*)\.sh\/(.*)\-\-final_kernel,(.*)" if regex is None else regex
    result = {}
    with open(filename) as f:
        text = f.read()
        for chunk in text.split(delim):
            for s in STATS_FIELD:
                if s in chunk:
                    findResult = re.findall(regex, chunk)
                    for r in findResult:
                        if label == 0:
                            field = r[0]
                        elif label == 1:
                            field = r[1]
                        else:
                            field = r[0] + "/" + r[1]
                        if field not in result:
                            result[field] = {}
                        result[field][s] = int(r[2])
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

def roofline(filename):
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
    ax.set_ylabel("Performance (GFLOPs/second)")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_title("DNN Models on TITAN V")
    for i, l in enumerate(labels):
        ax.annotate(l, (x[i], y[i] - 500))
    
    if True:
        # plot roofline
        bw = 652.8
        max_flops = 14900
        xticks = [2.**i for i in range(-4, 10)]
        xticks = list(frange(min(xticks), max(xticks), 0.01))
        ax.plot(xticks, [min(bw*xt, float(max_flops)) for xt in xticks])
    
    ax.set_xlim(0, max(x) * 1.2)
    
    # save figure
    plt.savefig('roofline.png')

def calc_miss_rate(stats):
    for k in stats:
        stats[k]["l1_miss_rate"] = stats[k][L1_MISSES] / stats[k][L1_ACCESSES]

def speedup(before, after):
    sb = read_stats(before)
    sa = read_stats(after)
    calc_miss_rate(sb)
    calc_miss_rate(sa)
    print("workload,speedup(%),L1 miss rate (before %),L1 miss rate (after %)")
    for k in sa:
        assert(k in sa)
        improvement = round((sb[k][CYCLE] / sa[k][CYCLE] - 1) * 100, 2)
        l1_miss_before = round(sb[k]["l1_miss_rate"] * 100, 3)
        l1_miss_after = round(sa[k]["l1_miss_rate"] * 100, 3)
        print(f"{k},{improvement},{l1_miss_before},{l1_miss_after}")

if __name__ == '__main__':
    assert(len(sys.argv) >= 3)
    operation = sys.argv[1]
    if operation == "roofline":
        roofline(sys.argv[2])
    elif operation == "speedup":
        assert(len(sys.argv) >= 4)
        speedup(sys.argv[2], sys.argv[3])
    else:
        regex = "(measure\_max\_flops)\/(.*)\-\-final_kernel,(.*)"
        option = 0
        stats = read_stats(sys.argv[2], option, regex)
        calc_gflops_per_sec(stats)
        calc_flops_per_byte(stats)
        calc_gb_per_sec(stats)
        
        # print stats
        print(stats)
    

    
    
    