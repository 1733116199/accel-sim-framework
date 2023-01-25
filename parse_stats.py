import sys
import re
import functools

def read_stats(filename):
    delim = "----------------------------------------------------------------------------------------------------,"
    stats = ["L1D_total_cache_accesses", "gpu_tot_sim_cycle", "L1D_total_cache_misses", "gpu_tot_sim_insn"]
    types = ["conv", "batchnorm", "relu"]
    prefix = "trace\.sh\/data\_dirs\_base\_pt\_"
    suffix = "\_.*\-\-final_kernel,(.*)"
    with open(filename) as f:
        text = f.read()
        for chunk in text.split(delim):
            for s in stats:
                if s in chunk:
                    for t in types:
                        findResult = re.findall(prefix + t + suffix, chunk)
                        res = functools.reduce((lambda a, b: int(a) + int(b)), findResult)
                        print(f"{s},{t},{res}")
if __name__ == '__main__':
    assert(len(sys.argv) <= 2)
    baseline = sys.argv[1]
    print("field,bench,value")
    read_stats(baseline)
    