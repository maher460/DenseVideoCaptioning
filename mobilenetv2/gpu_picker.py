import subprocess, re, os, sys
import argparse

# GPU picking
# Code is from: http://stackoverflow.com/a/41638727/419116
# Almost all credits for this code goes to the above link
# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def setup_one_gpu():
    # assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    gpu_id = pick_gpu_lowest_memory()
    # print("Picking GPU "+str(gpu_id))
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def pick_gpus_lowest_memory(num_gpus):
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    list_mem_gpu = sorted(memory_gpu_map)[0:num_gpus]
    best_gpus = map(lambda x: str(x[1]), list_mem_gpu)
    return best_gpus

def setup_multiple_gpus(num_gpus):
    # assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    gpu_ids = pick_gpus_lowest_memory(num_gpus)
    # print("Picking GPU "+str(gpu_ids))
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

def setup_no_gpu():
    if 'tensorflow' in sys.modules:
        print("Warning, GPU setup must happen before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

if __name__ == '__main__':

    '''
    We first need to parse the arguments
    '''
    parser = argparse.ArgumentParser()

    # parse the path to weights
    parser.add_argument("-n", "--numgpus",
                        help="optional specifier for number of gpus")

    args = parser.parse_args()

    if args.numgpus:
        setup_multiple_gpus(int(args.numgpus))
    else:
        setup_one_gpu()

    print("[MaherBot] GPU selected: " + os.environ["CUDA_VISIBLE_DEVICES"])