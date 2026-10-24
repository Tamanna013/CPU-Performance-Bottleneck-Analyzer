import os
import numpy as np

TRACES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'traces'))

def write_trace(file_path, addresses):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, addresses, fmt='0x%08x')

def generate_sequential(num=100, start=0x0, stride=0x40):
    addresses = np.arange(start, start + num * stride, stride, dtype=np.uint64)
    write_trace(os.path.join(TRACES_DIR, 'sequential.trace'), addresses)

def generate_repeated(num=100, address=0x1000):
    addresses = np.full(num, address, dtype=np.uint64)
    write_trace(os.path.join(TRACES_DIR, 'repeated.trace'), addresses)

def generate_random(num=100, low=0x0, high=0x10000, seed=0):
    rng = np.random.default_rng(seed)
    addresses = rng.integers(low, high, size=num, dtype=np.uint64)
    write_trace(os.path.join(TRACES_DIR, 'random.trace'), addresses)

def generate_mixed(num=100, seed=0):
    rng = np.random.default_rng(seed)
    # Mix sequential, repeated, and random patterns
    seq_len = num // 3
    rep_len = num // 3
    rand_len = num - seq_len - rep_len
    seq = np.arange(0x0, 0x0 + seq_len * 0x40, 0x40, dtype=np.uint64)
    rep = np.full(rep_len, 0x2000, dtype=np.uint64)
    rand = rng.integers(0x0, 0x10000, size=rand_len, dtype=np.uint64)
    addresses = np.concatenate([seq, rep, rand])
    rng.shuffle(addresses)
    write_trace(os.path.join(TRACES_DIR, 'mixed.trace'), addresses)

if __name__ == "__main__":
    os.makedirs(TRACES_DIR, exist_ok=True)
    generate_sequential()
    generate_repeated()
    generate_random()
    generate_mixed()
