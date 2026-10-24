import os
import pandas as pd
from collections import OrderedDict

# Cache configuration (fixed for this prototype)
CACHE_CAPACITY = 1024  # bytes
BLOCK_SIZE = 64        # bytes
NUM_BLOCKS = CACHE_CAPACITY // BLOCK_SIZE  # Fully‑associative LRU for simplicity

def get_block_number(address: int) -> int:
    return address // BLOCK_SIZE

def simulate_lru(addresses):
    """Return a list of hits (1) / misses (0) for each address using LRU eviction."""
    cache = OrderedDict()
    hits = []
    for addr in addresses:
        block = get_block_number(addr)
        if block in cache:
            hits.append(1)  # hit
            cache.move_to_end(block)  # mark as most‑recently used
        else:
            hits.append(0)  # miss
            cache[block] = None
            if len(cache) > NUM_BLOCKS:
                # evict least‑recently used block
                cache.popitem(last=False)
    return hits

def label_trace(trace_path: str, out_csv: str):
    # Load raw addresses (hex per line)
    addresses = []
    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Allow optional 0x prefix
            if line.lower().startswith("0x"):
                addr = int(line, 16)
            else:
                addr = int(line)
            addresses.append(addr)
    hits = simulate_lru(addresses)
    # Label: 0 = hit, 1 = miss (as specified in the project description)
    labels = [0 if h else 1 for h in hits]
    df = pd.DataFrame({"address": addresses, "label": labels})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    traces_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'traces'))
    labeled_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'labeled'))
    os.makedirs(labeled_dir, exist_ok=True)
    for fname in os.listdir(traces_dir):
        if not fname.lower().endswith('.trace'):
            continue
        trace_path = os.path.join(traces_dir, fname)
        out_csv = os.path.join(labeled_dir, fname.replace('.trace', '.csv'))
        label_trace(trace_path, out_csv)
