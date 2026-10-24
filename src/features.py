import pandas as pd
from collections import deque

BLOCK_SIZE = 64
RECENT_WINDOW = 8  # Number of recent accesses to consider for recent metrics

def compute_block_number(df: pd.DataFrame) -> pd.DataFrame:
    df['block_number'] = df['address'] // BLOCK_SIZE
    return df

def compute_stride_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df['prev_block'] = df['block_number'].shift(1)
    df['stride_blocks'] = df['block_number'] - df['prev_block']
    df['stride_blocks'].fillna(0, inplace=True)
    df.drop(columns=['prev_block'], inplace=True)
    return df

def compute_reuse_distance(df: pd.DataFrame) -> pd.DataFrame:
    # Naïve O(n²) implementation – fine for modest trace sizes.
    reuse_dist = []
    last_occurrence = {}
    for idx, block in enumerate(df['block_number']):
        if block in last_occurrence:
            start = last_occurrence[block] + 1
            distinct = df['block_number'].iloc[start:idx].nunique()
            reuse_dist.append(distinct)
        else:
            reuse_dist.append(-1)  # sentinel for unseen block
        last_occurrence[block] = idx
    df['reuse_distance'] = reuse_dist
    return df

def compute_recent_metrics(df: pd.DataFrame, window: int = RECENT_WINDOW) -> pd.DataFrame:
    recent_unique = []
    recent_same = []
    recent_window = deque(maxlen=window)
    for block in df['block_number']:
        # Number of distinct blocks in the recent window (excluding current)
        recent_unique.append(len(set(recent_window)))
        # Whether the current block appeared in the recent window
        recent_same.append(1 if block in recent_window else 0)
        recent_window.append(block)
    df['recent_unique_blocks'] = recent_unique
    df['recent_same_block'] = recent_same
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_block_number(df)
    df = compute_stride_blocks(df)
    df = compute_reuse_distance(df)
    df = compute_recent_metrics(df)
    df.fillna(0, inplace=True)
    return df
