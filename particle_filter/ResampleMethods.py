import numpy as np

def systematic_resample(weights: np.ndarray) -> np.ndarray:
    N = len(weights)

    offsets = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(np.asarray(weights) / np.sum(weights))
    i, j = 0, 0

    while i < N:
        if offsets[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else: 
            j += 1
    
    return indexes

def systematic_resamplev2(weights: np.ndarray) -> np.ndarray:
    N = len(weights)

    cumsum = np.cumsum(weights)
    indexes = np.zeros(N, 'i')
    
    for i in range(N):
        indexes[i] = np.searchsorted(cumsum, np.random.random())
    
    return indexes