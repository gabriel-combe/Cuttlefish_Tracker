import numpy as np

def systematic_resample(weights: np.ndarray) -> np.ndarray:
    N = len(weights)

    cumsum = np.cumsum(weights)
    indexes = np.zeros(N, 'i')
    
    for i in range(N):
        indexes[i] = np.searchsorted(cumsum, np.random.random())
    
    return indexes

def residual_resample(weights: np.ndarray) -> np.ndarray:
    N = len(weights)

    indexes = np.zeros(N, 'i')
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)

    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):
            indexes[k] = i
            k += 1

    residual = weights - num_copies
    residual /= sum(residual)
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.
    indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))

    return indexes

def stratified_resample(weights: np.ndarray) -> np.ndarray:
    N = len(weights)

    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes

def multinomial_resample(weights: np.ndarray) -> np.ndarray:
    cumulaive_sum = np.cumsum(weights)
    cumulaive_sum[-1] = 1.0
    return np.searchsorted(cumulaive_sum, np.random.random(len(weights)))