import numpy as np

def sliding_windows(data, lookback_len, predict_len):
    x = []
    y = []

    for i in range(lookback_len, len(data)):
        _x = data[i-lookback_len:i]
        _y = data[i:i+predict_len]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)
    