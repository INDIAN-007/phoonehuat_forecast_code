# Croston's Method - https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f
def Croston_TSB(ts, extra_periods=5, alpha=0.5, beta=0.7):
    import numpy as np
    import pandas as pd
    # Transform the input into a numpy array
    d = np.array(ts)
    # Historical period length
    cols = len(d)
    # Append np.nan into the demand array to cover future periods
    d = np.append(d, [np.nan] * extra_periods)

    # level (a), probability (p) and forecast (f)
    a, p, f = np.full((3, cols + extra_periods), np.nan)

    # Initialization (1st occurance of demand)
    # returns index of element where demand>0

    first_occurence = np.argmax(d[:cols] > 0)
    # first demand
    a[0] = d[first_occurence]
    # percentage of demand period
    p[0] = 1 / (1 + first_occurence)
    f[0] = p[0] * a[0]

    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * (1) + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]

    # Future Forecast
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]

    df = pd.DataFrame.from_dict({"Demand": d, "Forecast": f, "Period": p, "Level": a, "Error": d - f})
    return df