import pandas as pd
import numpy as np

# recreate the same window dates used in the Sensex forecast notebook
# without needing to rerun Chronos-2

df = pd.read_csv("sensex_clean.csv", parse_dates=["Date"])
prices = df["Close"].values

context_len = 512
horizon = 30
step = 10

start = context_len
end = len(prices) - horizon

window_dates = []
for i in range(start, end, step):
    window_dates.append(df["Date"].iloc[i])

window_dates = np.array(window_dates)

np.save("sensex_window_dates.npy", window_dates)

print("saved sensex_window_dates.npy")
print("count:", len(window_dates))
