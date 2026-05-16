
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/test/thermal_syncrotron.csv")

fig,ax = plt.subplots()

ax.loglog(
    df["x"],
    df["I"]
)
ax.loglog(
    df["x"],
    df["Ip_asym"]
)
print(df)

fig.savefig("fig/test/thermal_syncrotron.pdf")
