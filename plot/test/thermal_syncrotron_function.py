
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/test/thermal_syncrotron.csv")

fig,ax = plt.subplots()
fig.set_layout_engine("constrained")

def plot_main(ax):
    ax.loglog(
        df["x"],
        df["Ip"],
        color = "#000000",
        ls="-",
        label="$I'(x)$"
    )
    ax.loglog(
        df["x"],
        df["Ip_asym"],
        color = "#FF4B00",
        ls="--",
        label="$I'(x)$ (Mahadevan+96)"
    )
    labelfs=16
    ax.set_xlabel(r"$X_{\Theta}$",fontsize=labelfs)
    ax.set_ylabel(r"$I'(X_{\Theta})$",fontsize=labelfs)
    ax.legend()

def plot_error(ax):
    ax.loglog(
        df["x"],
        df["Ip_asym"]/df["Ip"]-1,
        color = "#FF4B00",
        ls="--",
    )
    ax.axhline(0.0)

plot_main(ax)
fig.savefig("fig/test/thermal_syncrotron.pdf")
