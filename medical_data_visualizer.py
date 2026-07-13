import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("medical_examination.csv")

# Add overweight column
bmi = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = (bmi > 25).astype(int)

# Normalize cholesterol and glucose
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


def draw_cat_plot():
    # Convert to long format
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight",
        ],
    )

    # Count each category
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # Draw plot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
    )

    return fig.fig


def draw_heat_map():
    # Clean data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Correlation matrix
    corr = df_heat.corr()

    # Mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        linewidths=0.5,
        center=0,
        ax=ax,
    )

    return fig