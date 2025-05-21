import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_country_stats(oecd_bli, gdp_per_capita):
    # Filter only 'Life satisfaction' and where inequality is 'Total'
    oecd_bli = oecd_bli[
        (oecd_bli["Indicator"] == "Life satisfaction")
        & (oecd_bli["Inequality"] == "Total")
    ]

    # Pivot to one row per country
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    # Clean GDP data
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    # Merge datasets
    full_country_stats = pd.merge(
        left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True
    )

    # Keep only relevant columns
    full_country_stats = full_country_stats[
        ["GDP per capita", "Life satisfaction"]
    ].dropna()

    return full_country_stats


def main():
    # Load the data
    oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=",")
    gdp_per_capita = pd.read_csv(
        "gdp_per_capita.csv",
        thousands=",",
        delimiter="\t",
        encoding="latin1",
        na_values="n/a",
    )

    # Prepare the data
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    print("✅ Final country_stats shape:", country_stats.shape)
    print(country_stats.head())

    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    # Select and train the model BEFORE using it
    model = LinearRegression()
    model.fit(X, y)

    # Visualize the data and regression line
    country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")

    # Add the regression line
    X_fit = np.linspace(
        country_stats["GDP per capita"].min(),
        country_stats["GDP per capita"].max(),
        100,
    )
    y_fit = model.predict(X_fit.reshape(-1, 1))
    plt.plot(X_fit, y_fit, color="red", linewidth=2, label="Linear Regression")
    plt.legend()
    plt.title("GDP per Capita vs Life Satisfaction")
    plt.grid(True)
    plt.show()

    # Make a prediction for Cyprus
    X_new = [[22587]]  # Cyprus' GDP per capita
    print(model.predict(X_new))  # outputs e.g., [[6.29]]


if __name__ == "__main__":
    main()

#
