import numpy as np
import pandas as pd


def ret2tick(df_returns, base=100):
    _, w = df_returns.shape
    df_zeros = pd.DataFrame(
        np.zeros((1, w)), columns=df_returns.columns.values)
    df_ret = pd.concat([df_zeros, df_returns], ignore_index=True)
    df_prices = base * (1 + df_ret).cumprod(axis=0)
    return df_prices


def tick2ret(df_prices):
    df_returns = df_prices.pct_change(1)
    df_returns = df_returns.iloc[1:, :]
    df_returns.reset_index(inplace=True)
    df_returns.drop("index", axis=1, inplace=True)
    return df_returns


if __name__ == "__main__":
    np.random.seed(42)
    num_assets = 10

    returns = np.random.randn(10, num_assets) / 100
    col_names = ["c" + str(r) for r in range(0, num_assets)]
    df_returns = pd.DataFrame(returns, columns=col_names)

    print(ret2tick(df_returns))

    print(tick2ret(ret2tick(df_returns)))
