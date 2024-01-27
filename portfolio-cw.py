import pandas as pd
import yfinance as yf
import numpy as np
import decimal as dec
import matplotlib.pyplot as plt
from utils_portfolio import tick2ret, ret2tick
import streamlit as st

run = False
filename = "market_cap.csv"
if run:
    sp500 = pd.read_excel("SP500-CapWeight-Ref.xlsx")
    symbols = sp500["Symbol"]

    market_cap = dict()

    for symbol in symbols:
        data = yf.Ticker(symbol)
        try:
            mrk_cap = data.info['marketCap']
            market_cap[symbol] = mrk_cap
        except:
            market_cap[symbol] = "N/A"

    market_cap = pd.DataFrame.from_dict(market_cap, orient='index')
    market_cap.to_csv(filename)

else:
    market_cap = pd.read_csv(filename, index_col="Symbol")


dec_market_cap = market_cap.apply(
    lambda x: dec.Decimal(float(x.values)), axis=1)
total_market_cap = np.sum(dec_market_cap)
print(total_market_cap)


cap_weights = dec_market_cap.apply(lambda x: x/total_market_cap)
cap_weights.to_csv("cap_weights.csv")
cap_weights.sort_values(ascending=False, inplace=True)

sp500 = pd.read_excel("SP500-CapWeight-Ref.xlsx")
sp500.drop(["Portfolio%"], axis=1, inplace=True)
sp500


# Top 10
top_10 = cap_weights[0:10].to_frame()
top_10.rename({0: 'Weight'},  axis=1, inplace=True)

for t in top_10.index:
    name = sp500.loc[sp500['Symbol'] == t, "Company"].values[0]
    top_10.loc[t, "Name"] = name

# fig, ax = plt.subplots()
# ax.pie(top_10['Weight'], labels=top_10['Name']);

top_10['Weight'] = top_10['Weight'] * 100
print(top_10.loc[:, ["Name", "Weight"]])


# Bottom 10
bottom_10 = cap_weights[-10:].to_frame()
bottom_10.rename({0: 'Weight'},  axis=1, inplace=True)
# print(bottom_10)

for b in bottom_10.index:
    name = sp500.loc[sp500['Symbol'] == b, "Company"].values[0]
    bottom_10.loc[b, "Name"] = name

# fig, ax = plt.subplots()
# ax.pie(bottom_10['Weight'], labels=bottom_10['Name']);

bottom_10['Weight'] = bottom_10['Weight'] * 100
print(bottom_10.loc[:, ["Name", "Weight"]])
