# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:47:20 2019

@author: razin.hussain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import datetime
import math

import blpapi
import pdblp

cd "path to the master_portfolioreturn.csv file"

# Requires connection to Bloomberg
con = pdblp.BCon(debug = False, port = 8194, timeout = 50000)
con.start()

df = pd.read_csv("master_portreturn.csv")
cols_to_keep = ["date", "port_capitalgain", "port_divyield", "port_totalreturn"]
df = df[cols_to_keep]

df["date"] = df["date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

# Index the returns 
index_value = 100.0
capital_returnlst = []
for cap_return in df["port_capitalgain"]:
    new_value = index_value * (1 + cap_return)
    capital_returnlst.append(new_value)
    index_value = new_value

df["PORTFOLIO_LOCAL"] = capital_returnlst

#index_value = 100.0
#total_returnlst = []
#for total_return in df["port_totalreturn"]:
#    new_value = index_value * (1 + total_return)
#    total_returnlst.append(new_value)
#    index_value = new_value
#
#df["port_totalret_indexed"] = total_returnlst

# df = df.set_index("date")
# df

# Calculating annual capital return, dividend yield and total return
# annual_div = df.resample("A-DEC").sum()
# annual_div


msciworld_local = con.bdh("MSDLWI Index", "PX_LAST", "20041231", "20190328")
msciworld_local.columns = msciworld_local.columns.droplevel("field")
msciworld_local = msciworld_local.pct_change(1)
msciworld_local = msciworld_local.iloc[1:]
# MSDLWI
index_value = 100.0
msciworld_returnlst = []
for ret in msciworld_local["MSDLWI Index"]:
    new_value = index_value * (1 + ret)
    msciworld_returnlst.append(new_value)
    index_value = new_value

msciworld_local["MSCIWORLD_LOCAL"] = msciworld_returnlst

msciworld_local = msciworld_local.reset_index()
# df = df.reset_index()

df = pd.merge(df, msciworld_local, how = "inner", on = "date")


# Calculating metrices
df = df.set_index("date")
annual_data = df[["PORTFOLIO_LOCAL", "MSCIWORLD_LOCAL", "port_divyield"]].resample("A-DEC").sum()

port_std = df["PORTFOLIO_LOCAL"].std()
msci_std = df["MSCIWORLD_LOCAL"].std()

port_std_3yr = df


df.to_csv("comps_data.csv")



df = pd.read_csv("comps1.csv")
df = df.drop(["Unnamed: 0"], axis = 1)
df["date"] = df["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
df = df.set_index("date")


# Year variables used to change the x-axis tick label
years = mdates.YearLocator()
yearsfmt = mdates.DateFormatter("%b-%y")

# Plotting port vs msci world, always create figure with fig, ax objects
fig, ax = plt.subplots(figsize = (12, 6))

# Plot of two time-series
ax.plot(df["PORTFOLIO_LOCAL"], color = "darkslategrey", linewidth = 1.3, linestyle = "-", label = "PORTFOLIO_LOCAL")
ax.plot(df["MSCIWORLD_LOCAL"], color = "steelblue", linewidth = 1.3, linestyle = "-", label = "MSCIWORLD_LOCAL")


# Common attributes to add to the graph
ax.legend(loc = "upper left", fontsize = 11)
# ax.set_xlabel("Date", fontsize = 12)
ax.set_ylabel("Index Points", fontsize = 12)
ax.set_title("PORTFOLIO vs MSCI WORLD - Gross Capital Returns (in local currency)", fontsize = 15)

# Change tick location and format tick label
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsfmt)

# Remove both major and minor ticks from the y-axis
ax.tick_params(axis = "y", which = "both", length = 0)

# Remove the figure boundaries
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)

# Dram vertical lines on the graph to indicate weak market periods and shade between the vertical lines
date_lst = [("10-12-2007 00:00:00", "04-01-2009 00:00:00"), ("06-10-2015 00:00:00", "06-20-2016 00:00:00"), ("09-03-2018 00:00:00", "02-03-2019 00:00:00")]
for dates in date_lst:
    ax.axvline(x = dates[0], linewidth = 1.3, linestyle = "--", color = "black")
    ax.axvline(x = dates[1], linewidth = 1.3, linestyle = "--", color = "black")
    
    ax.axvspan(dates[0], dates[1], alpha = 0.5, color = "lightgrey")

# Adding labels to the graph
style = dict(size = 9, color = "black")
ax.text("10-12-2007 00:00:00", 141.09361014756, "Recession start", **style, horizontalalignment = "center")
ax.text("04-01-2009 00:00:00", 72.2223069442472, "Recession end", **style, horizontalalignment = "center")
ax.text("12-25-2015 00:00:00", 153.531129283333, "High Vol period", **style, horizontalalignment = "center")
ax.text("12-20-2018 00:00:00", 195.679567745621, "High Vol period", **style, horizontalalignment = "center")

# Adding data points to the graph
ax.text("03-28-2019 00:00:00", 188.240517748025, "188.2", **style, horizontalalignment = "center")
ax.text("03-28-2019 00:00:00", 394.262961785275, "394.3", **style, horizontalalignment = "center")



# ax.yaxis.set_major_locator(plt.NullLocator())


# Very handy date_range module
index = pd.date_range("20050101", periods = 20, freq = "Q-DEC")
data = {"a": np.random.randint(1, 20, 20)}

df = pd.DataFrame(data, index = index)
plt.figure(figsize = (12, 7))
plt.plot(df["a"], "b-")

index


