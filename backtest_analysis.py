# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:17:34 2019

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

# Requires Bloomberg connection
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

msciworld_local = con.bdh("MSDLWI Index", ["PX_LAST", "EQY_DVD_YLD_12M"], "20041231", "20190328")
msciworld_local.columns = msciworld_local.columns.droplevel()
msciworld_local["PX_LAST"] = msciworld_local["PX_LAST"].pct_change(1)
msciworld_local.columns = ["msci_capitalgain", "msci_divyield"]
msciworld_local["msci_divyield"] = msciworld_local["msci_divyield"]/100
msciworld_local = msciworld_local.iloc[1:]

index_value = 100.0
msciworld_returnlst = []
for ret in msciworld_local["msci_capitalgain"]:
    new_value = index_value * (1 + ret)
    msciworld_returnlst.append(new_value)
    index_value = new_value


msciworld_local["MSCIWORLD_LOCAL"] = msciworld_returnlst

msciworld_local = msciworld_local.reset_index()

df = pd.merge(df, msciworld_local, how = "inner", on = "date")
df.to_csv("comps1.csv")


port_std = (df["port_capitalgain"].std()) * math.sqrt(252)
msci_std = (df["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std)
print (msci_std)

port_std_3yr = (df.loc["2016-03-28 00:00:00":]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_3yr = (df.loc["2016-03-28 00:00:00":]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_3yr)
print (msci_std_3yr)

port_std_5yr = (df.loc["2014-03-28 00:00:00":]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_5yr = (df.loc["2014-03-28 00:00:00":]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_5yr)
print (msci_std_5yr)

port_std_10yr = (df.loc["2009-03-28 00:00:00":]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_10yr = (df.loc["2009-03-28 00:00:00":]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_10yr)
print (msci_std_10yr)

port_std_recession = (df.loc["2007-10-12 00:00:00":"2009-05-12 00:00:00"]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_recession = (df.loc["2007-10-12 00:00:00":"2009-05-12 00:00:00"]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_recession)
print (msci_std_recession)

df
df = df.set_index("date")

port_std = (df["port_capitalgain"].std()) * math.sqrt(252)
msci_std = (df["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std)
print (msci_std)

port_std_3yr = (df.loc["2016-03-28 00:00:00":]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_3yr = (df.loc["2016-03-28 00:00:00":]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_3yr)
print (msci_std_3yr)

port_std_5yr = (df.loc["2014-03-28 00:00:00":]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_5yr = (df.loc["2014-03-28 00:00:00":]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_5yr)
print (msci_std_5yr)

port_std_10yr = (df.loc["2009-03-28 00:00:00":]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_10yr = (df.loc["2009-03-28 00:00:00":]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_10yr)
print (msci_std_10yr)

port_std_recession = (df.loc["2007-10-12 00:00:00":"2009-05-12 00:00:00"]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_recession = (df.loc["2007-10-12 00:00:00":"2009-05-12 00:00:00"]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_recession)
print (msci_std_recession)

port_std_2018 = (df.loc["2018-09-03 00:00:00":"2019-02-03 00:00:00"]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_2018 = (df.loc["2018-09-03 00:00:00":"2019-02-03 00:00:00"]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_2018)
print (msci_std_2018)

port_std_2015 = (df.loc["2015-06-10 00:00:00":"2016-06-10 00:00:00"]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_2015 = (df.loc["2015-06-10 00:00:00":"2016-06-10 00:00:00"]["msci_capitalgain"].std()) * math.sqrt(252)
port_std_2015 = (df.loc["2015-06-10 00:00:00":"2016-06-10 00:00:00"]["port_capitalgain"].std()) * math.sqrt(252)
msci_std_2015 = (df.loc["2015-06-10 00:00:00":"2016-06-10 00:00:00"]["msci_capitalgain"].std()) * math.sqrt(252)
print (port_std_2015)
print (msci_std_2015)



df = pd.read_csv("comps1.csv")
df = df.drop("Unnamed: 0", axis = 1)
df = df.set_index("date")

# 1 yr Analysis
df_1yr = df.loc["2018-03-28":]
df_1yr_ret = ((df_1yr.loc["2019-03-28", "PORTFOLIO_LOCAL"])/(df_1yr.loc["2018-03-28", "PORTFOLIO_LOCAL"]))**(1/1)-1
df_1yr_div = (df_1yr["port_divyield"].sum())
df_1yr_total = df_1yr_ret + df_1yr_div
df_1yr_std = df_1yr["port_capitalgain"].std()*math.sqrt(252)

df_1yr = df.loc["2018-03-28":]
df_1yr_ret = ((df_1yr.loc["2019-03-28", "MSCIWORLD_LOCAL"])/(df_1yr.loc["2018-03-28", "MSCIWORLD_LOCAL"]))**(1/1)-1
df_1yr_div = (df_1yr["msci_divyield"].mean())
df_1yr_total = df_1yr_ret + df_1yr_div
df_1yr_std = df_1yr["msci_capitalgain"].std()*math.sqrt(252)

df_1yr_ret
df_1yr_div
df_1yr_total
df_1yr_std

# 3 yr Analysis
df_3yr = df.loc["2016-03-28":]
df_3yr_ret = ((df_3yr.loc["2019-03-28", "PORTFOLIO_LOCAL"])/(df_3yr.loc["2016-03-28", "PORTFOLIO_LOCAL"]))**(1/3)-1
df_3yr_div = (df_3yr["port_divyield"].sum())/3
df_3yr_total = df_3yr_ret + df_3yr_div
df_3yr_std = df_3yr["port_capitalgain"].std()*math.sqrt(252)

df_3yr = df.loc["2016-03-28":]
df_3yr_ret = ((df_3yr.loc["2019-03-28", "MSCIWORLD_LOCAL"])/(df_3yr.loc["2016-03-28", "MSCIWORLD_LOCAL"]))**(1/3)-1
df_3yr_div = (df_3yr["msci_divyield"].mean())
df_3yr_total = df_3yr_ret + df_3yr_div
df_3yr_std = df_3yr["msci_capitalgain"].std()*math.sqrt(252)

df_3yr_ret
df_3yr_div
df_3yr_total
df_3yr_std


# 5 yr Analysis
df_5yr = df.loc["2014-03-28":]
df_5yr_ret = ((df_5yr.loc["2019-03-28", "PORTFOLIO_LOCAL"])/(df_5yr.loc["2014-03-28", "PORTFOLIO_LOCAL"]))**(1/5)-1
df_5yr_div = (df_5yr["port_divyield"].sum())/5
df_5yr_total = df_5yr_ret + df_5yr_div
df_5yr_std = df_5yr["port_capitalgain"].std()*math.sqrt(252)

df_5yr = df.loc["2014-03-28":]
df_5yr_ret = ((df_5yr.loc["2019-03-28", "MSCIWORLD_LOCAL"])/(df_5yr.loc["2014-03-28", "MSCIWORLD_LOCAL"]))**(1/5)-1
df_5yr_div = (df_5yr["msci_divyield"].mean())
df_5yr_total = df_5yr_ret + df_5yr_div
df_5yr_std = df_5yr["msci_capitalgain"].std()*math.sqrt(252)

df_5yr_ret
df_5yr_div
df_5yr_total
df_5yr_std


# 10 yr Analysis
df_10yr = df.loc["2009-03-27":]
df_10yr_ret = ((df_10yr.loc["2019-03-28", "PORTFOLIO_LOCAL"])/(df_10yr.loc["2009-03-27", "PORTFOLIO_LOCAL"]))**(1/10)-1
df_10yr_div = (df_10yr["port_divyield"].sum())/10
df_10yr_total = df_10yr_ret + df_10yr_div
df_10yr_std = df_10yr["port_capitalgain"].std()*math.sqrt(252)

df_10yr = df.loc["2009-03-27":]
df_10yr_ret = ((df_10yr.loc["2019-03-28", "MSCIWORLD_LOCAL"])/(df_10yr.loc["2009-03-27", "MSCIWORLD_LOCAL"]))**(1/10)-1
df_10yr_div = (df_10yr["msci_divyield"].mean())
df_10yr_total = df_10yr_ret + df_10yr_div
df_10yr_std = df_10yr["msci_capitalgain"].std()*math.sqrt(252)

df_10yr_ret
df_10yr_div
df_10yr_total
df_10yr_std


# Full history Analysis
df_full = df.loc[:]
df_full_ret = ((df_full.loc["2019-03-28", "PORTFOLIO_LOCAL"])/(df_full.loc["2005-01-03", "PORTFOLIO_LOCAL"]))**(1/14.25)-1
df_full_div = (df_full["port_divyield"].sum())/14.25
df_full_total = df_full_ret + df_full_div
df_full_std = df_full["port_capitalgain"].std()*math.sqrt(252)

df_full = df.loc[:]
df_full_ret = ((df_full.loc["2019-03-28", "MSCIWORLD_LOCAL"])/(df_full.loc["2005-01-03", "MSCIWORLD_LOCAL"]))**(1/14.25)-1
df_full_div = (df_full["msci_divyield"].mean())
df_full_total = df_full_ret + df_full_div
df_full_std = df_full["msci_capitalgain"].std()*math.sqrt(252)

df_full_ret
df_full_div
df_full_total
df_full_std



