# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 07:42:31 2019

@author: ABMRazin
"""

# Uncomment the change directory when you are directly running the script and comment out when importing the script for another program
cd "path to the bloomberg api"

import blpapi
import pdblp
import pandas as pd
import sqlite3
import datetime
import math
import numpy as np
from functools import reduce
from cvxopt import blas, solvers, matrix
import matplotlib.pyplot as plt


con = pdblp.BCon(debug = False, port = 8194, timeout = 50000)
con.start()

# Uncomment the change directory when you are directly running the script and comment out when importing the script for another program
cd T:\Razin.Hussain\low_vol_global_equity

conn = sqlite3.connect("msci_world.sqlite3")
cur = conn.cursor()


# period_5yr: Date 5 years ago from the cutoff date
# period_3yr: Date 3 years ago from the cutoff date
# cutoff_date: Cutoff date for the model; it can be either the last quarter end date or last month end date
# rebalancing_term: Either "Q" or "M" based on either quarterly or monthly rebalancing


# Connecting to the in-house database
def get_data(bb_ticker, period_5yr, cutoff_date):
    # Scrapes data from bloomberg on a dataframe
    df = con.bdh(bb_ticker, ["PX_LAST", "CUR_MKT_CAP", "T12_FCF_YIELD", "PX_TO_BOOK_RATIO", "EQY_DVD_YLD_IND", "BETA_ADJ_OVERRIDABLE", "RETURN_COM_EQY", "RETURN_ON_INV_CAPITAL", "NET_DEBT_TO_EBITDA"], period_5yr, cutoff_date)
    
    # Drop one layer of column heading
    df.columns = df.columns.droplevel()
    df = df.reset_index()

    # Rename column names
    for col in df.columns:
        if col == "date":
            df = df.rename(columns = {col: "price_date"})
        if col == "PX_LAST":
            df = df.rename(columns = {col: "close_price"})
        if col == "CUR_MKT_CAP":
            df = df.rename(columns = {col: "market_cap"})
        if col == "T12_FCF_YIELD":
            df = df.rename(columns = {col: "fcf_yield"})
        if col == "PX_TO_BOOK_RATIO":
            df = df.rename(columns = {col: "book_to_price"})
        if col == "EQY_DVD_YLD_IND":
            df = df.rename(columns = {col: "dividend_yield"})
        if col == "BETA_ADJ_OVERRIDABLE":
            df = df.rename(columns = {col: "2yr_adj_beta"})
        if col == "RETURN_COM_EQY":
            df = df.rename(columns = {col: "roe"})
        if col == "RETURN_ON_INV_CAPITAL":
            df = df.rename(columns = {col: "roic"})
        if col == "NET_DEBT_TO_EBITDA":
            df = df.rename(columns = {col: "net_debt_to_ebitda"})
    # Make a list of current column names and check whether a column exist in the df or not 
    col_lst = []
    for col in df.columns:
        col_lst.append(col)
    
    # If a column does not exist, create an empty column of the len of the df
    if "market_cap" not in col_lst:
        df["market_cap"] = [None for i in range (len(df))]
    if "fcf_yield" not in col_lst:
        df["fcf_yield"] = [None for i in range (len(df))]
    if "book_to_price" not in col_lst:
        df["book_to_price"] = [None for i in range (len(df))]
    if "dividend_yield" not in col_lst:
        df["dividend_yield"] = [None for i in range (len(df))]
    if "2yr_adj_beta" not in col_lst:
        df["2yr_adj_beta"] = [None for i in range (len(df))]
    if "roe" not in col_lst:
        df["roe"] = [None for i in range (len(df))]
    if "roic" not in col_lst:
        df["roic"] = [None for i in range (len(df))]
    if "net_debt_to_ebitda" not in col_lst:
        df["net_debt_to_ebitda"] = [None for i in range (len(df))]
    
    # Change data tyoe from string to float
    df["close_price"] = df["close_price"].astype("float64")
    
    # Calculate daily return
    df["daily_return"] = df["close_price"].pct_change()
        
    # Fill na
    df["daily_return"] = df["daily_return"].fillna(0)
#   df = df.set_index("price_date")
    
    # Calculate the standard deviations
    df["1yr_std"] = (df["daily_return"].rolling(window = 252).std())*math.sqrt(252)
    df["3yr_std"] = (df["daily_return"].rolling(window = 756).std())*math.sqrt(252)
    df["5yr_std"] = (df["daily_return"].rolling(window = 1260).std())*math.sqrt(252)
    
    # Reciprocal of price-to-book
    df["book_to_price"] = 1/df["book_to_price"]
    
#    df["dividend_yield"] = df["dividend_yield"].fillna(0)
    
    # Fetch from certain items from the locally hosted database
    cur.execute('select id, name, sector, country, weight, currency, exchange from symbol where bb_ticker = ?', (bb_ticker,))
    # Puts the fetched items in a tuple
    fetch = cur.fetchone()
    
    # Get symbol id from the tuple and make a list of that and put it in the dataframe
    symbol_id = fetch[0]
    symbol_id = [symbol_id for i in range(len(df))]
    df["symbol_id"] = symbol_id
    
    # Get company names from the tuple, make a list and put it in the df
    company_name = fetch[1]
    company_name = [company_name for i in range(len(df))]
    df["company_name"] = company_name

    # Get sector names from the tuple, make a list and put it in the df
    sector = fetch[2]
    sector = [sector for i in range(len(df))]
    df["sector"] = sector

    # Get country names from the tuple, make a list and put it in the df
    country = fetch[3]
    country = [country for i in range(len(df))]
    df["country"] = country

    # Get index weights of companies from the tuple, make a list and put it in the df
    index_weight = fetch[4]
    index_weight = [index_weight for i in range(len(df))]
    df["index_weight"] = index_weight
    
    # Get market currency for companies from the tuple, make a list and put it in the df
    currency = fetch[5]
    currency = [currency for i in range(len(df))]
    df["currency"] = currency

    # Get exchange for companies from the tuple, make a list and put it in the df
    exchange = fetch[6]
    exchange = [exchange for i in range(len(df))]
    df["exchange"] = exchange
    
    # Make a list of the tickers and put it in the df
    bb_ticker = [bb_ticker for i in range(len(df))]
    df["bb_ticker"] = bb_ticker
    
    # List all the columns you want to keep
    columns_to_keep = ["symbol_id", "price_date", "bb_ticker", "company_name", "sector", "country", "index_weight", "currency", "exchange", "1yr_std", "3yr_std", "5yr_std", "2yr_adj_beta", "fcf_yield", "book_to_price", "dividend_yield", "roe", "roic", "net_debt_to_ebitda", "market_cap"]
    df = df[columns_to_keep]
        
    # Filling in missing values
    try:
        df = df.fillna(method = "ffill")
    except:
        df = df.fillna(0)
    
    return df


def resample(bb_ticker, period_5yr, cutoff_date, rebalancing_term):
    df = get_data(bb_ticker, period_5yr, cutoff_date)
    df = df.set_index("price_date")
    df.index = pd.to_datetime(df.index, "s")
    
    # Resampling to change daily frequency to quarterly and get the last available data in the quarter
    df = df.resample(rebalancing_term).last()

    df[["fcf_yield", "book_to_price", "roe", "roic", "net_debt_to_ebitda"]] = df[["fcf_yield", "book_to_price", "roe", "roic", "net_debt_to_ebitda"]].fillna(method = "ffill")
    df["dividend_yield"] = df["dividend_yield"].fillna(0)
    
    df = df.tail(1)
    df = df.reset_index()
    
    return df


def concat(period_5yr, cutoff_date, rebalancing_term):
    cur.execute('SELECT bb_ticker FROM symbol')
    tickers = cur.fetchall()
    security_lst = []
    for ticker in tickers:
        ticker = ticker[0]
        security_lst.append(ticker)
    df_lst = []
    for bb_ticker in security_lst[0:1183]:
        try:
            df = resample(bb_ticker, period_5yr, cutoff_date, rebalancing_term)
        except:
            continue
        # Store dfs in a list
        df_lst.append(df)
    
#    print (len(df_lst))
    # Concatenate the dataframes that is stored in the list
    df = pd.concat(df_lst, axis = 0)
    
    return df


def z_score(period_5yr, cutoff_date, rebalancing_term):
    df = concat(period_5yr, cutoff_date, rebalancing_term)
#    key = lambda x: x.sector
    zscore = lambda x: (x - x.mean())/x.std()
    
    df_transform = df.groupby("sector")["1yr_std", "3yr_std", "5yr_std", "2yr_adj_beta", "fcf_yield", "book_to_price", "dividend_yield", "roe", "roic", "net_debt_to_ebitda"].transform(zscore)
    df_transform = df_transform.fillna(0)
    
    # Change the column names  
    for col in df_transform.columns:
        df_transform = df_transform.rename(columns = {col: col + "_" + "zscore"})
    
    df = pd.concat([df, df_transform], axis = 1)
    
    # Calculate the scores 

#    factor_weights = {"1yr_std_weight": 0.30, "3yr_std_weight": 0.15, "5yr_std_weight": 0.10, "2yr_adj_beta_weight": 0.15, "fcf_yield_weight": 0.10, "book_to_price_weight": 0.025, "dividend_yield_weight": 0.025, "roe_weight": 0.05, "roic_weight": 0.05, "net_debt_to_ebitda_weight": 0.05}
#    df_weight = pd.DataFrame(factor_weights)

    df["1yr_std_score"] = df["1yr_std_zscore"]*0.15
    df["3yr_std_score"] = df["3yr_std_zscore"]*0.25
    df["5yr_std_score"] = df["5yr_std_zscore"]*0.15
    df["2yr_adj_beta_score"] = df["2yr_adj_beta_zscore"]*0.15
    df["fcf_yield_score"] = df["fcf_yield_zscore"]*0.10
    df["book_to_price_score"] = df["book_to_price_zscore"]*0.025
    df["dividend_yield_score"] = df["dividend_yield_zscore"]*0.025
    df["roe_score"] = df["roe_zscore"]*0.025
    df["roic_score"] = df["roic_zscore"]*0.075
    df["net_debt_to_ebitda_score"] = df["net_debt_to_ebitda_zscore"]*0.05

    df["score"] = -df["1yr_std_score"] - df["3yr_std_score"] - df["5yr_std_score"] - df["2yr_adj_beta_score"] + df["fcf_yield_score"] + df["book_to_price_score"] + df["dividend_yield_score"] + df["roe_score"] + df["roic_score"] - df["net_debt_to_ebitda_score"]
    # Rank companies based on scores
    df["rank"] = df["score"].rank(ascending = False)

    df = df.sort_values(by = ["score"], ascending = False)
#    
    return df

# Pick top 10% company from every sector 
def basket(period_5yr, cutoff_date, rebalancing_term):
    df = z_score(period_5yr, cutoff_date, rebalancing_term)
    
    # Sorting within groups based on column "score"
    p = 0.065

    df = (df.groupby("sector", group_keys = False).apply(lambda x: x.nlargest(int(len(x) * p), "score")))
    
    return df


# Start of the optimization module
# Fetching price data from bb to calculate annual returns for companies and covariance matrix
def data_collector(period_5yr, period_3yr, cutoff_date, rebalancing_term):
    quant_basket = basket(period_5yr, cutoff_date, rebalancing_term)
    quant_basket = quant_basket.sort_values(by = ["country", "rank"])
    ticker_lst = [ticker for ticker in quant_basket["bb_ticker"]]

    # Create a df list to hold all company level df
    df_price_lst = []
    df_dailyreturn_lst = []
    # Scrap data for each company from bloomberg
    for bb_ticker in ticker_lst:
        df = con.bdh(bb_ticker, "PX_LAST", period_3yr, cutoff_date)
        df.columns = df.columns.droplevel("field")
        
        try:
            df.columns = [bb_ticker]
        except:
            quant_basket = quant_basket.set_index("bb_ticker")
            quant_basket = quabt_basket.drop([bb_ticker])
            quant_basket = quant_basket.reset_index()
            continue
        
        df_price = df.resample(rebalancing_term).last()
        df_price = df_price.reset_index()
        df_price_lst.append(df_price)
        
        df_dailyreturn = df.pct_change(1)
        df_dailyreturn = df_dailyreturn.reset_index()
        df_dailyreturn_lst.append(df_dailyreturn)
    
    # Concatenate the list of price dfs
    df_price = reduce(lambda left, right: pd.merge(left, right, how = "inner", on = "date"), df_price_lst)
    
    # Concatenate the list of daily return dfs
    df_dailyreturn = reduce(lambda left, right: pd.merge(left, right, how = "inner", on = "date"), df_dailyreturn_lst)
    df_dailyreturn = df_dailyreturn.set_index("date")
    
    df_dailyreturn = df_dailyreturn.iloc[1:]
    
    # Calculating covariance matrix and annualized return matrix
    cov_matrix = df_dailyreturn.cov()
    annual_return = df_dailyreturn.mean()*252
    return_matrix = annual_return.as_matrix()
    
    return df_dailyreturn, cov_matrix, return_matrix, ticker_lst, quant_basket, df_price


def portfolio_constraints(number_of_stocks, sector_lst, companycount_by_country_lst):
    n = number_of_stocks
        
    # Adding upper and lower company constraint functions (see documentation for the formulation of the optimization problem)
    G_company_lower = matrix(0.0, (n,n))
    G_company_lower[::n+1] = -1.0
    G_company_upper = matrix(0.0, (n,n))
    G_company_upper[::n+1] = 1.0
    
    G = np.append(G_company_lower, G_company_upper, axis = 0)

    h_company_lower = matrix(-0.008, (n,1)) # Minimum weight that can be invested in a company 
    h_company_upper = matrix(0.035, (n,1))  # Maximum weight that can be invested in a company

    h = np.append(h_company_lower, h_company_upper, axis = 0)
    
    # Adding sector limits (see documentation)
    
    # Append 1 by n matrix for every sector
    lst = [i for i, n in enumerate(sector_lst) if n == "Communication"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)
    
    lst = [i for i, n in enumerate(sector_lst) if n == "Consumer Discretionary"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)

    lst = [i for i, n in enumerate(sector_lst) if n == "Consumer Staples"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)
    
    lst = [i for i, n in enumerate(sector_lst) if n == "Energy"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)

    lst = [i for i, n in enumerate(sector_lst) if n == "Financials"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)
    
    lst = [i for i, n in enumerate(sector_lst) if n == "Health Care"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)

    lst = [i for i, n in enumerate(sector_lst) if n == "Industrials"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)
    
    lst = [i for i, n in enumerate(sector_lst) if n == "Information Technology"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)

    lst = [i for i, n in enumerate(sector_lst) if n == "Materials"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)
    
    lst = [i for i, n in enumerate(sector_lst) if n == "Real Estate"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)
    
    lst = [i for i, n in enumerate(sector_lst) if n == "Utilities"]
    x = np.zeros((n))
    i = np.array(lst)
    x[i] = 1.0
    x = x.reshape((1,n))
    G = np.append(G, x, axis = 0)

    # The sector weights are in order of Communication, Consumer Discretionary, Consumer Staples, Energy, Financials, Healthcare, Industrials, IT, Materials, Real Estate, Utilities
#    h_sector_cap = matrix([-0.044, 0.164, -0.0557, 0.1857, -0.0443, 0.1643, -0.0208, 0.1408, -0.1224, 0.2424, -0.0903, 0.2103, -0.071, 0.191, -0.1092, 0.2292, -0.0161, 0.1261, -0.0028, 0.1128, -0.0035, 0.1135])

    msci_sector_weight = matrix([0.0840, 0.1057, 0.0843, 0.0608, 0.1624, 0.1303, 0.1110, 0.1492, 0.0461, 0.0328, 0.0335])
    sector_delta = 0.08
    h_sector_cap = msci_sector_weight + sector_delta
    

    h = np.append(h, h_sector_cap, axis = 0)
    
    # Adding country limits
    country_total = 0
    for country_count in companycount_by_country_lst:
        previous_country_total = country_total
        country_total = previous_country_total + country_count

#        G_country_lower = matrix(0.0, (1,n))
#        G_country_lower[:, previous_country_total:country_total] = -1.0

        G_country_upper = matrix(0.0, (1,n))
        G_country_upper[:, previous_country_total:country_total] = 1.0
        
        G = np.append(G, G_country_upper, axis = 0)
    
    # The country weights are in order of Australia, Austria, Belgium, Canada, Denmark, Finland, France, Germany, Hong Kong, Ireland, Israel, Italy, Japan, Netherlands, New Zealnd,  Norway, Portugal, Singapore, Spain, Sweden, Switzerland, UK and US
    msci_country_weight = matrix([0.0240, 0.0036, 0.0036, 0.0348, 0.0036, 0.0036, 0.0376, 0.0297, 0.0132, 0.0036, 0.0036, 0.0036, 0.0851, 0.0116, 0.0036, 0.0036, 0.0036, 0.0036, 0.0107, 0.0036, 0.0300, 0.0582, 0.6193]) 
    country_delta = 0.10   # Maximum overweight above the index country weights
    h_country_upper = msci_country_weight + country_delta

    h = np.append(h, h_country_upper, axis = 0)
    
    G = matrix(G)
    h = matrix(h)

    # All weights needs to add to one
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    
    return G, h, A, b


def cash_balance():
    cash_weight = {"bb_ticker": "**CASH**", "optimal_weight": 0.015, "index_weight": 0.015, 
                   "equal_weight": 0.015, "company_name": None, "sector": None, "country": "Canada"}
    cash_weight = pd.DataFrame(cash_weight, index = [0])
    
    return cash_weight


# Module to search through the efficient frontier in 20bps increments
def std_search(required_std, df):
    count = 0
    for i in df["minimum_volatility"]:
        count += 1
        j = df["minimum_volatility"][count]
        if (((i - required_std > 0.0) and (j - required_std > 0.0)) or ((i - required_std < 0.0) and (j - required_std < 0.0))):
            if count < (len(df) - 1):
                continue
            elif count == (len(df) - 1):
                required_std += 0.002
#                print ("Adding 0.002")
                return std_search(required_std, df)
        else:
            right = i - required_std
            left = required_std - j
            if right < left:
                print (i)
                df = df[df["minimum_volatility"] == i]
                return df
            elif left < right:
                print (j)
                df = df[df["minimum_volatility"] == j]
                return df


def optimal_portfolio(period_5yr, period_3yr, cutoff_date, rebalancing_term):
    data = data_collector(period_5yr, period_3yr, cutoff_date, rebalancing_term)
    df1 = data[0]
    number_of_stocks = len(df1.columns)
    
    # Get quarter end price
    allocation_price = data[5]
    
    ticker_lst = data[3]
    
    df2 = data[4]
    sector_lst = df2["sector"].tolist()
#    print (companycount_by_sector_lst)

    companycount_by_country = df2.groupby("country").count()
    companycount_by_country = companycount_by_country.reset_index()
    portfolio_country_lst = companycount_by_country["country"].tolist()
    
    msci_country_lst = ["Australia", "Austria", "Belgium", "Canada", "Denmark", "Finland", "France", "Germany", "Hong Kong", "Ireland", "Israel", "Italy", "Japan", "Netherlands", "New Zealand", "Norway", "Portugal", "Singapore", "Spain", "Sweden", "Switzerland", "United Kingdom", "United States"]

    count = 0
    companycount_by_country_lst = []
    for country in msci_country_lst:
        if country in portfolio_country_lst:
            companycount_by_country_lst.append(companycount_by_country["bb_ticker"][count])
            count = count + 1
        else:
            companycount_by_country_lst.append(0)
    
#    print (companycount_by_country_lst)
    
    constraint_matrices = portfolio_constraints(number_of_stocks, sector_lst, companycount_by_country_lst)
    G = constraint_matrices[0]
    h = constraint_matrices[1]
    A = constraint_matrices[2]
    b = constraint_matrices[3]
    
    # Adding covariance and return matrix for the objective function
    Q = data[1]  # Covariance matrix (3 year average annualized)
    Q = np.matrix(Q)
    Q = matrix(Q)
    
    p = data[2]   # Returns matrix   (3 year average annualized) 
    p = matrix(p)
    
    # Adding a list called rhos which represents proportion of var-covar and return to optimize
    N = 100
    rhos = [100**(5 * t/N - 1.0) for t in range(N)] 
    
    # Calculating the efficient frontier weights with qp by calling the solvers function in cvxopt
    portfolios = [solvers.qp(rho*Q, -p, G, h, A, b)["x"] for rho in rhos]
    
    # Calculating return and std for each efficient frontier portfolio
    returns = [blas.dot(p, x) for x in portfolios]
    std = [(np.sqrt(blas.dot(x, Q*x)))*math.sqrt(252) for x in portfolios]
    
    df = pd.DataFrame()
    df["optimal_returns"] = returns
    df["minimum_volatility"] = std
    
    for counter, symbol in enumerate(ticker_lst):
        df[symbol] = [weight[counter] for weight in portfolios]

    
    # Closest weight that gives 6.8% standard deviation    
    required_std = 0.068
    # calling std_search function to find the best weight matrix - see documentation
    df_copy = std_search(required_std, df)
    
    optimal_moments = df_copy.iloc[:, 0:2]
    optimal_weights = df_copy.iloc[:, 2:]
    
    # Convert the dataframe to a series to convert the column names to rows 
    optimal_weights = optimal_weights.T.squeeze()
    
    final_df = pd.DataFrame()
    final_df["optimal_weight"] = optimal_weights
    final_df = final_df.reset_index()
    final_df = final_df.rename(columns = {"index": "bb_ticker"})
    
    df2_retain = df2[["index_weight", "bb_ticker", "company_name", "sector", "country", "currency", "exchange"]]
#    df_retain = df_retain.set_index("bb_ticker")

    final_df = pd.merge(final_df, df2_retain, how = "inner", on = "bb_ticker")
    final_df["index_weight"] = final_df["index_weight"].apply(lambda x: x/final_df["index_weight"].sum())
    
    final_df["equal_weight"] = [0.985/len(final_df) for i in range (len(final_df))]
#    columns_to_keep = ["bb_ticker", "optimal_weight", "index_weight", "equal_weight", "company_name", "sector", "country"]
#    final_df = final_df[columns_to_keep]
    
    # Imposing upper limit for company weight
    count = 0
    for weight in final_df["index_weight"]:
        upper_limit = 0.040
        if weight > upper_limit:
            final_df.iloc[count:count+1, 2:3] = upper_limit
            increment = (weight - upper_limit)/len(final_df)
            final_df["index_weight"] = final_df["index_weight"] + increment
        count = count + 1
    
    # Rescaling equity weights to equal 98.5% total equity weight and 1.5% cash balane
    final_df["optimal_weight"] = final_df["optimal_weight"].apply(lambda x: (x/final_df["optimal_weight"].sum())*0.985)
    final_df["index_weight"] = final_df["index_weight"].apply(lambda x: (x/final_df["index_weight"].sum())*0.985)    
    
    # Calling the cash function
    df_cash = cash_balance()
    final_df = pd.concat([df_cash, final_df], axis = 0, ignore_index = True)

    columns_to_keep = ["bb_ticker", "optimal_weight", "index_weight", "equal_weight", "company_name", "sector", "country", "currency", "exchange"]
    final_df = final_df[columns_to_keep]
    
    try:
        final_df.to_csv("Archive\port_{0}.csv".format(cutoff_date))
        df2.to_csv("Archive\quant_basket_{0}.csv".format(cutoff_date))
        allocation_price.to_csv("Archive\quarterend_prices_{0}.csv".format(cutoff_date))
    except:
        print ("Cannot save files as csv")
        pass

    # df2 is the quant basket - basket of securities with scores and score attributes
    # df contains all the optimal portfolios in the efficient frontier
    # final_df contains securities to be invested in along with the weights
    # Return and std of the optimal portfolio we are investing in    
    return df2, df, final_df, optimal_moments, ticker_lst, allocation_price


# Inputs to the model
period_5yr = "20131231"
period_3yr = "20151231"
cutoff_date = "20181231"
rebalancing_term = "Q"

x = optimal_portfolio(period_5yr, period_3yr, cutoff_date, rebalancing_term)


final_portfolio = x[2]
quant_basket = x[0]
#df = x[1]
#moments = x[3]
allocation_price = x[5]
#

#final_portfolio.to_csv("Archive\port_{0}.csv".format(cutoff_date))
#quant_basket.to_csv("Archive\quant_basket_{0}.csv".format(cutoff_date))
#allocation_price.to_csv("Archive\quarterend_prices_{0}.csv".format(cutoff_date))

dfx = pd.read_csv("Archive\port_{0}.csv".format(cutoff_date))
dfx["currency"]


def holdings(cutoff_date, portfolio_value):
    df = pd.read_csv("Archive\port_{0}.csv".format(cutoff_date))
    df = df.drop("Unnamed: 0", axis = 1)
    df = df.iloc[1:]
    
    currency_rate_lst = []
    for curr in df["currency"]:
        if curr != "CAD":
            bb_curr = "CAD" + curr[:] + " Curncy"
            df = df.replace(curr, bb_curr)
            curr_df = con.bdh(bb_curr, ["PX_LAST"], cutoff_date, cutoff_date)
            curr_df.columns = curr_df.columns.droplevel("field")
            curr_rate = curr_df.iloc[0][bb_curr]
            if curr == "GBP":
                curr_rate = curr_rate * 100      # Converting GBP into pence since stock price in London exchange is quoted in pence
            else:
                pass
            
            currency_rate_lst.append(curr_rate)
        else:
            curr_rate = 1.0
            currency_rate_lst.append(curr_rate)
    
    df["currency_rate"] = currency_rate_lst
    df["CAD_allocation"] = df["optimal_weight"]*portfolio_value
    df["domestic_allocation"] = df["currency_rate"] * df["CAD_allocation"]
    
    price = pd.read_csv("Archive\quarterend_prices_{0}.csv".format(cutoff_date))
    try:
        price = price.drop("Unnamed: 0", axis = 1)
    except:
        pass

    price = price.set_index("date")
    price = price.tail(1)
    price = price.T
    price = price.reset_index()

    price = price.rename(columns = {"index": "bb_ticker"})
    date = list(price.columns.values)[1]
    
    df = pd.merge(df, price, how = "inner", on = "bb_ticker")
    df["Shares"] = round((df["domestic_allocation"]/df[date]), 0)
    
    # convert the bb_ticker column and Shares column into a dictionary with bb_tickers as keys and Shares as values
    bb_ticker_lst = [ticker for ticker in df["bb_ticker"]]
    shares_lst = [shares for shares in df["Shares"]]
    portfolio = dict(zip(bb_ticker_lst, shares_lst))
    
    return portfolio, df 


x = holdings(cutoff_date, 1000000)   
port = x[0]
print(port)

df = x[1]
df.to_csv("test52.csv")





#def holdings(period_5yr, period_3yr, cutoff_date, rebalancing_term, portfolio_value):
#    data = optimal_portfolio(period_5yr, period_3yr, cutoff_date, rebalancing_term)
#    df = data[2]
#    price = data_collector(period_5yr, period_3yr, cutoff_date, rebalancing_term)[5]
#    price = price.tail(1)
#    price = price.transpose()
#    price = price.reset_index()
#    price = price.rename(columns = {"date": "bb_ticker"})
#    date = list(price.columns.values)[1]
#    
#    df = df.merge(df, price, how = "inner", on = "bb_ticker")
#    
#    df["allocation"] = df["optimal_weight"]*portfolio_value
#    df["Shares"] = round((df["allocation"]/df[date]), 0)
#    
#    # convert the bb_ticker column and Shares column into a dictionary with bb_tickers as keys and Shares as values 
#    bb_ticker_lst = [ticker for ticker in df["bb_ticker"]]
#    shares_lst = [shares for shares in df["Shares"]]
#    
#    portfolio = dict(zip(bb_ticker_lst, shares_lst))
#    
#    return portfolio, df


def order_generator(cutoff_date, portfolio_value):
    new_portfolio = holdings(cutoff_date, portfolio_value)
    new_portfolio = new_portfolio[0]
    
    current_portfolio = pd.read_excel("Archive\current_holdings.xlsx", sheet_name = "PORTFOLIO")
    current_tickerlst = [ticker for ticker in current_portfolio["bb_ticker"]]
    try:
        current_portfolio = current_portfolio.set_index("bb_ticker")
    except:
        pass

    new_tickerlst = []
    for new_ticker, new_shares in new_portfolio.items():
        new_tickerlst.append(new_ticker)

        if new_ticker in current_tickerlst:
            current_shares = current_portfolio.loc[new_ticker]["Shares"]
            if new_shares > current_shares:
                print ("BUY {0} shares of {1}".format((new_shares - current_shares), new_ticker))
            elif new_shares < current_shares:
                print ("SELL {0} shares of {1}".format((current_shares - new_shares), new_ticker))
            elif new_shares == current_shares:
                print ("NO CHANGE in {0}".format(new_ticker))
        elif new_ticker not in current_tickerlst:
            print ("NEW BUY {0} shares of {1}".format(new_shares, new_ticker))
    
    for current_ticker in current_tickerlst:
        if current_ticker not in new_tickerlst:
            current_shares = current_portfolio.loc[current_ticker]["Shares"]
            print("SELL ALL {0} shares of {1}".format(current_shares, current_ticker))
    
    return new_portfolio, current_portfolio

x = order_generator(cutoff_date, 1000000)









df = con.bdh("AAPL US Equity", ["PX_LAST"], "20160101", "20181231")
df.columns = df.columns.droplevel("field")
df.columns = ["AAPL US Equity"]
df = df.resample("Q").last()
df = df.tail(1)

df

df = pd.DataFrame({"date": ["20190101"], "AAPL US Equity": [174.95], "DHR US Equity": [75.56]})



df = df.set_index("date")
df = df.transpose()
df

list(df.columns.values)[1]

df.columns = df.columns.droplevel()
df

print (round(1515, 0))

dct1 = {"a": 5, "b": 10, "c": 15}
dct2 = {"a": 7, "b": 8, "d": 6}

dct1["a"]


for key, value in dct1.items():
    if key in 
    
lst = [1, 5, 8]
lst.index(5)

x = [1, 2, 5, 10]
for i in x:
    j = "jam"
    print ("My fav food is {0} and value is {1}".format(j, i))

# Changing the rows of a column
df = pd.DataFrame({"a": [5, 10, 15], "b": [10, 14, 16], "c": ["jpy", "gbp", "cad"]})
# df = df.set_index("c")

for curr in df["c"]:
    if curr == "cad":
        pass
    else:
        bb_curr = "cad" + curr[:] + " currncy"
        df = df.replace(curr, bb_curr)        

df


df["c"] != "cad"


if df["c"] != "cad":
    df["c"] = df["c"].apply(lambda x: "cad" + x[:] + " currncy")

else:
    df["c"] = df["c"].apply(lambda x: "cad" + x[:] + " currncy")

    
