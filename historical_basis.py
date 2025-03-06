# btp_pca.py
#from dataquery_api import DQInterface
import pandas as pd
import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
import os
import multiprocessing as mp
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), 'reuters_api'))
# Add spotcurve to the Python path
sys.path.append(os.path.abspath("../spotcurve"))
from main import main as run_spotcurve
import matplotlib.pyplot as plt
import subprocess
from pandas.tseries.offsets import CustomBusinessDay
import concurrent.futures


# Now you can import your functions
from reuters_api.mappings import cat_dict  #
from data_loader import load_basis_data, \
    align_yield_hist_with_maturities, prep_basis_data, \
    prep_hist_data, clean_data
from bond_calcs import get_business_days, compute_spot_rate, compute_forward_rate
from bond_calcs import generate_cashflows, compute_price_at_delivery, compute_forward_ytm
from basis_calculations import compute_net_basis, compute_hist_FV_NB, prep_basis_calc
from cf_calculator import get_next_delivery_dates

from pricing import apply_yield_bumps
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datetime import datetime

import logging

import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from mappings import contract_map, country_map, mat_map, future_contract_map


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)


def hist_basis(basis_contract, bond_hist, future_hist, deliverable, start_year):
    """
     Computes historical gross basis for each date and each deliverable bond.

    Parameters:
    - basis_contract (str): The futures contract (e.g., 'IK', 'DU').
    - bond_hist (pd.DataFrame): Historical bond data (filtered for deliverable bonds).
    - future_hist (pd.DataFrame): Historical futures price data.
    - start_year (int): The starting year for historical basis calculations.

    Returns:
    - pd.DataFrame: A DataFrame containing historical gross basis values.
    """
    # today = pd.Timestamp.now().strftime('%Y-%m-%d')
    #
    # dates = pd.date_range(start=f'{start_year}-01-01', end=today, freq='B')
    bond_hist['DATE'] = pd.to_datetime(bond_hist['DATE'])
    deliverable['DATE'] = pd.to_datetime(deliverable['DATE'])

    # ✅ Step 1: Merge deliverable info (Delivery Date & Conversion Factor) into bond_hist
    bond_hist = bond_hist.merge(
        deliverable[['DATE', 'ISIN', 'Delivery Date', 'Conversion Factor']],
        on=['DATE', 'ISIN'],
        how='left'
    )

    # ✅ Step 2: Merge in futures prices
    bond_hist = bond_hist.merge(future_hist, on='DATE', how='left')

    # Carry calculations
    bond_hist['days_to_delivery']  = (bond_hist['Delivery Date'] - bond_hist['DATE']).dt.days
    bond_hist['Income to delivery'] = bond_hist['Coupon'] * bond_hist['days_to_delivery'] / 360
    bond_hist['Cost to delivery'] = bond_hist['Dirty Price'] * (bond_hist['Repo Rate'].div(100)) * (bond_hist['days_to_delivery'] / 360)
    bond_hist['Carry to delivery'] = bond_hist['Income to delivery'] - bond_hist['Cost to delivery']


    # ✅ Step 3: Compute Gross Basis
    bond_hist['Gross Basis'] = bond_hist['Clean Price'] - (bond_hist['Conversion Factor'] * bond_hist['Future Price'])
    bond_hist['Net Basis'] = bond_hist['Gross Basis'] - bond_hist['Carry to delivery']


    return bond_hist[['DATE', 'ISIN', 'Gross Basis', 'Net Basis']]


def forecast_nb(
        today,
        start_year,
        bondData,
        bond_hist,
        long_bond_hist,
        deliverable,
        compounding):
    """
    Forecasts net basis by computing:


    Parameters:
    - basis (DataFrame): The original DataFrame containing bond details, including 'Maturity Date'.
    - last_delivery (str): The last delivery date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame: A DataFrame with bond names, time to maturity, and forward yields.
    """

    dates = get_business_days(start_year, pd.Timestamp(today) - pd.Timedelta(days=2))
    historical_forecasts = pd.DataFrame()
    for curr_date in dates:
        logger.info(f"Processing {curr_date}")
        bondData_ = bondData.copy()
        today = pd.Timestamp.today()
        bondData_, bondData_deliv, next_delivery, cashflows_df = prep_basis_calc(curr_date, today, bondData_, long_bond_hist, deliverable)

        # Get forward curve & segment boundaries from spotcurve model
        forward_curve, segment_boundaries = run_spotcurve(bondData_,curr_date)

        # Reshape forward_curve into (n_segments, 5)
        n_segments = len(segment_boundaries) + 1
        forward_curve_matrix = np.array(forward_curve).reshape(n_segments, 5)

        forecast_data = []

        for i, bond_row in bondData_deliv.iterrows():
            bond_name = bond_row["Description"]  # Use maturity ID
            maturity_date = pd.to_datetime(bond_row["Maturity Date"])
            ric = bond_row["RIC"]

            # Compute time to maturity TODAY (in years)
            time_to_maturity_today = (maturity_date - today).days / 365
            time_to_maturity_at_delivery = (maturity_date - next_delivery).days / 365
            time_to_delivery = (next_delivery - today).days / 365

            # Extract spot rate today for bond maturity
            spot_rate_maturity = compute_spot_rate(time_to_maturity_today, forward_curve_matrix, segment_boundaries)
            spot_rate_delivery = compute_spot_rate(time_to_delivery, forward_curve_matrix, segment_boundaries)

            forward_rate = compute_forward_rate(spot_rate_maturity, spot_rate_delivery, time_to_maturity_today, time_to_delivery, compounding)

            # Extract cashflows for this bond using RIC
            bond_cashflows = cashflows_df[cashflows_df["RIC"] == ric].sort_values(by=['CF Date']).reset_index(drop=True)

            # Compute expected bond price at delivery
            fwd_px_at_delivery, bond_cf = compute_price_at_delivery(
                bond_cashflows, forward_curve_matrix, segment_boundaries, time_to_delivery, compounding)

            fwd_ytm = compute_forward_ytm(bond_cf, fwd_px_at_delivery, compounding)

            # Convert bond_row to dictionary and add new fields
            bond_data = bond_row.to_dict()
            bond_data["DATE"] = curr_date
            bond_data["Fwd_Px_At_Delivery"] = fwd_px_at_delivery
            bond_data["Fwd_YTM"] = fwd_ytm
            bond_data["Fwd_Clean_Px_At_Delivery"] = fwd_px_at_delivery - bond_data.get("Income to delivery", 0)

            # Append to results list
            forecast_data.append(bond_data)


        forecast_df = pd.DataFrame(forecast_data)
        #Computes the historical Fair Value NB
        forecast_df = compute_hist_FV_NB(forecast_df)

        if not forecast_df.empty:
            historical_forecasts = pd.concat([historical_forecasts, forecast_df], ignore_index=True)
    historical_forecasts.to_csv("historical_forecasts.csv", index=False)
    return historical_forecasts
    breakpoint()


def main():

    basis_path = 'Basis data.xlsx'
    basis_contract = 'IK'  # IK #DU
    contract = contract_map[basis_contract]
    deliverable_path = 'DeliverableBondsHistory_' + basis_contract + '.csv'
    country = country_map[basis_contract]
    mat = mat_map[contract]
    compounding = 'Discrete'
    start_year = 2025

    today = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Get data from basis path excel spreadsheet and 'basis' sheet

    basis_monitor_sheet = basis_contract + ' Monitor'
    bondData = pd.read_excel(basis_path, sheet_name=basis_monitor_sheet, header=1)
    future_hist = pd.read_excel(basis_path, sheet_name='Futures')
    bond_hist = prep_hist_data(basis_contract, bondData)
    deliverable = pd.read_csv(deliverable_path)

    bondData, bond_hist, future_hist, deliverable, long_bond_hist = clean_data(bondData,basis_contract,future_hist,bond_hist,deliverable)

    forecasts = forecast_nb(today, start_year, bondData, bond_hist[['DATE', 'ISIN', 'Yield']], long_bond_hist[['DATE', 'ISIN', 'Yield', 'Dirty Price', 'Repo Rate']], deliverable, compounding)
    basis_hist = hist_basis(basis_contract, bond_hist, future_hist, deliverable, start_year)
    basis_hist.to_csv("basis_hist.csv")

    forecasts_df = forecasts.merge(basis_hist[["DATE", "ISIN", "Net Basis"]], on=["ISIN", "DATE"], how="left")
    breakpoint()


def plot_net_basis_comparison(forecasts_df):
    """
    Plots Fair Value Net Basis vs Actual Net Basis over time for each ISIN.

    Parameters:
    - forecasts_df (pd.DataFrame): DataFrame containing "DATE", "ISIN", "FV Net Basis", and "Net Basis".

    Returns:
    - None (displays plots).
    """
    # Ensure DATE column is in datetime format
    forecasts_df["DATE"] = pd.to_datetime(forecasts_df["DATE"])

    # Get unique ISINs for plotting
    unique_isins = forecasts_df["ISIN"].unique()

    # Create plots for each ISIN
    for isin in unique_isins:
        isin_data = forecasts_df[forecasts_df["ISIN"] == isin]

        plt.figure(figsize=(10, 5))
        plt.plot(isin_data["DATE"], isin_data["FV Net Basis"], label="Fair Value Net Basis", linestyle="-", color="blue")
        plt.plot(isin_data["DATE"], isin_data["Net Basis"], label="Actual Net Basis", linestyle="--", color="red")

        plt.xlabel("Date")
        plt.ylabel("Net Basis")
        plt.title(f"Net Basis Comparison for ISIN: {isin}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.show()


def plot_net_basis_comparison(forecasts_df):
    """
    Plots Fair Value Net Basis vs Actual Net Basis over time on the same chart.

    Each ISIN is represented by a unique color.

    Parameters:
    - forecasts_df (pd.DataFrame): DataFrame containing "DATE", "ISIN", "FV Net Basis", and "Net Basis".

    Returns:
    - None (displays the plot).
    """
    # Ensure DATE column is in datetime format
    forecasts_df["DATE"] = pd.to_datetime(forecasts_df["DATE"])

    # Get unique ISINs and define a color map
    unique_isins = forecasts_df["ISIN"].unique()
    colors = plt.cm.get_cmap("tab10", len(unique_isins))

    plt.figure(figsize=(12, 6))

    # Plot each ISIN with the same color for FV Net Basis and Net Basis
    for idx, isin in enumerate(unique_isins):
        isin_data = forecasts_df[forecasts_df["ISIN"] == isin]
        color = colors(idx)  # Assign a unique color to each ISIN

        plt.plot(isin_data["DATE"], isin_data["FV Net Basis"], label=f"{isin} FV Net Basis", linestyle="-", color=color)
        plt.plot(isin_data["DATE"], isin_data["Net Basis"], label=f"{isin} Actual Net Basis", linestyle="--",
                 color=color, alpha=0.7)

    # Labels, legend, and formatting
    plt.xlabel("Date")
    plt.ylabel("Net Basis")
    plt.title("Fair Value Net Basis vs Actual Net Basis Over Time")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside
    plt.grid(True)
    plt.xticks(rotation=45)

    # Adjust layout for readability
    plt.tight_layout()

    plt.show()
if __name__ == "__main__":
    main()


#ToDo if deliverable bonds is not run, this means it will not find deliverable bonds.



# Time without multi threading: