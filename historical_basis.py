# btp_pca.py
#from dataquery_api import DQInterface
import pandas as pd
import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import subprocess


# Now you can import your functions
from reuters_api.mappings import cat_dict  #
from data_loader import load_basis_data, align_yield_hist_with_maturities, prep_basis_data, prep_hist_data
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

def clean_data(basis, basis_contract,future_hist,bond_hist,deliverable):

    # Future cleaning
    date_col = future_hist.columns[0]
    future_hist = future_hist.rename(columns={date_col: 'DATE'})
    future_hist['DATE'] = pd.to_datetime(future_hist['DATE'])
    future_hist = future_hist.set_index('DATE')
    future_hist = future_hist.sort_index()
    future_hist = future_hist[future_contract_map[basis_contract]]
    future_hist = future_hist.to_frame()
    future_hist.columns = ['Future Price']

    # Bond cleaning
    # We filter the bonds in the history such that we only hold the ones which are present in the
    # deliverable history file - so as to reduce the weight of long_hist
    long_hist_size = len(bond_hist)
    deliverable_isins = deliverable['ISIN'].unique()  # Get unique deliverable ISINs
    bond_hist = bond_hist[bond_hist['ISIN'].isin(deliverable_isins)]  # Keep only deliverable bonds
    logger.info(f"Current history length reduced from {long_hist_size} to {len(bond_hist)}")

    # Clean deliverable
    deliverable = deliverable.rename(columns={'Date': 'DATE'})  # Rename 'Date' to match bond_hist
    deliverable['DATE'] = pd.to_datetime(deliverable['DATE'])  # Ensure date format consistency
    deliverable['Delivery Date'] = pd.to_datetime(deliverable['Delivery Date'])  # Ensure date format consistency

    # We merge deliverable bonds with 'inner' so only when there's a match, and we merge the coupons in
    bond_hist = bond_hist.merge(deliverable[['DATE', 'ISIN']], on=['DATE', 'ISIN'], how='inner')
    bond_hist = bond_hist.merge(basis[['ISIN', 'Coupon']], on='ISIN', how='left')

    logger.info(f"Current history length reduced from {long_hist_size} to {len(bond_hist)} after removing all "
                f"pre-delivererable bond history")

    return bond_hist, future_hist, deliverable

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
    breakpoint()





    return bond_hist[['DATE', 'ISIN', 'Gross Basis']]





def main():

    basis_path = 'Basis data.xlsx'
    basis_contract = 'IK'  # IK #DU
    contract = contract_map[basis_contract]
    deliverable_path = 'DeliverableBondsHistory_' + basis_contract + '.csv'
    country = country_map[basis_contract]
    mat = mat_map[contract]
    compounding = 'Discrete'
    start_year = 2018

    today = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Get data from basis path excel spreadsheet and 'basis' sheet

    basis_monitor_sheet = basis_contract + ' Monitor'
    basis = pd.read_excel(basis_path, sheet_name=basis_monitor_sheet, header=1)
    future_hist = pd.read_excel(basis_path, sheet_name='Futures')
    bond_hist = prep_hist_data(basis_contract, basis)
    deliverable = pd.read_csv(deliverable_path)

    bond_hist, future_hist, deliverable = clean_data(basis,basis_contract,future_hist,bond_hist,deliverable)

    hist_basis(basis_contract, bond_hist, future_hist, deliverable, start_year)


if __name__ == "__main__":
    main()