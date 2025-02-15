# btp_pca.py
#from dataquery_api import DQInterface
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'reuters_api'))
# Add spotcurve to the Python path
sys.path.append(os.path.abspath("../spotcurve"))

import subprocess


# Now you can import your functions
from reuters_api.mappings import cat_dict  #
from data_loader import load_basis_data, align_yield_hist_with_maturities, prep_basis_data
from data_loader import build_yield_hist
from main import main as run_spotcurve
from reuters_api.hist_downloader import main as download_history  # Rename import to avoid conflicts
from yield_processing import compute_bond_betas
from basis_calculations import compute_net_basis, compute_futures_price
from bond_calcs import compute_spot_rate, get_forward_rate, compute_forward_yield, generate_cashflows
from pricing import forecast_nb

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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)


country_map = {
    'IK': 'IT',
    'RX': 'DE',
    'DU': 'DE'
}

contract_map = {
    'IK': 'IK',
    'DU': 'Schatz',
    'RX': 'Bund'
}

mat_map = {
    'Schatz': 2,
    'IK': 10,
    'Bund': 10
}

def yield_focaresting(basis, yield_hist, last_delivery, compounding, cashflows_df):
    yield_hist = align_yield_hist_with_maturities(basis, yield_hist)

    betas = compute_bond_betas(yield_hist)  # New function

    yield_bumps = [-100, -90, -80, -70, -60, -50,-40,-30,-20,-10,0,10,20,30,40,50,60,70, 80, 90, 100]

    basis['Beta'] = basis.apply(lambda x: betas.get(x['maturityID'], None), axis=1)

    price_df = apply_yield_bumps(basis, betas, yield_bumps)

    net_basis = compute_net_basis(basis, price_df)

    net_basis = net_basis.round(3)
    forecast_nb(basis, last_delivery, compounding, cashflows_df)


    breakpoint()

import numpy as np





def main():

    basis_path = 'Basis data.xlsx'
    basis_contract = 'IK'  # IK #DU
    contract = contract_map[basis_contract]
    country = country_map[basis_contract]
    mat = mat_map[contract]
    compounding = 'Discrete'
    last_delivery = '2025-03-10'


    # Get data from basis path excel spreadsheet and 'basis' sheet
    basis = load_basis_data(basis_path, contract)
    basis = prep_basis_data(basis, last_delivery)


    today = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Generate cashflows **before** yield forecasting
    cashflow_list = basis.apply(lambda bond: generate_cashflows(bond, today), axis=1)
    cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True)

    yield_hist = build_yield_hist(basis, country, mat)
    da = yield_focaresting(basis, yield_hist, last_delivery, compounding, cashflows_df)


if __name__ == "__main__":
    main()