import pandas as pd
import os, re
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prep_hist_data(contract, basis) -> pd.DataFrame:
    today = pd.Timestamp.now()
    basis['Ticker Composite'] = basis['Ticker Composite'].str.rstrip('=')


    mon_hist_sheet = f"{contract}_history"
    repo_hist_sheet = f"{contract}_repo_history"

    hist_data = pd.read_excel('Basis data.xlsx', sheet_name=mon_hist_sheet)
    repo_hist = pd.read_excel('Basis data.xlsx', sheet_name=repo_hist_sheet)

    # Prep repo history data
    logger.info("Processing repo history data")
    repo_hist = repo_hist[1:].reset_index(drop=True)
    repo_hist.rename(columns={'Unnamed: 0': 'DATE'}, inplace=True)
    repo_hist['DATE'] = pd.to_datetime(repo_hist['DATE'])
    # Select the correct Spot Next (2+1) repo rate column
    repo_hist = repo_hist[['DATE', 'ITGCITASN=TT']].rename(columns={'ITGCITASN=TT': 'Repo Rate'})

    # Prep monitor history data
    logger.info("Processing bond history data")
    # Extract first row as variable names and set correct headers
    ric_to_isin_map = dict(zip(basis['Ticker Composite'], basis['ISIN']))
    variable_names = hist_data.iloc[0, 1:].values  # Exclude first column (dates)

    # Function to clean RIC names (remove .1, .2, etc.)
    def clean_ric_name(ric):
        return re.sub(r"\.\d+$", "", ric)  # Removes the last dot + number suffix

    # Rename columns properly
    new_columns = ['DATE'] + [f"{clean_ric_name(ric)}{var}" for ric, var in zip(hist_data.columns[1:], variable_names)]

    # Apply the new column names and remove the first row (which was just variable names)
    hist_data.columns = new_columns
    hist_data = hist_data[1:].reset_index(drop=True)

    # Convert DATE column to datetime
    hist_data['DATE'] = pd.to_datetime(hist_data['DATE'])
    logger.info("Converting bond data to long format")
    long_hist_data = hist_data.melt(id_vars=['DATE'], var_name='RIC_Variable', value_name='Value')
    long_hist_data[['RIC', 'Variable']] = long_hist_data['RIC_Variable'].str.rsplit('=', n=1, expand=True)
    long_hist_data['ISIN'] = long_hist_data['RIC'].map(ric_to_isin_map)
    long_hist_data = long_hist_data.drop(columns=['RIC_Variable', 'RIC'])
    # Mapping the RICs to ISINs

    long_hist_data = long_hist_data.pivot(index=['DATE', 'ISIN'], columns='Variable', values='Value').reset_index()
    long_hist_data.columns.name = None  # Remove the 'Variable' label

    # Rename columns for clarity
    long_hist_data = long_hist_data.rename(columns={
        'B_YLD_1': 'Yield',
        'MID_PRICE': 'Clean Price',
        'DIRTY_PRC': 'Dirty Price',
        'ACCR_INT': 'Accrued Interest'
    })

    # Where we have no dirty price but we have accrued interest, we fill in dirty price by
    # summing accrued interest to clean price
    long_hist_data.loc[long_hist_data['Dirty Price'].isna(), 'Dirty Price'] = long_hist_data['Clean Price'] + long_hist_data['Accrued Interest']

    logger.info("Merging repo data into bond data")
    long_hist_data = long_hist_data.merge(repo_hist, on='DATE', how='left')
    # Convert numerical columns to float
    numeric_cols = ['Yield', 'Clean Price', 'Dirty Price', 'Accrued Interest', 'Repo Rate']
    long_hist_data[numeric_cols] = long_hist_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # We remove all rows where there is not at least one value in the key columns
    key_columns = ['Accrued Interest', 'BPV', 'Yield', 'Clean Price', 'Dirty Price', 'Repo Rate']
    # Drop rows where ALL key financial columns are NaN
    long_hist_data = long_hist_data.dropna(subset=key_columns, how='all')


    missing_before = long_hist_data[['Accrued Interest', 'Yield', 'Repo Rate']].isnull()
    long_hist_data = long_hist_data.sort_values(by=['ISIN', 'DATE']).reset_index(drop=True)
    # Forward fill missing values within each RIC
    long_hist_data[['Accrued Interest', 'Yield']] = long_hist_data.groupby('ISIN')[['Accrued Interest', 'Yield']].ffill()
    missing_after = long_hist_data[['Accrued Interest', 'Yield']].isnull()
    ffill_counts = (missing_before & ~missing_after).sum()
    logging.info(f"We have the following missing values that we forward filled {ffill_counts.to_dict()}")

    return long_hist_data
    breakpoint()
    

def prep_basis_data(basis: pd.DataFrame, last_delivery: str) -> pd.DataFrame:
    """
    Cleans and prepares the bond basis data by adding necessary financial calculations.

    Parameters:
    - basis (pd.DataFrame): The original bond basis data.
    - last_delivery (str): The delivery date of the futures contract (e.g., "2025-03-10").

    Returns:
    - pd.DataFrame: Processed bond data with additional calculated columns.
    """

    today = pd.Timestamp.now()
    last_delivery = pd.Timestamp(last_delivery)
    days_to_delivery = (last_delivery - today).days

    # Convert maturity date to datetime if not already
    basis["Maturity Date"] = pd.to_datetime(basis["Maturity Date"])

    # Calculate accrued coupon
    current_year = datetime.now().year
    basis['Coupon_accrued'] = basis['Maturity Date'].apply(
        lambda x: (datetime(current_year, x.month, x.day) - today).days / 360
    )
    basis['Coupon_accrued'] = basis['Coupon_accrued'].apply(
        lambda x: (-1 * x if x < 0 else 1 - x)) * basis['Coupon']

    # Compute additional financial values
    basis['Dirty Price'] = basis['Clean Price'] + basis['Coupon_accrued']
    basis['Income to delivery'] = basis['Coupon'] * days_to_delivery / 360
    basis['Cost to delivery'] = basis['Dirty Price'] * (basis['Repo Rate'].div(100)) * (days_to_delivery / 360)
    basis['Carry to delivery'] = basis['Income to delivery'] - basis['Cost to delivery']

    return basis

def load_basis_data(file_path: str, contract: str) -> pd.DataFrame:
    """
    Loads bond and futures data from an Excel file.

    Parameters:
    - file_path (str): Path to the Excel file.
    - contract (str): Contract name (e.g., "BTP", "Bund").

    Returns:
    - pd.DataFrame: Dataframe containing bond and futures data.
    """
    # Read the Excel file
    basis = pd.read_excel(file_path, sheet_name=contract)

    # Rename columns using the mappings dictionary
    from reuters_api.mappings import cat_dict
    basis.columns = basis.columns.map(lambda x: cat_dict.get(x, x))

    # Ensure dates are properly formatted
    basis["Maturity Date"] = pd.to_datetime(basis["Maturity Date"])

    return basis

import pandas as pd

def align_yield_hist_with_maturities(basis: pd.DataFrame, yield_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Renames yield history columns using bond maturity IDs and handles duplicate maturity dates.

    Parameters:
    - basis (pd.DataFrame): The dataframe containing bond details, including RIC2 and maturity dates.
    - yield_hist (pd.DataFrame): The dataframe containing historical yields.

    Returns:
    - pd.DataFrame: Yield history with renamed columns and sorted by date.
    """
    # rename columns of yield hist with respective maturity date in basis
    # Convert maturity date to datetime and create a formatted maturity ID
    basis['Maturity Date'] = pd.to_datetime(basis['Maturity Date'])
    basis['maturityID'] = basis['Maturity Date'].dt.strftime('%b%y')

    # Handle duplicate maturities by appending RIC2
    duplicates = basis.duplicated(subset=['Maturity Date', 'Coupon'], keep=False)
    basis.loc[duplicates, 'maturityID'] = (
        basis.loc[duplicates, 'RIC2'] + " " + basis.loc[duplicates, 'Maturity Date'].dt.strftime('%b%y')
    )

    # Create dictionary mapping RIC2 to maturity ID
    rename_dict = basis.set_index('RIC2')['maturityID'].to_dict()

    # Rename columns in yield history
    yield_hist = yield_hist.rename(columns=rename_dict)
    yield_hist.set_index('DATE', inplace=True)
    yield_hist.sort_values(by='DATE', inplace=True)

    return yield_hist

def build_yield_hist(basis: pd.DataFrame, country: str, mat: int) -> pd.DataFrame:
    """
    Fetches historical yield data and aligns it with bond maturity dates.

    Parameters:
    - basis (pd.DataFrame): The dataframe containing bond details.
    - country (str): The country code for the bonds.
    - mat (int): The maturity period in years.

    Returns:
    - pd.DataFrame: Yield history with renamed columns and sorted by date.
    """
    start = '2023-10-01'
    end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    fields = 'B_YLD_1'

    # Fetch yield data from Reuters API
    if len(basis['RIC2']) == 1:
        data = download_history(basis['RIC2'][0], start, end, fields)
    else:
        data = download_history(basis['RIC2'][0], start, end, fields)

    # Load additional historical yield data from Excel
    basis_path = 'Basis data.xlsx'
    sheet_name = f"{country}{mat}_yields"
    yields = pd.read_excel(basis_path, sheet_name=sheet_name)
    yields = yields.drop([0])
    yields = yields.rename(columns={'Unnamed: 0': 'DATE'})
    yields['DATE'] = pd.to_datetime(yields['DATE'])

    return yields