import pandas as pd
import os
from datetime import datetime

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

from reuters_api.hist_downloader import main as download_history
import pandas as pd

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