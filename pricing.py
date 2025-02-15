import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'reuters_api'))
# Add spotcurve to the Python path
sys.path.append(os.path.abspath("../spotcurve"))
from main import main as run_spotcurve
from bond_calcs import get_forward_rate, compute_spot_rate, compute_forward_yield, compute_forward_rate
from scipy.optimize import brentq
from bond_calcs import compute_price_at_delivery, compute_spot_rate, compute_forward_rate



def apply_yield_bumps(basis: pd.DataFrame, betas: dict, yield_bumps: list) -> pd.DataFrame:
    """
    Computes bond price changes for different yield shift scenarios.

    Parameters:
    - basis (pd.DataFrame): Bond data, including clean price and BPV.
    - betas (dict): Yield betas for each bond.
    - yield_bumps (list): List of yield shifts to apply (e.g., [-100, -50, 0, +50, +100] bps).

    Returns:
    - pd.DataFrame: Price changes for each bond under different yield scenarios.
    """
    price_df = pd.DataFrame(columns=['Bonds'] + [f'{bump}' for bump in yield_bumps])

    for index, row in basis.iterrows():
        if pd.notna(row['bpv']):  # Ensure BPV is available
            bond_id = row['maturityID']
            beta = betas.get(bond_id, 1)  # Default to 1 if beta is missing

            # Compute price changes based on yield bumps
            price_changes = [row['Clean Price'] + (bump * beta * -(row['bpv']) / 100) for bump in yield_bumps]

            # Store in DataFrame
            price_df.loc[len(price_df)] = [bond_id] + price_changes

    price_df.set_index('Bonds', inplace=True)

    return price_df




def forecast_nb(basis, last_delivery, compounding, cashflows_df):
    """
    Forecasts net basis by computing:
    1) Time to maturity today.
    2) Time to maturity at the last delivery date.
    3) Forward yield for each bond in 1 month.

    Parameters:
    - basis (DataFrame): The original DataFrame containing bond details, including 'Maturity Date'.
    - last_delivery (str): The last delivery date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame: A DataFrame with bond names, time to maturity, and forward yields.
    """

    # Get forward curve & segment boundaries from spotcurve model
    forward_curve, segment_boundaries = run_spotcurve()

    # Reshape forward_curve into (n_segments, 5)
    n_segments = len(segment_boundaries) + 1
    forward_curve_matrix = np.array(forward_curve).reshape(n_segments, 5)

    # Convert last delivery date to a timestamp
    last_delivery = pd.Timestamp(last_delivery)
    today = pd.Timestamp.today()

    forecast_data = []

    for i, bond_row in basis.iterrows():
        bond_name = bond_row["maturityID"]  # Use maturity ID
        maturity_date = bond_row["Maturity Date"]
        ric = bond_row["RIC"]

        # Compute time to maturity TODAY (in years)
        time_to_maturity_today = (maturity_date - today).days / 365
        time_to_maturity_at_delivery = (maturity_date - last_delivery).days / 365
        time_to_delivery = (last_delivery - today).days / 365

        # Extract spot rate today for bond maturity
        spot_rate_maturity = compute_spot_rate(time_to_maturity_today, forward_curve_matrix, segment_boundaries)
        spot_rate_delivery = compute_spot_rate(time_to_delivery, forward_curve_matrix, segment_boundaries)

        forward_rate = compute_forward_rate(spot_rate_maturity, spot_rate_delivery, time_to_maturity_today, time_to_delivery, compounding)

        # Extract cashflows for this bond using RIC
        bond_cashflows = cashflows_df[cashflows_df["RIC"] == ric]
        cashflows = bond_cashflows["Coupon"].tolist()
        cashflow_times = (bond_cashflows["Ttm"] - time_to_delivery).tolist()  # Adjust time from delivery date

        # Compute expected bond price at delivery
        price_at_delivery = compute_price_at_delivery(cashflows, cashflow_times, spot_rate_delivery, compounding)

        # Append results
        forecast_data.append([
            bond_name,
            time_to_maturity_today,
            time_to_maturity_at_delivery,
            time_to_delivery,
            spot_rate_maturity,
            spot_rate_delivery
        ])

        # Convert to DataFrame
    forecast_df = pd.DataFrame(forecast_data, columns=[
        "Bond", "Time to Maturity Today", "Time to Maturity at Delivery",
        "Time to Delivery", "Spot Rate Today (Maturity)", "Spot Rate Today (Delivery)"
    ])

    return forecast_df


def compute_forward_ytm(cashflows, times, price_at_delivery, spot_rate_delivery, compounding="Discrete"):
    """
    Computes the Forward Yield to Maturity (YTM) at delivery using numerical solving.

    Parameters:
    - cashflows (list): List of future cashflows (coupons + principal).
    - times (list): Time (in years) from delivery until each cashflow.
    - price_at_delivery (float): Expected bond price at delivery.
    - spot_rate_delivery (float): Spot rate at delivery.
    - compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

    Returns:
    - float: Forward Yield to Maturity (YTM).
    """

    def ytm_function(ytm):
        if compounding == "Cont":
            return sum(cf * np.exp(-ytm * t) for cf, t in zip(cashflows, times)) - price_at_delivery
        elif compounding == "Discrete":
            return sum(cf / ((1 + ytm) ** t) for cf, t in zip(cashflows, times)) - price_at_delivery
        else:
            raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

    # Solve for YTM numerically using Brent's method
    try:
        ytm_forward = brentq(ytm_function, -0.5, 0.5)  # Bounds for solving YTM
        return ytm_forward
    except ValueError:
        return np.nan  # Return NaN if solving fails