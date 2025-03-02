import pandas as pd
import numpy as np
from scipy.optimize import brentq, root_scalar


def compute_forward_yield(T1, T2, forward_curve_matrix, segment_boundaries):
    """
    Computes the forward yield for a bond with time-to-maturity T1 today,
    that will have time-to-maturity T2 in 1 month.

    Parameters:
    - T1 (float): Current time-to-maturity of the bond.
    - T2 (float): Future time-to-maturity in 1 month.
    - forward_curve_matrix (numpy.ndarray): A (n_segments, 5) matrix of quartic coefficients.
    - segment_boundaries (numpy.ndarray): List of segment breakpoints, including 0.

    Returns:
    - float: The forward yield for the bond in 1 month.
    """
    # Get today’s spot rate for T1 (10-year)
    y_T1 = get_forward_rate(T1, forward_curve_matrix, segment_boundaries)

    # Get today’s spot rate for 1 month (1/12 year)
    y_1M = get_forward_rate(1 / 12, forward_curve_matrix, segment_boundaries)

    # Compute the forward yield using the spot-to-forward formula
    y_T2 = ((1 + y_T1) ** T1) / ((1 + y_1M) ** (1 / 12))

    # Convert back to an annualized yield
    forward_yield = y_T2 ** (1 / T2) - 1

    return forward_yield

def get_forward_rate(maturity, forward_curve, segment_boundaries):
    """
    Computes the forward rate at a given maturity from the segmented forward curve.

    Parameters:
    - maturity (float): The target time to maturity (e.g., 9.92 years).
    - forward_curve (list): List of quartic polynomial coefficients for each segment.
    - segment_boundaries (list): List of segment breakpoints.

    Returns:
    - float: The forward rate at the given maturity.
    """
    # Find which segment the maturity falls into
    for i in range(len(segment_boundaries) - 1):
        if segment_boundaries[i] <= maturity < segment_boundaries[i + 1]:
            a, b, c, d, e = forward_curve[i]  # Extract the coefficients for the segment
            T = maturity  # The time to maturity
            return a * T**4 + b * T**3 + c * T**2 + d * T + e  # Evaluate the quartic polynomial

    # If maturity is beyond the last segment, use the last segment's polynomial
    a, b, c, d, e = forward_curve[-1]
    return a * maturity**4 + b * maturity**3 + c * maturity**2 + d * maturity + e
def compute_spot_rate(T, forward_curve_matrix, segment_boundaries):
    """
    Computes the spot rate for a given time-to-maturity T by integrating the forward curve.

    Parameters:
    - T (float): The time to maturity in years for which we want the spot rate.
    - forward_curve_matrix (numpy.ndarray): A (n_segments, 5) matrix of quartic coefficients.
    - segment_boundaries (numpy.ndarray): List of segment breakpoints, including 0.

    Returns:
    - float: The spot rate for the given maturity.
    """
    # Ensure segment_boundaries includes 0 as the starting point
    segment_boundaries = np.insert(segment_boundaries, 0, 0)
    segment_boundaries = np.append(segment_boundaries, segment_boundaries[-1] + 1)

    # Ensure T is within the bounds
    if T < 0 or T > segment_boundaries[-1]:
        raise ValueError(f"🚨 Error: Requested maturity {T} is out of bounds (0 to {segment_boundaries[-1]}).")

    # Find the correct segment
    total_integral = 0
    last_T = 0

    for i in range(len(segment_boundaries) - 1):
        T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]

        # Extract coefficients for the segment
        a, b, c, d, e = forward_curve_matrix[i]

        # Determine integration range
        if T <= T_end:
            T_values = np.linspace(last_T, T, 100)
        else:
            T_values = np.linspace(last_T, T_end, 100)

        # Compute forward rates over this segment
        f_values = a * T_values ** 4 + b * T_values ** 3 + c * T_values ** 2 + d * T_values + e

        # Approximate integration using cumulative sum
        total_integral += np.trapz(f_values, T_values)  # Numerical integration

        # If T is within this segment, stop
        if T <= T_end:
            break

        last_T = T_end

    # Compute the spot rate
    spot_rate = total_integral / T if T > 0 else 0
    return spot_rate

import numpy as np

def compute_forward_rate(spot_rate_long, spot_rate_short, T_long, T_short, compounding="Discrete"):
    """
    Computes the implied forward rate between two maturities.

    Parameters:
    - spot_rate_long (float): Spot rate for longer maturity (e.g., bond maturity).
    - spot_rate_short (float): Spot rate for shorter maturity (e.g., time to delivery).
    - T_long (float): Time to the longer maturity (years).
    - T_short (float): Time to the shorter maturity (years).
    - compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

    Returns:
    - forward_rate (float): Implied forward rate between T_short and T_long.
    """

    # Compute discount factors
    if compounding == "Cont":
        df_long = np.exp(-spot_rate_long * T_long)
        df_short = np.exp(-spot_rate_short * T_short)
    elif compounding == "Discrete":
        df_long = (1 / (1 + spot_rate_long)) ** T_long
        df_short = (1 / (1 + spot_rate_short)) ** T_short
    else:
        raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

    # Compute forward rate
    forward_rate = (df_short / df_long) ** (1 / (T_long - T_short)) - 1

    return forward_rate

def generate_cashflows(bond, today=None):
    """
    Generate a DataFrame of cashflows for a given bond.

    Parameters:
    - bond (Series): A single bond row from `basis` DataFrame.
    - today (str or Timestamp): The reference date (defaults to today).

    Returns:
    - DataFrame: Cashflows with time-to-cashflow.
    """
    # Convert relevant fields to Timestamps
    maturity = pd.to_datetime(bond["Maturity Date"], dayfirst=True)
    #issue_date = pd.to_datetime(["Issue Date"], dayfirst=True)
    today = pd.Timestamp.today() if today is None else pd.to_datetime(today)

    coupon = bond['Coupon'] / 2
    cashflow_dates = []
    next_coupon_date = maturity

    # Generate semi-annual coupon dates
    while next_coupon_date >= today:
        cashflow_dates.append(next_coupon_date)
        next_coupon_date -= pd.DateOffset(months=6)

    cashflow_dates = pd.DatetimeIndex(cashflow_dates)

    # Compute Time to Maturity (Ttm) for each cashflow
    ttms = (cashflow_dates - today).days / 365  # Convert TimedeltaIndex to numeric
    ttms = pd.Series(ttms).round(3)  # Convert to Series before rounding

    # Construct DataFrame
    cashflow_df = pd.DataFrame({
        'CF Date': cashflow_dates,
        'Coupon': coupon,
        'Ttm': ttms,
        'RIC': bond['RIC'],
        'Dirty Price': bond['Dirty Price'],
        'Maturity Date': maturity,
        'Spot': None,
    })



    # Add face value (100) at maturity
    cashflow_df.loc[cashflow_df['CF Date'] == maturity, 'Coupon'] += 100

    return cashflow_df

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


def compute_price_at_delivery(bond_cf, forward_curve_matrix, segment_boundaries, time_to_delivery, compounding="Discrete"):
    """
    Computes the expected bond price at delivery using the correct forward-implied spot rates.

    Parameters:
    - cashflows (list): Future cashflows (coupons + principal).
    - cashflow_times (list): Time (in years) from today until each cashflow.
    - forward_curve_matrix (numpy.ndarray): A (n_segments, 5) matrix of quartic coefficients for the forward curve.
    - segment_boundaries (numpy.ndarray): List of segment breakpoints, including 0.
    - time_to_delivery (float): Time (in years) from today to the delivery date.
    - compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

    Returns:
    - float: Expected bond price at delivery.
    """
    today = pd.Timestamp.today()
    bond_cf = bond_cf.copy()
    bond_cf['fwd_Ttm'] = bond_cf["Ttm"] - time_to_delivery

    # Compute today's spot rate for the delivery time
    delivery_spot = compute_spot_rate(time_to_delivery, forward_curve_matrix, segment_boundaries)
    bond_cf['spot_today'] = bond_cf['Ttm'].apply(lambda t: compute_spot_rate(t, forward_curve_matrix, segment_boundaries))


    price_at_delivery = 0

    # Initialize empty lists to store computed values
    current_spot_list, fwd_spot_list, fwd_df_list, discounted_CF_list = [], [], [], []

    # Loop through each row in the dataframe
    for _, row in bond_cf.iterrows():
        cf_amount = row['Coupon']
        cf_t = row['Ttm']
        fwd_ttm = row['fwd_Ttm']

        # Compute today's spot rate for this cashflow time
        current_spot = compute_spot_rate(cf_t, forward_curve_matrix, segment_boundaries)
        current_spot_list.append(current_spot)

        # Compute the forward-implied spot rate at delivery for this cashflow
        fwd_spot = compute_forward_rate(
            current_spot, # Spot rate today for this cashflow maturity
            delivery_spot, # Spot rate today for delivery date
            cf_t,  # Time to cashflow
            time_to_delivery, # Time to delivery
            compounding)
        fwd_spot_list.append(fwd_spot)

        # Compute forward discount factors
        if compounding == "Cont":
            fwd_df = np.exp(-fwd_spot * fwd_ttm)
        elif compounding == "Discrete":
            fwd_df = 1 / ((1 + fwd_spot) ** fwd_ttm)
        else:
            raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

        fwd_df_list.append(fwd_df)

        # Compute discounted cashflow
        discounted_CF = cf_amount * fwd_df
        discounted_CF_list.append(discounted_CF)

    # Store computed values into bond_cf DataFrame
    bond_cf['spot_today'] = current_spot_list
    bond_cf['fwd_spot'] = fwd_spot_list
    bond_cf['fwd_df'] = fwd_df_list
    bond_cf['discounted_CF'] = discounted_CF_list

    bond_fwd_px = bond_cf['discounted_CF'].sum()
    return bond_fwd_px, bond_cf

def compute_forward_ytm(bond_cf, fwd_px_at_delivery, compounding):
    """
    Computes the forward yield to maturity (YTM) for a bond at the delivery date.

    Parameters:
    - bond_cf (DataFrame): DataFrame containing bond cashflows with 'Coupon' and 'fwd_Ttm'.
    - fwd_px_at_delivery (float): The forward price of the bond at delivery.
    - compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

    Returns:
    - float: The forward yield to maturity.
    """

    def present_value_of_cashflows(y):
        """Computes the present value of bond cashflows discounted at YTM y."""
        if compounding == "Cont":
            discounted_cf = bond_cf['Coupon'] * np.exp(-y * bond_cf['fwd_Ttm'])
        elif compounding == "Discrete":
            discounted_cf = bond_cf['Coupon'] / ((1 + y) ** bond_cf['fwd_Ttm'])
        else:
            raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

        return discounted_cf.sum() - fwd_px_at_delivery

    # Solve for YTM using numerical root-finding
    try:
        result = root_scalar(present_value_of_cashflows, bracket=[-0.1, 0.5], method='brentq')
        return result.root if result.converged else np.nan
    except ValueError:
        return np.nan  # If the solver fails, return NaN