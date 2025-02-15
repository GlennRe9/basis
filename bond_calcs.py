import pandas as pd
import numpy as np
from scipy.optimize import brentq



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

    coupon = bond['Coupon']
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


def compute_price_at_delivery(cashflows, times, spot_rate_delivery, compounding="Discrete"):
    """
    Computes the expected bond price at delivery.

    Parameters:
    - cashflows (list): Future cashflows (coupons + principal).
    - times (list): Time (in years) from delivery until each cashflow.
    - spot_rate_delivery (float): Spot rate at delivery.
    - compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

    Returns:
    - float: Expected bond price at delivery.
    """
    if compounding == "Cont":
        return sum(cf * np.exp(-spot_rate_delivery * t) for cf, t in zip(cashflows, times))
    elif compounding == "Discrete":
        return sum(cf / ((1 + spot_rate_delivery) ** t) for cf, t in zip(cashflows, times))
    else:
        raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

