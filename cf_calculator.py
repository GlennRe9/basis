import pandas as pd
import numpy as np
import os, re
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay, BMonthEnd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

def clean_monitor_data(bondData, ref_date):
    # We make time data readable
    bondData['Maturity Date'] = pd.to_datetime(bondData['Maturity Date'], errors='coerce')
    bondData['Coupon Last Date'] = pd.to_datetime(bondData['Coupon Last Date'], errors='coerce')
    bondData['Next Coupon Date'] = pd.to_datetime(bondData['Next Coupon Date'], errors='coerce')
    bondData['Issue Date'] = pd.to_datetime(bondData['Issue Date'], errors='coerce')
    bondData['Maturity Date'] = bondData['Maturity Date'].dt.strftime('%d-%m-%Y')
    bondData['Coupon Last Date'] = bondData['Coupon Last Date'].dt.strftime('%d-%m-%Y')
    bondData['Next Coupon Date'] = bondData['Next Coupon Date'].dt.strftime('%d-%m-%Y')
    bondData['Issue Date'] = bondData['Issue Date'].dt.strftime('%d-%m-%Y')
    bondData = bondData.drop(columns=['Ticker', 'Coupon Frequency', 'Price Accrued Interest Flag'])
    bondData['Ttm'] = ((pd.to_datetime(bondData['Maturity Date'], dayfirst=True,
                                       errors='coerce') - ref_date).dt.days / 365).round(3)
    bondData['Original Maturity'] = ((pd.to_datetime(bondData['Maturity Date'], dayfirst=True, errors='coerce')
                                      - pd.to_datetime(bondData['Issue Date'], dayfirst=True,
                                                       errors='coerce')).dt.days / 365).round(3)
    bondData['Coupon'] = pd.to_numeric(bondData['Coupon'], errors='coerce')
    bondData = bondData[bondData['Coupon'] != 0]
    bondData['Coupon'] = bondData['Coupon'] / 2
    bondData[~bondData['Yield to Maturity'].isna()]
    bondData['Yield to Maturity'] = bondData['Yield to Maturity'] / 100
    # Remove rows where column 'Dirty Price' is not a number
    bondData = bondData[~bondData['Dirty Price'].astype(str).str.contains('[a-zA-Z]', na=False)]
    # Drop first bond as it's usually volatile
    # bondData = bondData.iloc[1:].reset_index(drop=True)

    # Remove '/d' from description end
    bondData['Description'] = bondData['Description'].str.rstrip('/d')
    bondData['Description'] = bondData['Description'].str.replace(r'(\d{2})(\d{2})$', r'\1/20\2', regex=True)

    bondData['Maturity Date'] = pd.to_datetime(bondData["Maturity Date"], format='%d-%m-%Y', dayfirst=True)
    bondData['Issue Date'] = pd.to_datetime(bondData["Issue Date"], format='%d-%m-%Y',dayfirst=True)
    # Convert the 'Next Coupon Date' column to ensure consistent handling of NaT values
    bondData['Next Coupon Date'] = pd.to_datetime(bondData['Next Coupon Date'], format='%d-%m-%Y', errors='coerce')
    bondData['Next Coupon Date'] = bondData['Next Coupon Date'].replace(['None', 'NA', '', ' '], np.nan)
    bondData = bondData[bondData['Issue Date'] <= ref_date]
    return bondData

def couponCalculator(bond, bondData, today):
    """
    Calculate the coupon payment details for a given bond and bond data.
    """
    maturity = pd.to_datetime(bond["Maturity Date"], dayfirst=True)
    issue_date = pd.to_datetime(bond["Issue Date"], dayfirst=True)
    today = pd.Timestamp.today() if today is None else today
    coupon = bond['Coupon']

    cashflow_dates = []
    next_coupon_date = maturity
    while next_coupon_date >= issue_date:
        cashflow_dates.append(next_coupon_date)
        next_coupon_date -= pd.DateOffset(months=6)
    cashflow_dates = pd.DatetimeIndex(cashflow_dates)
    cashflow_dates = cashflow_dates[cashflow_dates >= today]

    # Convert to a DataFrame for better visualization
    cashflow_df = pd.DataFrame({
        'ISIN': bond['ISIN'],
        'Description': bond['Description'],
        'CF Date': cashflow_dates,
        'Dirty Price': bond['Dirty Price'],
        'Coupon': coupon,
        'Maturity Date': maturity,
        'Yield to Maturity': bond['Yield to Maturity'],
        'Ttm': bond['Ttm'],
        'Spot': None,
        'Z-spread': bond['Z-Spread']
    })

    # Add final coupon payment with face value at maturity
    cashflow_df.loc[cashflow_df['CF Date'] == maturity, 'Coupon'] = float(coupon) + 100
    cashflow_df['Ttm'] = ((cashflow_df['CF Date'] - today).dt.days / 365).round(3)
    cashflow_df['Spot'] = pd.to_numeric(cashflow_df['Spot'], errors='coerce')

    # Generate a random ISIN if missing
    if cashflow_df['ISIN'].isna().any():
        new_isin = 'IT' + ''.join([str(np.random.randint(0, 10)) for _ in range(10)])
        cashflow_df['ISIN'] = new_isin
    return cashflow_df

def get_next_futures_month(ref_date):
    """Finds the next futures contract month (March, June, September, December)."""
    month = ref_date.month
    if month <= 3:
        return 3
    elif month <= 6:
        return 6
    elif month <= 9:
        return 9
    else:
        return 12


def get_next_delivery_dates(year):
    """
    Determines all four futures delivery dates for a given year (March, June, September, December).
    """
    delivery_months = [3, 6, 9, 12]
    delivery_dates = []

    for month in delivery_months:
        delivery_date = datetime(year, month, 10)

        # Adjust to next business day if it's a weekend
        if delivery_date.weekday() >= 5:  # Saturday or Sunday
            delivery_date = (pd.Timestamp(delivery_date) + BDay(1)).to_pydatetime()

        delivery_dates.append(delivery_date)

    return delivery_dates


def extract_bond_dates(bond, delivery_date):
    """
    Extracts key dates for conversion factor calculation.
    """
    delivery_date = delivery_date
    maturity = pd.to_datetime(bond["Maturity Date"], dayfirst=True)
    coupon_freq = 6  # Semi-annual
    coupon = bond["Coupon"]

    # Find Next Coupon Date (NCD) after delivery date
    ncd = maturity
    while ncd >= delivery_date:
        ncd -= pd.DateOffset(months=coupon_freq)
    ncd += pd.DateOffset(months=coupon_freq)  # Step forward to get actual NCD

    # Last Coupon Date (LCD) is the previous coupon date
    lcd = ncd - pd.DateOffset(months=coupon_freq)

    # Previous coupon dates
    ncd_1cp = lcd
    ncd_2cp = ncd_1cp - pd.DateOffset(months=coupon_freq)

    # Compute time deltas
    δe = (ncd_1cp - delivery_date).days
    if δe < 0:
        act1 = (ncd - ncd_1cp).days
    else:
        act1 = (ncd_1cp - ncd_2cp).days
    δi = (ncd_1cp - lcd).days
    if δi < 0:
        act2 = (ncd - ncd_1cp).days
    else:
        act2 = (ncd_1cp - ncd_2cp).days

    f = 1 + ( δe / act1 )
        # First Coupon Adjustment
    first_coupon_adj = (coupon / (coupon * f )) * (δi / act2)
    accrued_interest = (coupon / (coupon * f)) * ((δi / act2) - (δe / act1))


    return {
        "LCD": lcd,
        "NCD": ncd,
        "NCD-1cp": ncd_1cp,
        "NCD-2cp": ncd_2cp,
        "δe": δe,
        "act₁": act1,
        "δᵢ": δi,
        "act₂": act2,
        "f": f,
        "f_c_adj": first_coupon_adj,
        "acc_int": accrued_interest
    }


def calculate_conversion_factor(bond, delivery_date, cashflows_df):
    """
    Calculates the conversion factor for a bond based on discounted cashflows.
    Double checked with one bond and it is correct to the 2nd decimal.
    """
    c = bond["Coupon"]
    f = bond['CF prep data']['f']
    acc_int = bond['CF prep data']['acc_int']
    f_c_adj = bond['CF prep data']['f_c_adj']
    cxf = c * f
    cn = 0.06 # Assumed standard notional coupon rate of 6%
    bond_cashflows = cashflows_df[cashflows_df["ISIN"] == bond["ISIN"]]

    # We assume for simplicity that coupon payment is always on a bus-day
    # Else, we should calculate the 'delay' variable correctly
    delay = 0
    cf_sum = 0
    t = 0
    logger.info(f"Starting to reprice the bond {bond['Description']}")
    for i in range(0, len(bond_cashflows)):
        cash_flow = bond_cashflows.iloc[i]['Coupon']
        power = t + f * 0.5
        b_cashflow = cash_flow / ((1 + (cn)) ** power)
        t += 0.5
        cf_sum += b_cashflow

    cf = cf_sum / 100
    #logger.info(f"Conversion factor is {cf}")
    return cf


def delivery_basket_calculator(bond_data, delivery_dates, year):
    """
    Filters deliverable bonds for each delivery date within the year.
    According to Eurex:https://www.eurex.com/ex-en/markets/int/fix/government-bonds/Long-Term-Euro-BTP-Futures-137382
    A delivery obligation arising out of a short position may only be fulfilled by the delivery
    of certain debt securities issued by the Federal Republic of Germany, the Republic of Italy,
     the Republic of France, the Kingdom of Spain or the Swiss Confederation with a remaining term
     on the delivery day within the remaining term of the underlying.
    Bonds qualify if their time to maturity is between 8.5 and 10.5 years ON THE DELIVERY DAY.

    Debt securities issued by the Republic of Italy must have an original term of no longer than 11 years
    (FBTS), 16 years (FBTM), and 17 years (FBTP).
    """
    all_bonds = []
    today = pd.Timestamp.today().normalize()

    # Also get the first delivery date of the next year
    next_year_delivery_dates = get_next_delivery_dates(year + 1)
    first_next_year_delivery_date = pd.Timestamp(next_year_delivery_dates[0])  # First delivery date of next year
    delivery_dates = delivery_dates + [first_next_year_delivery_date]

    prev_delivery_date = f"{year}-01-01"  # Start from January 1st initially

    for delivery_date in delivery_dates:
        del_date = pd.Timestamp(delivery_date)
        if delivery_date.year > year:
            del_date = datetime(delivery_date.year - 1, 12, 31)
        quarter_dates = pd.date_range(start=prev_delivery_date, end=del_date, freq='D')
        prev_delivery_date = del_date + timedelta(days=1)  # Update for the next iteration

        for curr_date in quarter_dates:
            # Coming out of the loop as calculating deliverable bonds in the future is not useful
            if curr_date == today:
                logger.info(f"Stopping all calculations at {curr_date} (Reached today's date)")
                deliverable_df = pd.DataFrame(all_bonds)  # Convert collected data into DataFrame
                return deliverable_df  # 🚀 Exit the entire function immediately
            if curr_date == '2025-01-01':
                breakpoint()

            bond_data["Maturity at delivery"] = (bond_data["Maturity Date"] - delivery_date).dt.days / 365.0
            # Given we're using end of year bond lists, we re-apply the issue-date filter so as to remove
            # bonds that were issued after the current date
            bond_data = bond_data[bond_data["Issue Date"] <= curr_date]
            rules = (
                    (bond_data["Maturity at delivery"] >= 8.5) &
                    (bond_data["Maturity at delivery"] <= 10.5) &
                    (bond_data["Amount Outstanding"] >= 5e9) &     # TODO: this needs to be historical, whilst it's currently fixed
                    (bond_data["Original Maturity"] <= 17)
            )

            filtered_bonds = bond_data[rules].copy()
            logger.info(f"{len(filtered_bonds)} bonds filtered")
            for isin in filtered_bonds['ISIN'].unique():
                all_bonds.append({"Date": curr_date, "Delivery Date": delivery_date, "ISIN": isin})

    deliverable_df = pd.DataFrame(all_bonds)
    return deliverable_df

def main():

    start_year = 2017
    end_year = 2025 # 8 years of historical data

    contract = 'IK'
    mon_sheet = f"{contract} Monitor"
    basis_data = pd.read_excel('Basis data.xlsx', sheet_name=mon_sheet, header=1)
    all_data = []

    deliverable = []

    # We loop through yeach year and extract the year's future delivery dates
    for year in range(start_year, end_year + 1):
        logger.info(f"Starting processing year {year}")
        delivery_dates = get_next_delivery_dates(year)
        logger.info(f"Delivery dates for the year are {delivery_dates}")

        # For each delivery date, for each day, we calculate the relevant conversion factors of each bond on the curve
        for delivery_date in delivery_dates:
            logger.info(f"Processing delivery date {delivery_date}")
            ref_date = delivery_date - BMonthEnd(2)
            logger.info(f"Reference date for calculations is is {ref_date}")
            monitor_data = basis_data.copy()
            mon_data = clean_monitor_data(monitor_data, ref_date)

            cashflow_list = mon_data.apply(lambda row: couponCalculator(row, mon_data, ref_date), axis=1)
            cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True).sort_values(by=['CF Date'])

            mon_data["CF prep data"] = mon_data.apply(lambda row: extract_bond_dates(row, delivery_date), axis=1)
            mon_data["Conversion Factor"] = mon_data.apply(lambda row: calculate_conversion_factor(row, delivery_date, cashflows_df), axis=1)

            # Create a DataFrame for the current delivery date and append to all_data list
            current_cf_monitor = mon_data[["ISIN", "Conversion Factor"]].copy()
            current_cf_monitor["Delivery Date"] = delivery_date  # Add the delivery date as a new column

            all_data.append(current_cf_monitor)  # Append the DataFrame instead of looping row by row

        # For each year, we also define the deliverable ISINs for each delivery date, positioning ourselves before each delivery
        logger.info(f"Finished processing year {year}")
        deliverable_ = delivery_basket_calculator(mon_data, delivery_dates, year)
        logger.info((f"Finished processing deliverable ISINs for year {year}"))
        deliverable.append(deliverable_)
    cf_df = pd.concat(all_data, ignore_index=True)
    deliverable_df = pd.concat(deliverable, ignore_index=True)
    # We filter the conversion factor df for it to include only the deliverable ISINs
    cf_df = cf_df[cf_df['ISIN'].isin(deliverable_df['ISIN'])]
    cf_df.to_csv(f"conversion_factors_{contract}.csv", index=False)
    deliverable_df.to_csv(f"Deliverables_{contract}.csv", index=False)
    # When merging conversion factors into deliverable ISINs,
    cf_hist = deliverable_df.merge(cf_df, on=['ISIN', 'Delivery Date'], how='left')
    cf_hist.to_csv(f"DeliverableBondsHistory_{contract}.csv", index=False)

if __name__ == "__main__":
    main()
