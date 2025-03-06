import pandas as pd
import logging
from bond_calcs import generate_cashflows
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_net_basis(basis: pd.DataFrame, price_df: pd.DataFrame = None, forecasts: pd.DataFrame = None) -> pd.DataFrame:
    """
    Computes the net basis for each bond in the delivery basket.

    Parameters:
    - basis (pd.DataFrame): Bond data, including conversion factors and carry-to-delivery.
    - price_df (pd.DataFrame): Computed bond prices under different yield scenarios.

    Returns:
    - pd.DataFrame: Net basis for each bond.
    """

    # If forecasts are provided, replace price_df with forecasted prices
    if forecasts is not None:
        price_df = forecasts[['maturityID','Fwd_Px_At_Delivery']].set_index('maturityID')

    # Potentially add carry in order to get dirty price
    # Creating df of converted prices net of carry to delivery
    # The bonds that have the lowest converted fwd price are also the cheapest to deliver
    cf_df = price_df.copy()
    for bond in cf_df.index:
        if bond in basis['maturityID'].values:  # Check if the bond exists in the basis_df
            cf = basis.loc[basis['maturityID'] == bond, 'Conv Factor'].values[0]
            income_to_del = basis.loc[basis['maturityID'] == bond, 'Carry to delivery'].values[0] if forecasts is None else 0
            cf_df.loc[bond] = (price_df.loc[bond] + income_to_del) / cf

    price_df = compute_futures_price(basis, cf_df, price_df)


    net_basis = price_df.copy()  # Start with price_df copy

    for bond in net_basis.index[:-1]:  # Exclude 'Future Price' row
        future_prices = net_basis.loc['Future Price']  # Extract future price row
        conv_factor = basis.loc[basis['maturityID'] == bond, 'Conv Factor'].values[0]
        carry_to_delivery = basis.loc[basis['maturityID'] == bond, 'Carry to delivery'].values[0]
        bond_price = price_df.loc[bond]

        # Apply the correct net basis formula
        net_basis.loc[bond] = bond_price - carry_to_delivery - (conv_factor * future_prices)

    return net_basis.round(3)


def prep_basis_calc(curr_date, today, bondData_, long_bond_hist, deliverable):
    bondData_ = bondData_[bondData_['Issue Date'] <= curr_date]
    logger.info(f"{len(bondData_)} were available on {curr_date}")
    curr_date_str = curr_date.strftime('%Y-%m-%d')
    hist_data = long_bond_hist[long_bond_hist['DATE'] == curr_date][['ISIN', 'Yield', 'Dirty Price', 'Repo Rate']]
    # rename Dirty Price to 'historical dirty price'
    hist_data.rename(columns={'Dirty Price': 'Historical Dirty Price', 'Repo Rate': 'Historical Repo Rate'}, inplace=True)

    # Merge historical yields and prices into bondData
    n_yields = len(bondData_['Yield to Maturity'])
    bondData_ = bondData_.merge(hist_data[['ISIN', 'Yield', 'Historical Dirty Price', 'Historical Repo Rate']], on='ISIN', how='left')
    bondData_['Yield to Maturity'] = bondData_['Yield']
    bondData_['Dirty Price'] = bondData_['Historical Dirty Price']
    bondData_['Repo Rate'] = bondData_['Historical Repo Rate']
    bondData_.drop(columns=['Yield', 'Historical Dirty Price', 'Historical Repo Rate'], inplace=True)

    n_yields_new = len(bondData_['Yield to Maturity'])
    logger.info(f"We have {n_yields} yields before and {n_yields_new} after merging historical yields.")

    # **STEP 2: Find the next available delivery date**
    next_delivery = None
    # **STEP 2: Find the latest available delivery date**
    search_date = curr_date

    while search_date not in deliverable['DATE'].values:
        logger.info(f"No delivery date found for {search_date}, trying previous day.")
        search_date -= pd.Timedelta(days=1)  # Move one day backward

    # **STEP 3: Filter deliverable bonds**
    deliverable = deliverable[deliverable['DATE'] == search_date].copy()
    next_delivery = deliverable['Delivery Date'].values[0]

    bondData_deliv = bondData_[bondData_['ISIN'].isin(deliverable['ISIN'])]
    bondData_deliv = bondData_deliv.merge(deliverable[['ISIN', 'Conversion Factor']], on='ISIN', how='left')
    bondData_deliv = bondData_deliv.drop(columns=['Coupon Frequency', 'Price Accrued Interest Flag', 'Z-Spread'])
    # Carry calculations
    days_to_delivery = ( next_delivery - today ).days
    bondData_deliv['Income to delivery'] = ( bondData_deliv['Coupon'] * days_to_delivery / 360)
    bondData_deliv['Cost to delivery'] = bondData_deliv['Dirty Price'] * (bondData_deliv['Repo Rate'].div(100)) * ( days_to_delivery / 360 )
    bondData_deliv['Carry to delivery'] = bondData_deliv['Income to delivery'] - bondData_deliv['Cost to delivery']

    # Get cashflow list
    # Generate cashflows **before** yield forecasting
    cashflow_list = bondData_.apply(lambda bond: generate_cashflows(bond, today), axis=1)
    cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True)

    return bondData_, bondData_deliv, next_delivery, cashflows_df

def compute_hist_FV_NB(basis) -> pd.DataFrame:
    """
    Computes the Fair Value net basis for each bond in the delivery baskets
    Returns:
    - pd.DataFrame: Net basis for each bond.
    """

    dov = 0
    # We add income to delivery to the clean price and then divide by conversion factor
    price_df = basis.copy()
    price_df['Converted Price'] = price_df['Fwd_Px_At_Delivery'] / price_df['Conversion Factor']

    ctd_index = price_df['Converted Price'].idxmin()
    logger.info(f"Cheapest to deliver bond is expected to be {ctd_index}")
    # Extract values for CTD bond
    ctd_px = price_df.loc[ctd_index, 'Fwd_Px_At_Delivery']
    carry_to_delivery = price_df.loc[ctd_index, 'Carry to delivery']
    ctd_cf = price_df.loc[ctd_index, 'Conversion Factor']
    f_price = (ctd_px - carry_to_delivery - dov) / ctd_cf

    net_basis = price_df.copy()

    net_basis['FV Net Basis'] = net_basis['Fwd_Clean_Px_At_Delivery'] - net_basis['Carry to delivery'] - (net_basis['Conversion Factor'] * f_price)

    return net_basis[['DATE', 'RIC', 'ISIN', 'Description', 'Maturity Date', 'Repo Rate', 'Conversion Factor', 'Carry to delivery', 'Fwd_Px_At_Delivery', 'Fwd_YTM', 'Fwd_Clean_Px_At_Delivery', 'Converted Price', 'FV Net Basis']].round(3)

def compute_futures_price(basis: pd.DataFrame, cf_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes futures prices based on Cheapest-to-Deliver (CTD) bond selection.

    Parameters:
    - basis (pd.DataFrame): Bond data including conversion factors.
    - cf_df (pd.DataFrame): Adjusted prices net of carry to delivery.
    - price_df (pd.DataFrame): Computed bond prices under different yield scenarios.

    Returns:
    - pd.DataFrame: Updated `price_df` with futures prices added.
    """
    # DOV is our estimate of the value of end-of-month switch option - at maturity, this is correct! NB will be 0

    dov = 0  # End-of-month switch option value (assumed 0 for now)
    price_df1 = price_df.copy()

    for bump in cf_df.columns:
        # Find the CTD bond
        ctd = cf_df[bump].idxmin()
        ctd_px = price_df.loc[ctd, bump]
        logging.info(f'The CTD bond in scenario of yields  {bump} is {ctd} with price {ctd_px}')

        # Retrieve conversion factor and carry to delivery
        ctd_cf = basis.loc[basis['maturityID'] == ctd, 'Conv Factor'].values[0]
        carry_to_delivery = basis.loc[basis['maturityID'] == ctd, 'Carry to delivery'].values[0]

        # Compute futures price
        f_price = (ctd_px - carry_to_delivery - dov) / ctd_cf
        price_df.at['Future Price', bump] = f_price
        price_df1.at['Future Price', bump] = f_price

    return price_df