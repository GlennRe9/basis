import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_net_basis(basis: pd.DataFrame, price_df: pd.DataFrame, forecasts: pd.DataFrame = None) -> pd.DataFrame:
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