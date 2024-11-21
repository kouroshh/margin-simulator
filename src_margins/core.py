import pandas as pd

import pyarrow as pa

import numpy as np


"""

    Reads an Apache Arrow file from the given path, handles dictionary-encoded columns with null values,

    and converts the Arrow table to a pandas DataFrame for further use in data processing or analysis.

 

    Args:

    - path (str): The file system path where the Arrow file is stored.

 

    Returns:

    - pandas.DataFrame: A DataFrame representation of the data contained in the Arrow file.

 

    Usage:

    - This function is particularly useful for efficiently handling large datasets stored in Apache Arrow format,

      especially those that utilize dictionary encoding to optimize for both space and performance. The function

      ensures that any null values in dictionary-encoded columns are handled gracefully by replacing them with

      a default string 'Missing' to maintain data integrity.

 

    Example:

    >>> arrow_data_path = '/path/to/your/data.arrow'

    >>> df = read_arrow(arrow_data_path)

    >>> print(df.head())

 

    Notes:

    - The function first attempts to read the Arrow file as a stream; if that fails, it tries to read it as a file.

    - For dictionary-encoded columns, if the dictionary has null values, those nulls are replaced with the string

      'Missing'. This is done by converting the dictionary to a pandas Series, filling null values, and then

      converting it back to an Arrow array.

    - The function ensures that all columns, including those corrected for null values, are correctly reassembled

      into a new Arrow table which is then converted to a DataFrame.

    - This function requires the 'pyarrow' library for reading Arrow files and 'pandas' for handling data in memory.

    """ 
def read_arrow(path):      

    with open(path, 'rb') as f:

        try:

            # Attempt to read the file as a stream

            table = pa.ipc.open_stream(f).read_all()

        except pa.lib.ArrowInvalid:

            # If stream fails, reset file pointer and try reading as a file

            f.seek(0)

            table = pa.ipc.open_file(f).read_all()

 

    # Process each column that is dictionary-encoded

    new_columns = []

    for i, field in enumerate(table.schema):

        column = table.column(i)

        if pa.types.is_dictionary(field.type):

            new_chunks = []

            for chunk in column.chunks:

                dictionary = chunk.dictionary

                if dictionary.null_count > 0:

                    # Convert to Pandas series to utilize fillna

                    temp_series = pd.Series(dictionary.to_pandas())

                    temp_series = temp_series.fillna('Missing')

                    # Convert back to PyArrow Array

                    cleaned_dict = pa.array(temp_series, type=pa.string())

                    new_chunk = pa.DictionaryArray.from_arrays(chunk.indices, cleaned_dict)

                else:

                    new_chunk = chunk

                new_chunks.append(new_chunk)

            new_column = pa.chunked_array(new_chunks, type=column.type)

            new_columns.append(new_column)

        else:

            new_columns.append(column)

 

    # Replace old columns with new ones in the table

    new_table = pa.Table.from_arrays(new_columns, schema=table.schema)

 

    # Convert the cleaned Arrow Table to a pandas DataFrame

    df = new_table.to_pandas()

 

    return df


def enrich_portfolio(portfolio, rf04):

    """

    Enriches the portfolio DataFrame by merging it with reference data (rf04)

    and calculates the position value for each asset.

   

    Parameters:

        portfolio (pd.DataFrame): The portfolio data containing the following columns:

            - 'isin': Unique identifier for securities.

            - 'qty': Quantity held in the portfolio (positive or negative).

            - 'mult': Multiplier for the instrument (e.g., lot size or contract size).

            - 'price': Current market price of the instrument.

        rf04 (pd.DataFrame): Reference data to be merged with the portfolio, containing:

            - 'isin': Unique identifier for securities.

            - Additional columns for enrichment.

   

    Returns:

        pd.DataFrame: Enriched portfolio DataFrame with an additional column:

            - 'pos_value': Absolute position value calculated as:

              abs(qty) * mult * price

    """

    df = portfolio.merge(rf04, on='isin', how='left')

    df['pos_value'] = abs(df['qty']) * df['mult'] * df['price']

    return df

 

def mtm(enr_port, rf03):

    """

    Calculates the mark-to-market (MTM) for a portfolio based on current exchange rates,

    separating options and cash-like instruments from futures.

   

    Parameters:

        enr_port (pd.DataFrame): The enriched portfolio DataFrame containing:

            - 'prod_curcy': Product currency of the instrument.

            - 'asset_type': Type of the asset ('C' for cash, 'O' for options, etc.).

            - 'qty': Quantity held.

            - 'price': Current price of the instrument.

            - 'mult': Multiplier for the instrument.

            - 'ctv': Current value (used for cash instruments).

        rf03 (pd.DataFrame): Exchange rate data containing:

            - 'scenario': Scenario identifier (e.g., 'C' for current scenario).

            - 'base_curcy': Base currency for the exchange rate.

            - Additional columns if needed.

   

    Returns:

        tuple:

            - pd.DataFrame: Filtered DataFrame of non-futures instruments with an additional column:

                - 'mtm': Mark-to-market value for each row, calculated as:

                  - For cash-like instruments ('C'):

                    qty * price - ctv

                  - For other non-futures instruments:

                    qty * price * mult

            - float: Total MTM value (sum of all MTM values in the filtered DataFrame).

    """

   

    current_exc_rate = rf03[rf03['scenario'] == 'C']

    df = enr_port.merge(current_exc_rate, left_on='prod_curcy', right_on='base_curcy', how='left')

    df_opt_cash = df[df['asset_type'] != 'F']

    df_opt_cash['mtm'] = np.where(df_opt_cash['asset_type'] == 'C',

                                  df_opt_cash['qty'] * df_opt_cash['price'] - df_opt_cash['ctv'],

                                  df_opt_cash['qty'] * df_opt_cash['price'] * df_opt_cash['mult'])

    return df_opt_cash, df_opt_cash.mtm.sum()

   

def expected_shortfall(portfolio, rf01, rf02, rf03, rf04):

    """

        Calculates the Expected Shortfall (ES) for a portfolio under both ordinary and stressed scenarios,

        including diversified and undiversified ES measures. The function also computes add-ons for

        decorrelation and initial margin requirements.

    

        Parameters:

            portfolio (pd.DataFrame): The portfolio data, expected to contain:

                - 'isin': Unique identifier for instruments.

                - 'qty': Quantity held.

                - 'mult': Multiplier for instruments (e.g., lot size or contract size).

                - 'price': Current market price.

                - 'prod_curcy': Product currency.

                - 'asset_type': Type of the instrument (e.g., 'C', 'O', 'F').

            rf01 (pd.DataFrame or dict): Configuration parameters for ES calculation, containing:

                - 'ordinary_cl': Confidence level for ordinary scenarios.

                - 'stressed_cl': Confidence level for stressed scenarios.

                - 'deco': Decorrelation factor.

                - 'ordinary_w': Weight for ordinary scenarios.

                - 'stressed_w': Weight for stressed scenarios.

            rf02 (pd.DataFrame): Scenario data for instruments, expected to contain:

                - 'isin': Unique identifier for instruments.

                - 'prod_curcy': Product currency.

                - 'value': Scenario value for instruments.

                - 'scenario': Scenario identifier ('C', 'S', 'U').

            rf03 (pd.DataFrame): FX scenario data, expected to contain:

                - 'ref_dt': Reference date.

                - 'scenario': Scenario identifier ('C', 'S', 'U').

                - 'base_curcy': Base currency.

                - 'value': FX scenario value.

            rf04 (pd.DataFrame): Reference data for enriching the portfolio, containing:

                - 'isin': Unique identifier for instruments.

    

        Returns:

            tuple: A tuple containing:

                - port_scen_with_c (pd.DataFrame): Portfolio with all scenarios applied, including P&L calculations.

                - pnl_s (pd.DataFrame): P&L results for ordinary scenarios ('S').

                - pnl_u (pd.DataFrame): P&L results for stressed scenarios ('U').

                - output (pd.DataFrame): Final summary containing:

                    - 'ord_div_ES': Diversified ordinary ES.

                    - 'str_div_ES': Diversified stressed ES.

                    - 'ord_undiv_ES': Undiversified ordinary ES.

                    - 'str_undiv_ES': Undiversified stressed ES.

                    - 'ord_deco_addon': Decorrelation add-on for ordinary ES.

                    - 'str_deco_addon': Decorrelation add-on for stressed ES.

                    - 'whatif': Worst-case margin scenario.

                    - 'mtm': Total mark-to-market value.

                    - 'initial_margin': Initial margin requirement.

                    - 'gross_pos_value': Gross position value.

                    - 'margin_%': Margin as a percentage of the gross position value.

    

        Detailed Steps:

            1. Enrich the portfolio using the `enrich_portfolio` function to include position values.

            2. Filter the scenario data for instruments and currencies in the portfolio.

            3. Merge portfolio data with scenario values and FX scenarios.

            4. Extract baseline values ('C' scenario) and calculate P&L for 'S' (ordinary) and 'U' (stressed) scenarios:

                - For options and cash ('C', 'O'), P&L = (scenario value - baseline value) × qty × mult.

                - For futures ('F'), P&L = (scenario value - baseline value) × FX rate × qty × mult.

            5. Calculate diversified and undiversified ES for both ordinary and stressed scenarios.

            6. Compute decorrelation add-ons for ordinary and stressed ES.

            7. Determine the "what-if" margin scenario, total MTM value, initial margin, and margin percentage.

    

        Notes:

        - Ordinary and stressed confidence levels are used to determine the number of observations

          for calculating ES (based on the smallest P&L values).

        - Decorrelation factors are applied to adjust for diversification effects.

        - The function integrates MTM calculation via the `mtm` function.

   

    """

   

    

    enr_port = enrich_portfolio(portfolio, rf04)

 

    # Filtra strumenti e valute dal portafoglio

    instruments = set(enr_port['isin'])

    fx = set(enr_port['prod_curcy'])

 

    # Filtra scenari per strumenti e valute

    es_scen = rf02[rf02['isin'].isin(instruments)]

    fx_scen = rf03[rf03['base_curcy'].isin(fx)].rename(columns={

        'base_curcy': 'prod_curcy',

        'value': 'fx_scenario'

    })

 

    # Merge del portafoglio con scenari e FX

    port_scen = enr_port.merge(es_scen, on=['isin', 'prod_curcy'], how='left')

    port_scen_fx = port_scen.merge(

        fx_scen[['ref_dt', 'scenario', 'prod_curcy', 'fx_scenario']],

        on=['ref_dt', 'prod_curcy', 'scenario'],

        how='left'

    )

 

    # Estrai i valori di riferimento (scenario "C")

    scenario_c = port_scen_fx[port_scen_fx['scenario'] == 'C'].copy()

    scenario_c = scenario_c[['isin', 'prod_curcy', 'ref_dt', 'value', 'fx_scenario']].rename(columns={

        'value': 'last_price',

        'fx_scenario': 'last_fx'

    })

 

    # Unisci i valori "C" con il dataset completo

    port_scen_with_c = port_scen_fx[port_scen_fx['scenario'] != 'C'].merge(

        scenario_c[['isin', 'prod_curcy', 'last_price', 'last_fx']],

        on=['isin', 'prod_curcy'],

        how='left'

    )

 

    # Calcola P&L per scenari "S" e "U"

    port_scen_with_c.loc[port_scen_with_c['asset_type'].isin(['C', 'O']), 'P&L'] = (

        (port_scen_with_c['value'] * port_scen_with_c['fx_scenario'] -

         port_scen_with_c['last_price'] * port_scen_with_c['last_fx']) *

        port_scen_with_c['qty'] * port_scen_with_c['mult']

    )

    port_scen_with_c.loc[port_scen_with_c['asset_type'] == 'F', 'P&L'] = (

        (port_scen_with_c['value'] - port_scen_with_c['last_price']) *

        port_scen_with_c['fx_scenario'] *

        port_scen_with_c['qty'] *

        port_scen_with_c['mult']

    )

 

    # Separa i risultati per scenario "S" e "U"

    pnl_s = port_scen_with_c[port_scen_with_c['scenario'] == 'S'].copy()

    pnl_u = port_scen_with_c[port_scen_with_c['scenario'] == 'U'].copy()

 

    # Calcola il numero di osservazioni ordinarie e stressate

    ordinary_cl = float(rf01['ordinary_cl'].iloc[0]) if isinstance(rf01, pd.DataFrame) else rf01['ordinary_cl']

    stressed_cl = float(rf01['stressed_cl'].iloc[0]) if isinstance(rf01, pd.DataFrame) else rf01['stressed_cl']

 

    ord_observations = len(set(pnl_s['ref_dt']))

    ord_observations = int(max(1, np.floor(ord_observations - ordinary_cl * ord_observations)))

 

    stress_observations = len(set(pnl_u['ref_dt']))

    stress_observations = int(max(1, np.floor(stress_observations - stressed_cl * stress_observations)))

 

    # Diversified ES

    ord_div_ES_pnl = pnl_s.groupby('ref_dt').agg(port_pnl=('P&L', 'sum'))

    stress_div_ES_pnl = pnl_u.groupby('ref_dt').agg(port_pnl=('P&L', 'sum'))

 

    ord_div_ES = ord_div_ES_pnl['port_pnl'].nsmallest(ord_observations).mean()

    stress_div_ES = stress_div_ES_pnl['port_pnl'].nsmallest(stress_observations).mean()

 

    # Undiversified ES

    ord_und_ES_pnl = pnl_s.groupby(['ref_dt', 'und_isin']).agg(pnl=('P&L', 'sum')).reset_index()

    ord_und_ES_pnl_app = ord_und_ES_pnl.groupby('und_isin').agg(undiv_pnl=('pnl', lambda x: x.nsmallest(ord_observations).mean()))

    ord_und_ES = ord_und_ES_pnl_app['undiv_pnl'].sum()

 

    stress_und_ES_pnl = pnl_u.groupby(['ref_dt', 'und_isin']).agg(pnl=('P&L', 'sum')).reset_index()

    stress_und_ES_pnl_app = stress_und_ES_pnl.groupby('und_isin').agg(undiv_pnl=('pnl', lambda x: x.nsmallest(stress_observations).mean()))

    stress_und_ES = stress_und_ES_pnl_app['undiv_pnl'].sum()

 

    # Calcolo del deco addon

    deco = float(rf01['deco'].iloc[0]) if isinstance(rf01, pd.DataFrame) else rf01['deco']

    ord_deco_addon = (1 - deco) * (ord_und_ES - ord_div_ES)

    stress_deco_addon = (1 - deco) * (stress_und_ES - stress_div_ES)

   

    output = pd.DataFrame({'ord_div_ES':[abs(ord_div_ES)],

                           'str_div_ES':[abs(stress_div_ES)],

                           'ord_undiv_ES':[abs(ord_und_ES)],

                           'str_undiv_ES':[abs(stress_und_ES)],

                           'ord_deco_addon':[abs(ord_deco_addon)],

                           'str_deco_addon':[abs(stress_deco_addon)]})

    ord_weight = float(rf01['ordinary_w'].iloc[0]) if isinstance(rf01, pd.DataFrame) else rf01['ordinary_w']

    str_weight = float(rf01['stressed_w'].iloc[0]) if isinstance(rf01, pd.DataFrame) else rf01['stressed_w']

   

    output['whatif'] = np.absolute(np.maximum(ord_weight*(output['ord_div_ES'] + output['ord_deco_addon']) + str_weight * (output['str_div_ES'] + output['str_deco_addon']), (output['ord_div_ES']+output['ord_deco_addon'])))

 

    mtm_details, mtm_total = mtm(enr_port, rf03)

   

    output['mtm'] = mtm_total

    output['initial_margin'] = np.maximum(0, output['mtm'] + output['whatif'])

    output['gross_pos_value'] = enr_port['pos_value'].sum()

    output['margin_%'] = float(output['initial_margin'] / output['gross_pos_value'])

 

    return port_scen_with_c, pnl_s, pnl_u, output

 

 

# Input Paths (for large file use local resoruces)

path = r'C:\Users\GTeodori\margin-simulator\input_data'

 

# Load Data Files

rf01 = read_arrow(path + '\\2024-11-20_RF01.arrow')

rf02 = read_arrow(path + '\\2024-11-20_RF02.arrow')

rf03 = read_arrow(path + '\\2024-11-20_RF03.arrow')

rf04 = read_arrow(path + '\\2024-11-20_RF04.arrow')

 

# dummy portfolio

portfolio = pd.DataFrame({'isin':['FREN02742905', 'FRENX7284263'],

                          'pos_type':['S', 'S'],

                          'qty':[-1, 1],

                          'ctv':[0, 0]})

 

 

# enr_port = enrich_portfolio(portfolio, rf04)

# mtm_details, mtm_total = mtm(enr_port, rf03)

port_scen_with_c, pnl_s, pnl_u, output = expected_shortfall(portfolio, rf01, rf02, rf03, rf04)

# print(port_scen_with_c)
# print(pnl_s)
# print(pnl_u)
print(output)