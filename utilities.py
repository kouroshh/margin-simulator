from flask import Flask, request, jsonify
import pyarrow as pa
import pandas as pd

app = Flask(__name__)

def convert_to_dataframe(json_data):
    # Create empty lists to store data
    portfolio_nb = []
    isin = []
    prod_curcy = []
    qty = []
    trade_price = []

    # Loop through portfolios in the JSON data
    for portfolio in json_data['portfolios']:
        portfolio_title = portfolio['title']  # Portfolio title
        
        # Loop through positions in each portfolio
        for position in portfolio['positions']:
            portfolio_nb.append(portfolio_title)
            isin.append(position['isin'])
            prod_curcy.append(position['currency'])
            qty.append(int(position['quantity']))  # Convert quantity to int
            trade_price.append(int(position['tradingPrice']))  # Trading price is directly added

    # Convert the lists into a DataFrame
    df = pd.DataFrame({
        'portfolio_nb': portfolio_nb,
        'isin': isin,
        'prod_curcy': prod_curcy,
        'qty': qty,
        'trade_price': trade_price
    })
    
    return df


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

if __name__ == '__main__':
    
  path = r'C:\margin-simulator\margin-simulator\input_data'
  df = read_arrow(path + '\\2024-11-20_RF04.arrow')
  #top_50_rows = df[df.iloc[:, 0].duplicated(keep=False)]
  top_50_rows = df.head(10)
  top_50_rows.to_csv("isins.csv", index=False)
  print(f"Saved top 50 rows to {df}")