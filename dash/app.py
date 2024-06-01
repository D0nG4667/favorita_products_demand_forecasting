from dash import Dash, html, dcc, callback, Output, Input

# Visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly_calplot import calplot

# Data handling
import polars as pl
import polars.selectors as cs
from itertools import product, combinations

app = Dash()

header = html.H1(children='Favorita Dashboard', style={'textAlign': 'center'})

def to_date(df: pl.DataFrame, date_col: str = 'date') -> pl.DataFrame:
    """
    Convert a specified column of a polars DataFrame to a date type and sort the DataFrame by this column.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame containing the date column.
    date_col (str): The name of the column to convert and sort by.

    Returns:
    pl.DataFrame: A new polars DataFrame with the date column converted to date type and sorted.
    """
    # Convert date column to date type
    df = df.with_columns(df[date_col].cast(pl.Date))
    
    # Sort the dataframe using the date column
    df = df.sort(date_col)
        
    return df

train_df = pl.read_csv(
    'https://media.githubusercontent.com/media/D0nG4667/favorita_products_demand_forecasting/main/data/source/github/train.csv')

train_df = to_date(train_df)


plot_data = train_df.group_by_dynamic(
    'date', every='1d').agg(pl.col('sales').sum())
plot_data = plot_data.to_pandas()
plot_data['date'] = plot_data['date'].astype('datetime64[ns]')

graph = dcc.Graph(
    figure =
        calplot(
            plot_data,
            x='date',
            y='sales',
            years_title=True,
            colorscale='YlGn',
            showscale=True,
            title='Sales by calendar days, months, and years'
        )
)

app.layout = [
    header,
    graph
]

if __name__ == '__main__':
    app.run(debug=True)
