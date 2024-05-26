<a name="readme-top"></a>

# Favorita Product Demand Forecasting

A machine learning project with multivariate time series to forecast sales, built with Python and scikit-learn, designed for corporation favorita and retail companies aiming to forecast sales and optimize product inventory.

![Python Version](https://img.shields.io/badge/Python-3.11-blue)
![Data Analysis](https://img.shields.io/badge/Data-Analysis-blue)
![Data Visualization](https://img.shields.io/badge/Data-Visualization-blue)
![Hypothesis Testing](https://img.shields.io/badge/Hypothesis-Testing-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue)
![Medium Article](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)


## Overview

Corporation Favorita aims to optimize its inventory management by accurately forecasting the demand for various products across its stores in Ecuador. The goal is to ensure that each store has the right quantity of products in stock to meet customer demand while minimizing overstocking or stockouts.

## Key Objectives

The objective is to build machine learning models that can predict unit sales for different product families at Favorita stores accurately. These models will help optimize inventory levels, improve sales forecasting accuracy, and ultimately enhance customer satisfaction by ensuring product availability.

## Framework

The CRoss Industry Standard Process for Data Mining (CRISP-DM).

## Features

- Jupyter Notebook containing data analysis, visualizations, and interpretation.
- Detailed documentation outlining methodology, data sources, and analysis results.
- Interactive visualizations in Power BI and Dash showcasing sales trends and key insights.

### Dash app

- [Favorita Product Demand Forecasting](https://favorita.gabcares.xyz)

### PowerBI Dashboard

![Dashboard](/screenshots/dashboard.png)

## Technologies Used

- Anaconda
- PowerBI
- Python
- Polars
- Dash
- Plotly
- plotly_resampler
- Jupyter Notebooks
- Git
- Scipy
- Sklearn
- Xgboost
- LinearRegression
- HistGradientBoostingRegressor
- Prophet
- Pyodbc
- Re
- Typing
- Kaleido

## Installation

### Recommended install

```bash
conda env create -f favorita_env.yml
```

## Code Snippet- used to create date features

```python
def create_date_features(df: pl.DataFrame)-> pl.DataFrame:
    # To date
    df = to_date(df, 'date')
    
    # Year
    # Extracting the year from the date and adding it to the df as a new column
    df = df.with_columns(pl.col('date').dt.year().alias('year'))
    
    # Extracting the ordinal day from the date and adding it to the df as a new column
    df = df.with_columns(pl.col('date').dt.ordinal_day().alias('day_of_year'))
    
    # Extracting the year start bool from the date and adding it to the df as a new column
    df = df.with_columns((pl.col('day_of_year')==1).alias('is_year_start'))
    
    # Extracting the year end bool from the date and adding it to the df as a new column
    df = df.with_columns((pl.col('day_of_year')==31).alias('is_year_end'))
    
    # Month
    # Extracting the month from the date and adding it to the df as a new column (1-12)
    df = df.with_columns(pl.col('date').dt.month().alias('month'))
    
    # Extracting the month start bool from the date and adding it to the df as a new column
    df = df.with_columns((pl.col('date')==pl.col('date').dt.month_start()).alias('is_month_start'))
    
    # Extracting the month end bool from the date and adding it to the df as a new column
    df = df.with_columns((pl.col('date')==pl.col('date').dt.month_end()).alias('is_month_end'))
    
    # Extracting the month name from the date and adding it to the df as a new column
    df = df.with_columns(pl.col('date').dt.to_string('%B').alias('month_name')) 
    
    # Day
    # Extracting the day from the date and adding it to the df as a new column
    df = df.with_columns(pl.col('date').dt.day().alias('day'))           
    
    # Extracting the day name from the date and adding it to the df as a new column
    df = df.with_columns(pl.col('date').dt.to_string('%A').alias('day_name'))
    
    # Quarter
    # Extracting the quarter from the date and adding it to the df as a new column (1-4)
    df = df.with_columns(pl.col('date').dt.quarter().alias('quarter'))
    
    # Extracting the quarter start bool from the date and adding it to the df as a new column
    df = df.with_columns((((((pl.col('quarter')-1)*3)+1)==pl.col('month')) & (pl.col('is_month_start'))).alias('is_quarter_start'))
    
    # Extracting the quarter end bool from the date and adding it to the df as a new column
    df = df.with_columns((((pl.col('quarter')*3)==pl.col('month')) & (pl.col('is_month_end'))).alias('is_quarter_end'))
    
    # Week
    # Extracting the week from the date and adding it to the df as a new column (1-52)
    df = df.with_columns(pl.col('date').dt.week().alias('week'))
    
    # Extracting the week day from the date and adding it to the df as a new column (1-7)
    df = df.with_columns(pl.col('date').dt.weekday().alias('week_day'))
    
    # Extracting the weekend bool from the date and adding it to the df as a new column
    df = df.with_columns(pl.col('week_day').is_in([6,7]).alias('is_week_end'))         
    
    return df

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Code Snippet- used to create lag features

```python
def create_lag_features(df: pl.DataFrame, lag_days: List[int] = [1, 6, 7], rows_per_day: int = 1782, lag_factor: int = 91) -> pl.DataFrame:
    # Initialize a list to hold the lag columns
    lag_columns = []

    # Iterate to create each lag feature, fill nulls with zero since they are not known or are probably zero if the first date is the inception
    for i in lag_days:
        lag_col = df['sales'].shift(i*rows_per_day*lag_factor).alias(f'lag_{i}').fill_null(strategy='zero')
        lag_columns.append(lag_col)

    # Combine the original df with the lag columns. Drop rows with null values in lag columns if they were not filled, probably redundant since they were filled
    lagged_df = df.with_columns(lag_columns).drop_nulls([col.name for col in lag_columns])

    return lagged_df
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Code Snippet- used to create all features

```python
def create_all_features(df):
    if not isinstance(df, pl.DataFrame): # Handle arrays inputs from pipeline
        schema = ['id', 'store_nbr', 'family', 'sales', 'onpromotion', 'city', 'state', 'store_type', 'cluster', 
              'promotion', 'holiday', 'year', 'day_of_year', 'is_year_start', 'is_year_end', 'month', 'is_month_start', 
              'is_month_end', 'month_name', 'day', 'day_name', 'quarter', 'is_quarter_start', 'is_quarter_end', 'week', 
              'week_day', 'is_week_end', 'lag_1', 'lag_6', 'lag_7', 'oil_price']
    
        df = pl.DataFrame(data=df, schema=schema)
        
    return column_dropper(
        create_oil_feature(
            create_lag_features(
                create_date_features(
                    create_holiday_feature(
                        create_promotion_feature(
                            create_store_features(fix_col_types(df))
                        )
                    )
                )
            )
        )
    )
    
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributions

### How to Contribute

1. Fork the repository and clone it to your local machine.
2. Explore the Jupyter Notebooks and documentation.
3. Implement enhancements, fix bugs, or propose new features.
4. Submit a pull request with your changes, ensuring clear descriptions and documentation.
5. Participate in discussions, provide feedback, and collaborate with the community.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Feedback and Support

Feedback, suggestions, and contributions are welcome! Feel free to open an issue for bug reports, feature requests, or general inquiries. For additional support or questions, you can connect with me on [LinkedIn](https://www.linkedin.com/in/dr-gabriel-okundaye).

Link to article on Medium: [From Meeting Product Demands to transforming Inventory Management: Forecasting Sales in Ecuador's Leading Retailer, Corporación Favorita.](https://medium.com/@gabriel007okuns/from-meeting-product-demands-to-transforming-inventory-management-forecasting-sales-in-ecuadors-45fc8f236240)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 👥 Authors <a name="authors"></a>

🕺🏻**Gabriel Okundaye**

- GitHub: [GitHub Profile](https://github.com/D0nG4667)

- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/dr-gabriel-okundaye).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ⭐️ Show your support <a name="support"></a>

If you like this project kindly show some love, give it a 🌟 **STAR** 🌟. Thank you!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📝 License <a name="license"></a>

This project is [MIT](/LICENSE) licensed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
