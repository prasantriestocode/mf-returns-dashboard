import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(
    page_title="Mutual Fund Returns Dashboard",
    layout="wide"
)

st.title("Mutual Fund Returns Dashboard")

# -------------------------------------------------
# Load data (cached)
# -------------------------------------------------
@st.cache_data
def load_nav_data():
    df = pd.read_parquet("nav_monthly.parquet")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

nav_df = load_nav_data()

# -------------------------------------------------
# Backend engine
# -------------------------------------------------
def prepare_nav_data(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['SchemeCode', 'Date'])


def filter_schemes_by_history(df, min_years):
    history = (
        df.groupby('SchemeCode')['Date']
        .agg(['min', 'max'])
        .reset_index()
    )
    history['years'] = (history['max'] - history['min']).dt.days / 365.25
    eligible = history.loc[history['years'] >= min_years, 'SchemeCode']
    return df[df['SchemeCode'].isin(eligible)]


def calculate_rolling_returns(df, window_years):
    months = window_years * 12

    df = df.copy()
    df = df.sort_values(['SchemeCode', 'Date'])
    df = df.set_index('Date')

    rr = (
        df.groupby('SchemeCode')['NAV']
        .apply(lambda x: (x / x.shift(months)) ** (1 / window_years) - 1)
        .dropna()
        .reset_index()
    )

    # 🔒 FORCE correct column names (Cloud-safe)
    rr.columns = ['SchemeCode', 'Date', f'{window_years}Y']

    return rr

def calculate_calendar_year_returns(df):
    df = df.copy()
    df['Year'] = df['Date'].dt.year

    year_end = (
        df.sort_values('Date')
        .groupby(['SchemeCode', 'Year'])
        .tail(1)
    )

    year_end['Return'] = (
        year_end.groupby('SchemeCode')['NAV'].pct_change()
    )

    return year_end.dropna(subset=['Return'])[['SchemeCode', 'Year', 'Return']]


def build_return_table(df, return_type, window_years=None, year_range=None):
    df = prepare_nav_data(df)

    if return_type == "rolling":
        df = filter_schemes_by_history(df, window_years)
        rr = calculate_rolling_returns(df, window_years)

        table = rr.pivot(
            index='SchemeCode',
            columns='Date',
            values=f'{window_years}Y'
        )

    elif return_type == "calendar":
        cr = calculate_calendar_year_returns(df)

        if year_range:
            start, end = year_range
            cr = cr[(cr['Year'] >= start) & (cr['Year'] <= end)]

        table = cr.pivot(
            index='SchemeCode',
            columns='Year',
            values='Return'
        )

    else:
        raise ValueError("Invalid return type")

    return table.sort_index()

# -------------------------------------------------
# Colour coding
# -------------------------------------------------
def colour_returns(val):
    if pd.isna(val):
        return ""
    if val > 0.12:
        return "background-color: #d4f8d4"
    elif val >= 0.06:
        return "background-color: #fff3cd"
    else:
        return "background-color: #f8d7da"

# -------------------------------------------------
# UI controls
# -------------------------------------------------
st.subheader("Return Configuration")

return_type_label = st.radio(
    "Select return type",
    ["Rolling Returns", "Calendar Year Returns"],
    horizontal=True
)

if return_type_label == "Rolling Returns":
    window_years = st.selectbox(
        "Rolling return window (years)",
        [1, 3, 5, 7, 10, 15],
        index=2
    )

if return_type_label == "Calendar Year Returns":
    min_year = int(nav_df['Date'].dt.year.min())
    max_year = int(nav_df['Date'].dt.year.max())
    year_range = st.slider(
        "Select calendar year range",
        min_year,
        max_year,
        (2015, max_year)
    )

# -------------------------------------------------
# Bucket filter
# -------------------------------------------------
st.subheader("Scheme Filters")

selected_buckets = st.multiselect(
    "Select buckets",
    sorted(nav_df['Bucket'].dropna().unique()),
    default=sorted(nav_df['Bucket'].dropna().unique())
)

filtered_nav = nav_df[nav_df['Bucket'].isin(selected_buckets)]

# -------------------------------------------------
# Build return table
# -------------------------------------------------
if return_type_label == "Rolling Returns":
    return_table = build_return_table(
        filtered_nav,
        return_type="rolling",
        window_years=window_years
    )
else:
    return_table = build_return_table(
        filtered_nav,
        return_type="calendar",
        year_range=year_range
    )

# -------------------------------------------------
# Merge metadata
# -------------------------------------------------
meta = filtered_nav[['SchemeCode', 'SchemeName', 'Bucket']].drop_duplicates()

final_df = meta.merge(
    return_table,
    left_on='SchemeCode',
    right_index=True,
    how='inner'
)

meta_cols = ['SchemeCode', 'SchemeName', 'Bucket']
return_cols = [c for c in final_df.columns if c not in meta_cols]

# -------------------------------------------------
# Display
# -------------------------------------------------
st.markdown(
    "**Colour guide:** 🟢 Above 12% | 🟡 6–12% | 🔴 Below 6%  \n"
    "_Returns are annualised. Only schemes with complete history are shown._"
)

styled_df = (
    final_df
    .style
    .applymap(colour_returns, subset=return_cols)
    .format({c: "{:.2%}" for c in return_cols})
)

st.dataframe(
    styled_df,
    use_container_width=True,
    height=650
)