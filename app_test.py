import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation

import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
import plotly.graph_objs
import plotly
from scipy.interpolate import griddata

st.set_page_config(
    page_title="Option Dashboard",
    page_icon="✅",
    layout="wide",
)

# read csv from a github repo
dataset_url = "https://raw.githubusercontent.com/bm1125/cboe/refs/heads/master/output/MSFT.csv"

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

df['Strike'] = df['Strike'].str.replace("MSFT","").astype('float')


st.title("Option Visu Dashboard")

# df
st.subheader("Option DataFrame")
st.write(f"Data Source: https://raw.githubusercontent.com/bm1125/cboe/refs/heads/master/output/MSFT.csv")
st.dataframe(df)


# Pivot IV
st.subheader("IV DataFrame")

df['Date']= pd.to_datetime(df['Date'])
df['timestamp'] = df['Date'].values.astype(np.int64) // 10 ** 9

iv_df = pd.pivot_table(df, values='IV', index=['timestamp'],columns=['Strike'])

st.dataframe(iv_df)

# IV Surface
st.subheader("IV Surface")

option = st.selectbox("Display Call/Put",("Call", "Put"))

df['Date']= pd.to_datetime(df['Date'])
df['timestamp'] = df['Date'].values.astype(np.int64) // 10 ** 9
df['timestamp'] = df['timestamp'].astype('float')
df = df[df['Type']==option[0]]
df = pd.pivot_table(df, values='IV', index=['timestamp'],columns=['Strike'])

X, Y = df.columns.values.astype(float), df.index.values.astype(float)
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=df.values, colorscale="Blues")])

st.plotly_chart(fig,height=1000,width=1000)





