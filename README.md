## Michael Dorado's Capstone (Last Revised 3/11/2024)
=========================

### Project Overview  
- Alternative fueling stations are located throughout the United States, and their availability continues to grow.
The Alternative Fuels Data Center (AFDC) maintains a website where you can find alternative fueling stations near you or on a route, obtain counts of alternative fueling stations by state, view maps, and more.
The dataset used for this notebook was found on data.gov, and published by the National Renewable Energy Laboratory on July 31st, 2021.
Using Time Series Forecasting, I will forecast the growth of Alternative Fuel Stations and compare to electric vehicle (EV) sales in the United States.


## Introduction
- Alternative fueling stations are located throughout the United States, and their availability continues to grow. The Alternative Fuels Data Center (AFDC) maintains a website where you can find alternative fueling stations near you or on a route, obtain counts of alternative fueling stations by state, view maps, and more.
The dataset used for this notebook was published by the National Renewable Energy Laboratory on July 31st, 2021.


## Problem Statement
- Many consumers are now opting for electric vehicles over gas powered vehicles when purchasing a new vehicle. However, The United States' public electric vehicle charging network must meet the demand of EV sales rapid increase. In an October 2022 survey, 46 percent of U.S. consumers mentioned the lack of public charging as one of their leading concerns regarding battery-electric vehicles.


## The Problem area
- My area of interest is AFS infrastructure growth, as well as EV market growth. I want my project to address the feasibility of purchasing an EV based on AFS growth. In other words, comprehend the growth of AFS versus the growth of EV vehicles.


## The Big Idea
- Time Series Forecasting can help measure the growth of alternative fuel stations infrastructure in the United States. Through Time Series Forecasting, consumers can make informed decisions when deciding between an electric or gas-powered vehicle purchase.

### Walkthrough Demo
(TBD)
...
...
...

### Project Flowchart
(TBD)
...
...
...

### Project Organization
(TBD)
...
...
...

* `data` 
    - https://drive.google.com/drive/folders/1dePFXd1pqRGa6UPXgZ2p4HVrS8XknXEP?usp=sharing
    - https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2

* `model`
    - #!/usr/bin/env python
# coding: utf-8

# # Michael Dorado
# ## Sprint 2 - Alternative Fuel Stations Exploratory Data Analysis
# ### March 12th, 2024

# ## Table of Contents
# 
# 0. [Introduction](#section0) <br>
# 0.1 [Problem Statement](#section0.1) <br>
# 0.2 [The Problem Area](#section0.2) <br>
# 0.3 [The Big Idea](#section0.3) <br>
# 0.4 [Libraries and Packages](#section0.4) <br>
# 1. [Exploratory Data Analysis](#section1) <br>
# 1.1 [Sprint 1 EDA](#section1.1) <br>
# 1.1.1 [Data Dictionary](#section1.1.1) <br>
# 1.2 [Sprint 2 EDA](#section1.2) <br>
# 1.2.1 [Exploring time-of-year](#section1.2.1) <br>
# 2. [Preprocessing](#section2) <br>
# 2.1 [Trend-Seasonal Decomposition ](#section2.1) <br>
# 2.2 [Forecasting](#section2.2) <br>
# 3. [Baseline Forecasts and Evaluation](#section3) <br>

# <a id='section0'></a>
# ### <center> 0. Introduction
# Alternative fueling stations are located throughout the United States, and their availability continues to grow. The Alternative Fuels Data Center (AFDC) maintains a website where you can find alternative fueling stations near you or on a route, obtain counts of alternative fueling stations by state, view maps, and more.
# The dataset used for this notebook was published by the National Renewable Energy Laboratory on July 31st, 2021.
# 
# <a id='section0.1'></a>
# ### <center> 0.1 Problem Statement
# Many consumers are now opting for electric vehicles over gas powered vehicles when purchasing a new vehicle. However, The United States' public electric vehicle charging network must meet the demand of EV sales rapid increase. In an October 2022 survey, 46 percent of U.S. consumers mentioned the lack of public charging as one of their leading concerns regarding battery-electric vehicles.
# 
# <a id='section0.2'></a>
# ### <center> 0.2 The Problem area
# My area of interest is AFS infrastructure growth, as well as EV market growth. I want my project to address the feasibility of purchasing an EV based on AFS growth. In other words, comprehend the growth of AFS versus the growth of EV vehicles.
# 
# 
# <a id='section0.3'></a>
# ### <center> 0.3 The Big Idea
# Time Series Forecasting can help measure the growth of alternative fuel stations infrastructure in the United States. Through Time Series Forecasting, consumers can make informed decisions when deciding between an electric or gas-powered vehicle purchase.

# In[1]:


# Formatting the Notebook width
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# <a id='section0.4'></a>
# ### <center> 0.4 Libraries and Packages

# In[2]:


# Importing Libraries
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# Stats
from statsmodels.api import tsa # Time Series Analysis
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import month_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# Miscellaneous
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Importing Dataset as 'df'
df_original = pd.read_csv('Alternative Fuel Stations.csv')
df = df_original.copy()


# <a id='section1'></a>
# ### <center> 1. Exploratory Data Analysis

# <a id='section1.1'></a>
# ### <center> 1.1 Sprint 1 Exploratory Data Analysis

# In[4]:


# df info overview
df.info()


# The Dataset contains columns for alternative fuel stations that do not pertain to electric vehicles.
# These do not support my analysis, therefore, I am going to get rid of any columns that are irrelevant to my study, and columns with a low non-null count.

# In[5]:


# Removing columns
non_EV_Data = ['Intersection Directions','Station Phone', 'Plus4','Expected Date','BD Blends','EV Other Info', 'NG Fill Type Code', 'NG PSI','Owner Type Code', 'Federal Agency ID',
               'Federal Agency Name','Hydrogen Status Link','NG Vehicle Class','LPG Primary','E85 Blender Pump','Intersection Directions (French)','Access Days Time (French)',
              'BD Blends (French)','Groups With Access Code (French)','Hydrogen Is Retail','Access Detail Code','Federal Agency Code','Federal Agency Code','CNG Dispenser Num','CNG On-Site Renewable Source',
               'CNG Total Compression Capacity','CNG Storage Capacity','LNG On-Site Renewable Source','E85 Other Ethanol Blends','EV Pricing (French)','LPG Nozzle Types','Hydrogen Pressures',
              'Hydrogen Standards','CNG Fill Type Code','CNG PSI','CNG Vehicle Class','LNG Vehicle Class','Restricted Access','EV On-Site Renewable Source','Status Code','Country','ID'] 

df = df.drop(columns=non_EV_Data)


# In[6]:


# Sanity Check
df.info()


# In[7]:


# Checking for duplicate columns 
df.T.duplicated().any()


# In[8]:


# I removed the columns that did not pertain to EVs, however, there are still records for non-EV on the 'Fuel Type Code' column
df['Fuel Type Code'].value_counts()


# In[9]:


# Cleaning the dataframe further to only see "ELEC" records
EV_Only = 'ELEC'  # Replace this with the value you want to filter by

df = df[df['Fuel Type Code'] == EV_Only]


# In[10]:


# Sanity Check
df.info()


# In[11]:


# Changing the 'Open Date','Updated At', 'Date Last Confirmed' column values to datetime dtype, using a For Loop
to_datetime = ['Open Date', 'Updated At','Date Last Confirmed']  # Replace these with the columns you want to convert

for column in to_datetime:
    df[column] = pd.to_datetime(df[column])


# In[12]:


# Reposition column 'Access Code' to where it makes more sense
move_AccessCode = 'Access Code'
new_position = 6
column = df.pop(move_AccessCode)
df.insert(new_position, move_AccessCode, column)


# In[13]:


# Reposition column 'Facility Type' to where it makes more sense
move_FacilityType = 'Facility Type'
new_position = 6
column = df.pop(move_FacilityType)
df.insert(new_position, move_FacilityType, column)


# In[14]:


# Snapshot of the cleaned dataframe
df.head(5)


# In[15]:


# Checking Value counts
for column in df.columns:
    print(f"Column: {column}")
    print(df[column].value_counts())
    print("\n")


# <a id='section1.1.1'></a>
# ### <center> 1.1.1 Data Dictionary
# `Fuel Type Code:` Dataset Fuel Code, filtered to 'ELECTRIC' only <br>
# `Station Name:` Name of Alternative Fuel Station <br>
# `Street Address:` Address of Alternative Fuel Station <br>
# `City:` City where the Alternative Fuel Station is located <br>
# `State:` State where the Alternative Fuel Station is located <br>
# `Zip:` Address of Alternative Fuel Station <br>
# `Facility Type:` Type of facility for the Alternative Fuel Station location <br>
# `Access Code:` Type of access where the Alternative Fuel Station is located <br>
# `Group With Access Code:` Group category for the type of access for the Alternative Fuel Station location <br>
# `Access Days Time:` Accessible days and time for the locations <br>
# `Cards Accepted:` Types of cards accepted A = Amex, D = Discovery, M = MasterCard, V = Visa <br>
# `EV Level1 EVSE Num:` Qty of Standard 120V Charging Station <br>
# `EV Level2 EVSE Num:` Qty of More Powerful 208/240V Charging Station <br>
# `EV DC Fast Count:` Qty of Fastest Charging Station Type <br>
# `EV Network:` Charging station infrastructure system <br>
# `EV Network Web:` Link for charging station infrastructure system <br>
# `Geocode Status:` Address Geocode <br>
# `Latitude:` Facility's location latitude <br>
# `Longitude:` Facility's location longitude <br>
# `Date Last Confirmed:` Last revision date for the dataframe record <br>
# `Updated At:` Last revision date and time for the dataframe record <br>
# `Open Date:` Date when the facility was inaugurated <br>
# `EV Connector Types:` Type of charging station connector <br>
# `EV Pricing:` Cost to charge vehicles at the facility <br>

# I am now going to visualize some of the records in my dataframe, to take a different angle on exploring the data.

# In[16]:


# Let's look at the deployment of Electric Charging Stations by State
plt.figure(figsize=(10, 10))
df["State"].value_counts().sort_values(ascending=True).plot(kind="barh")
plt.xlabel("Number of EV Charging Station Facilities")
plt.ylabel("State")
plt.title("Number of EV Charging Station Facilities by State")
plt.tight_layout()
plt.show()


# From the chart above it can be visualized the states in the US with the most EV charging station facilities.
# California is the state with the most EV charging stations facilities.

# In[52]:


# Formatting the 'Open Year' values from float to int
df['Open Year'] = df['Open Year'].fillna(0).astype(int)


# In[66]:


# Converting Open Year back to DateTime
df['Open Year'] = pd.to_datetime(df['Open Year'], errors='coerce')


# In[71]:


# Converting Open Date back to Year
df['Open Year'] = df['Open Date'].dt.year


# In[78]:


plt.figure(figsize=(10, 10))
plt.bar(df['Open Year'].value_counts().index,
        df['Open Year'].value_counts().values)
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('# of Facilities Inaugurated')
plt.title('EV Charging Facilities Inaugurated by Year')
plt.show()


# From the chart below we see the number of EV Charging station facilities inaugurated over the years.
# The numbers are growing significantly over since 2017.

# In[17]:


# Assuming 'Open Date' is the name of your column
df['Open Date'] = pd.to_datetime(df['Open Date'])

# Extracting the week of the year and creating a new column
df['Week of Year'] = df['Open Date'].dt.isocalendar().week

# Displaying the updated DataFrame
df.head()


# <a id='section1.2'></a>
# ### <center> 1.2 Sprint 2 Exploratory Data Analysis

# In Section 1.2, EDA continues with more advanced techniques, and a focus towards the Time Series Forecasting modelling.

# In[16]:


# Create a copy of original df as df_TM for all time series related EDA
df_TM = df.copy()


# In[18]:


# Narrow down my dataframe columns for Time Series Forecasting purposes
selected_columns = ['Open Date','State']
df_TM = df_TM[selected_columns].copy()


# In[20]:


# Sanity Check
df_TM.info()


# In[21]:


# Sanity Check
df_TM.head()


# In[22]:


# Checking for Null values
df_TM.isna().sum()


#  `Open Date` column has 1,295 null values, which is less than 3% of the overall column.
#  Because these are factual/set dates for when AFS were first opened, I choose to not fill-in the blanks using a method such as interpolation, to maintain the integrity of my data.

# In[24]:


# Drop Nulls
df_TM = df_TM.dropna()


# In[25]:


# Sanity Check
df_TM.isna().sum()


# In[26]:


# Sanity Check
df_TM.info()


# In[27]:


# Re-setting `Open Date` as DateTime (once again), and assigning as Index Column, per Time Series requirements
df_TM['Open Date'] = pd.to_datetime(df_TM['Open Date'])
df_TM = df_TM.set_index('Open Date')

df_TM.info()


# In[29]:


# Checking Index Range using pandas 'Time Stamp' objects
first_day = df_TM.index.min()
last_day = df_TM.index.max()


# <a id='section1.2.1'></a>
# ### <center> 1.2.1 Exploring time-of-year

# In[39]:


# Create Dummies Columns for 'State'
df_TM_dummies = pd.get_dummies(df_TM, columns=['State'], prefix='State')


# In[40]:


# Sum the Dummies columns along the columns axis (axis=1)
df_TM_dummies['Sum States by Date'] = df_TM_dummies.iloc[:, 1:].sum(axis=1)


# For `Time Series Forecasting` the dataset must have a timestamp/index column and a column to forecast.
# Because my original `State` column contained Objects, I had to create dummies for each of the State values accross the dataset.
# Then, I created a new column `Sum States by Date` which represents the sum of AFS opened accross all states on a given date.
# `Sum States by Date` will be the variable used for forecasting.

# In[41]:


#Sanity Check
df_TM_dummies.info()


# In[42]:


# Resampling the dataframe using the "MS" option to set Monthly frequency by Start day
df_TM_dummies_monthly = df_TM_dummies.resample("MS").sum()

# Sanity check
df_TM_dummies_monthly.head()


# Because my original dataset has missing irregular day intervals, in other words, there are day dates missing, I re-sampled my data to a Monthly frequency, to reduce the amount of missing day-dates throughout the years.

# In[48]:


# Advanced Plotting using Plotly

# Group by the index and sum the dummy columns
df_sum = df_TM_dummies_monthly.groupby(df_TM_dummies_monthly.index).sum()

# Create a line plot with the index on the x-axis and the sum of dummy columns on the y-axis
fig = px.line(df_sum, x=df_sum.index, y=df_sum.columns, title='Monthly Opened AFS Over the Years by States and Sum of States')

# Display the plot
fig.show()


# From the chart above we can see the variable monthly trend of all US States opening AFS over the years.
# The tall green line represents the sum of all US States.

# In[44]:


# Resampling the dataframe using the "YS" option to set Yearly frequency by Start day
df_TM_dummies_yearly = df_TM_dummies.resample("YS").sum()

# Sanity Check
df_TM_dummies_yearly.head()


# In[47]:


# Advanced Plotting using Plotly
# Group by the index and sum the dummy columns
df_sum = df_TM_dummies_yearly.groupby(df_TM_dummies_yearly.index).sum()


# Create a line plot with the index on the x-axis and the sum of dummy columns on the y-axis
fig = px.line(df_sum, x=df_sum.index, y=df_sum.columns, title='Yearly Opened AFS Over the Years by States and Sum of States')

# Display the plot
fig.show()


# From the chart above we can see the variable yearly trend of all US States opening AFS over the years.
# The tall green line represents the sum of all US States.

# In[77]:


plt.figure(figsize=(10, 10))

# Seasonal Plot using matplotlib and seaborn
month_plot(df_TM_dummies_monthly["Sum States by Date"], ax=plt.gca())

plt.title("AFS Monthly Growth Over the Years")
sns.despine()
plt.show()


# The chart above shows the individual monthly trend of AFS monthly openings, cumulative over the years.
# The red lines mark the mean for each month, cumulative over the years.
# We see that January and June have been the months with the highest amount of AFS openings in the United States.

# <a id='section2'></a>
# ### <center> 2. Preprocessing

# <a id='section2.1'></a>
# ### <center> 2.1 Trend-Seasonal Decomposition 

# A fundamental step in time series EDA is the trend-seasonal decomposition. Here, we extract three series from our original observation: 
# - a trend component $T_t$ calculated using a moving average,
# - a seasonal component $S_t$ which is the monthly/daily average of the de-trended series, and
# - the residual $R_t$ that remains after subtracting the trend and seasonal component from the original series.
# 
# Adding up these three components will give back the original series:
# 
# $$y_t = T_t + S_t + R_t$$
# 
# There are different approaches for computing the trend and seasonal components, but the most standard method is implemented by the `seasonal_decompose` function from the time series analysis module of `statsmodels`. 
# 
# **In this section I am exploring this technique using the notes from the Time Series Notebook we learned during the 12/16/2023 class.**
# 

# In[51]:


# Creating new dataframe 'df_TM_monthly_decomp' for decomposition technique, with just index column and the 'Sum States by Date' variable.
columns_to_keep = ['Sum States by Date']
df_TM_monthly_decomp = df_TM_dummies_monthly[columns_to_keep].copy()


# In[52]:


# Sanity Check
df_TM_monthly_decomp.tail()


# In[54]:


# Decompose the time series
decomposition = tsa.seasonal_decompose(df_TM_monthly_decomp, model='additive')


# In[55]:


type(decomposition)


# In[56]:


# Add the decomposition data

df_TM_monthly_decomp["Trend"] = decomposition.trend
df_TM_monthly_decomp["Seasonal"] = decomposition.seasonal
df_TM_monthly_decomp["Residual"] = decomposition.resid

df_TM_monthly_decomp.head(10)


# The NaN values are due the fact that the trend component is calculated with a rolling average that is not defined at the beginning and end of the series.

# In[57]:


cols = ["Trend", "Seasonal", "Residual"]

fig = make_subplots(rows=3, cols=1, subplot_titles=cols)

for i, col in enumerate(cols):
    fig.add_trace(
        go.Scatter(x=df_TM_monthly_decomp.index, y=df_TM_monthly_decomp[col]),
        row=i+1,
        col=1
    )

fig.update_layout(height=800, width=1200, showlegend=False)
fig.show()


# On a macro level, the trend is upward. It is observable that starting in 2010, the significant upward trend started, reaching its peak on November 2020.
# 
# The seasonal plot shows a similar behavior to what was seen on the 'AFS Monthly Growth Over the Years' chart, having the highest amounts of AFS openings on January and June.
# 
# The residual still shows some seasonality. Significant changes in the variance can also be seen towards the end of the residual timeframe.

# <a id='section2.2'></a>
# ### <center> 2.2 Forecasting

# Two different approaches could be taken:
# single-step (short-term) forecasts, or
# multi-step (long-term) forecasts.
# 
# However, we are interested in at least 11 years in the future due to Europe's laws banning all sales of EV vehicles by 2035. Though, similar laws have not been considered in the United States, it can give us a hypothetical scenario.
# Thus, multi-step forecast will be explored.

# In[58]:


# Add seasonal_difference' column
df_TM_monthly_decomp["seasonal_difference"] = df_TM_monthly_decomp["Sum States by Date"].diff(12)


# In[60]:


# Sanity Check
df_TM_monthly_decomp[['seasonal_difference','Sum States by Date']].tail(8)


# In[62]:


# Plot Seasonal Difference using Plotly
fig = px.line(df_TM_monthly_decomp, x=df_TM_monthly_decomp.index, y="seasonal_difference")

fig.update_layout(
    yaxis_title="Difference (AFS Openings)", 
    xaxis_title="Date",
    title="Change in AFS Openings throughout thee United States over the Years"
)

fig.show()


# Based on this differenced series, we can observe that our data is non-stationary. Which means that the data's statistical properties, such as mean and variance, change over time. As we have seen in previous charts, the growth of AFS has grown significantly in the more recent years.

# <a id='section3'></a>
# ### <center> 3. Baseline Forecasts and Evaluation

# For non-stationary data, we will model the trend using Linear Regression, to forecast future trend and add the seasonality to obtain a forecast for the whole series.

# In[63]:


# Creating a new dataframe 'df_TM_monthly' which includes the Monthly Index column, and the 'Sum States by Date' variable to forecast
columns_to_keep = ['Sum States by Date']
df_TM_monthly = df_TM_dummies_monthly[columns_to_keep].copy()


# In[65]:


# Sanity Check
df_TM_monthly.info()


# In[66]:


# Model the trend (non-stationary) using linear regression
X = np.arange(len(df_TM_monthly)).reshape(-1, 1)
y = df_TM_monthly['Sum States by Date'].values.reshape(-1, 1)


# In[67]:


linear_reg = LinearRegression()
linear_reg.fit(X, y)


# In[68]:


# Predict future trend
future_time = pd.date_range(start=df_TM_monthly.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
future_trend = linear_reg.predict(np.arange(len(df_TM_monthly), len(df_TM_monthly) + 12).reshape(-1, 1))


# In[69]:


result = seasonal_decompose(df_TM_monthly['Sum States by Date'], model='additive', period=12)


# In[71]:


# Plot the decomposed components once again during our baseline forecasting model
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(result.trend, label='Trend')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(result.seasonal, label='Seasonality')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(result.resid, label='Residuals')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(df_TM_monthly['Sum States by Date'], label='Original')
plt.legend()

plt.suptitle('Time Series Decomposition')
plt.show()


# This decomposition approach in pandas looks just like the one we did earlier on Section 2.1.

# In[60]:


# Convert the predicted trend index to datetime
future_time = pd.to_datetime(future_time)


# In[72]:


# Forecast future trend and add seasonality for the next year
future_trend = linear_reg.predict(pd.to_numeric((future_time - df_TM_monthly.index[0]).days).values.reshape(-1, 1))

# Including the prevoiously observed seasonality
future_seasonality = result.seasonal.values[-12:]


# In[73]:


# Forecast future AFS by Open Dates by combining trend and seasonality
forecasted_afs = future_trend.flatten() + future_seasonality


# In[74]:


# Plot the Forecast
plt.figure(figsize=(10, 6))
plt.plot(df_TM_monthly.index, df_TM_monthly['Sum States by Date'], label='Observed AFS')
plt.plot(future_time, forecasted_afs, label='Forecasted AFS', linestyle='dashed', marker='o')
plt.title('AFS Forecast with Trend and Seasonality')
plt.xlabel('Time')
plt.ylabel('AFS')
plt.legend()
plt.show()


# In this baseline forecast chart, we can observe the model predicts AFS openings to almost duplicate by 2023.
# This is only the baseline, using entry level knowledge on forecasting.
# 
# More advance knowledge on Time Series Forecasting will be applied for Sprint 3 using PyTorch packages.


* `notebooks`
    - https://drive.google.com/file/d/1f1kQMhVX5aY9SMnOn1lKSOM9OHzL8-JC/view?usp=sharing

* `reports`
    - https://drive.google.com/file/d/1gfsh7OHqkhOUeJ5oI0Ta8SDFqJt84Tfe/view?usp=sharing

* `references`
    - https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2
    - https://drive.google.com/file/d/1uZ5s6nGSdIEIW1o1OMtTaLCXAZ_Tq1hY/view?usp=sharing

* `src`
    - Contains the project source code (refactored from the notebooks) [(TBD)]

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control (TBD)

* `capstine_env.yml`
    - Conda environment specification (TBD)

* `Makefile`
    - Automation script for the project (TBD)

* `README.md`
    - https://github.com/michaeldorado/capstone/edit/main/README.md

* `LICENSE`
    - Project license (TBD)

### Dataset

- https://drive.google.com/file/d/1rRBdXpF4gHcNwsWSMjXkNnBixUPrWxO_/view?usp=sharing

### Credits & References
https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2
https://drive.google.com/file/d/1uZ5s6nGSdIEIW1o1OMtTaLCXAZ_Tq1hY/view?usp=sharing
