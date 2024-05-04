## Michael Dorado's Capstone (Last Revised 5/04/2024)
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

## Data Cleaning Summary
- Original dataframe shape (56800, 65)
- Modeling dataframe shape (316, 1)

## EDA Summary
- Trend is positive.
- Seasonality is highest during January and June.
- Residual shows slight seasonality, notable variance at the end.
- Data is non-stationary, its statistical properties, such as mean and variance, change over time.

## Baseline Model Summary
- Rolling Moving Average with a 2-month window was used for the Baseline.
- Mean % Error = -132.92%
- Underestimates the original data
- 2-month window size provides finer granularity and sensitivity variations in the data.

## Linear Regression Findings
- 𝑅^2 = 7.8% 
- Correlation between Sum States by Date Variability and Time Index is low.
- From the plotted chart we can visualize the future trend, 2023 through 2035, is a positive trend, with a (m = 25).

## Pytorch Time Series Forecasting Findings
- The Mean Squared Error (MSE) on the testing data was approximately The Mean Squared Error (MSE) on the testing data was approximately 213125136.
- From the plotted chart we can visualize the trend of the number of predicted AFS over the years 2023 through 2035. It is a positive with a (m = 0.1).

## Considerations
- For each EV charged, there are 6 gas powered vehicles fueled
EV market in the US expected to grow 1,500% by the year 2035
Although the current ratio of EV per EV Station is better than the gas powered vehicle infrastructure (even with the turnaround factored in), to maintain a healthy ratio, the growth of EV stations would have to grow by at least 130% to keep a 300 EV per 1 EV station ratio.

## Findings and Conclusion
- Charging stations infrastructure must have a yearly rate of change of ~6.5% to keep a healthy vehicles to charging stations infrastructure ratio.
- Linear regression slope = 25
- Pytorch Time Series Forecast slope = 0.1
- EV Consumers would face charging stations infrastructure challenges.

* `data` 
    - https://drive.google.com/drive/folders/1dePFXd1pqRGa6UPXgZ2p4HVrS8XknXEP?usp=sharing
    - https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2

* `model`
  - In this baseline forecast chart, we can observe the model predicts AFS openings to almost duplicate by 2023.
    This is only the baseline, using entry level knowledge on forecasting.
    More advance knowledge on Time Series Forecasting will be applied for Sprint 3 using PyTorch packages.
    https://drive.google.com/file/d/1f1kQMhVX5aY9SMnOn1lKSOM9OHzL8-JC/view?usp=sharing


* `notebooks`
    - https://drive.google.com/file/d/1f1kQMhVX5aY9SMnOn1lKSOM9OHzL8-JC/view?usp=sharing

* `reports`
    - https://drive.google.com/file/d/1gfsh7OHqkhOUeJ5oI0Ta8SDFqJt84Tfe/view?usp=sharing

* `references`
    - https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2
    - https://drive.google.com/file/d/1uZ5s6nGSdIEIW1o1OMtTaLCXAZ_Tq1hY/view?usp=sharing


* `README.md`
    - https://github.com/michaeldorado/capstone/edit/main/README.md

- https://drive.google.com/file/d/1rRBdXpF4gHcNwsWSMjXkNnBixUPrWxO_/view?usp=sharing

### Credits & References
https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2
https://drive.google.com/file/d/1uZ5s6nGSdIEIW1o1OMtTaLCXAZ_Tq1hY/view?usp=sharing
