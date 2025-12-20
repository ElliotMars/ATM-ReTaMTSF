import pandas as pd

# dictionary create
physical_quantities = {
    "Air Quality Index": [
        "The Air Quality Index (AQI) is a numerical scale used to communicate how polluted the air currently is or how polluted it is forecast to become. It is calculated based on concentrations of major air pollutants."
    ],
    "Category": [
        "Category refers to the qualitative description of air quality corresponding to an AQI range, such as 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', or 'Hazardous'."
    ],
    "Defining Parameter": [
        "The defining parameter is the specific air pollutant (e.g., PM2.5, PM10, Ozone, CO, SO2, NO2) that determines the AQI value for a given period, based on which pollutant has the highest impact."
    ],
    "Number of Sites Reporting": [
        "The number of sites reporting indicates the total count of monitoring stations that submitted valid air quality data for a given pollutant and time period."
    ]
}


# Pandas DataFrame
df = pd.DataFrame(physical_quantities)
print(df)

# save as Parquet file
df.to_parquet("../../../dataset/Time-MMD/textual/Environment/QueryTextPackage.parquet", engine="pyarrow")