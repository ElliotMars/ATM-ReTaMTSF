import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Environment/Environment.csv')
dropcolumn = ['date', 'CBSA', 'CBSA Code', 'Defining Site']
df.drop(dropcolumn, axis=1, inplace=True)
cols = df.columns.tolist()
new_order = cols[-2:] + cols[:-2]
df = df[new_order]
df.columns = ['start_date', 'end_date', 'Air Quality Index', 'Category', 'Defining Parameter', 'Number of Sites Reporting']
replace_dict = {
    "Good": 5,
    "Moderate": 4,
    "Unhealthy": 3,
    "Unhealthy for Sensitive Groups": 2,
    "Very Unhealthy": 1,
    "CO": 1,
    "NO2": 2,
    "Ozone": 3,
    "PM10": 4,
    "PM2.5": 5
}
df = df.replace(replace_dict)

# transform Date into datetime
df['start_date'], df['end_date'] = pd.to_datetime(df['start_date']), pd.to_datetime(df['end_date'])

# sort according to Date
df = df.sort_values(by='start_date', ascending=True).reset_index(drop=True)

df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
print(df)

df.to_parquet('../../../dataset/Time-MMD/numerical/Environment/Environment.parquet')