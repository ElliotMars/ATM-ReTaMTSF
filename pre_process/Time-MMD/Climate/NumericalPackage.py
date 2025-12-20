import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Climate/Climate.csv')
dropcolumn = ['date', 'MapDate', 'AreaOfInterest', 'ValidStart', 'ValidEnd', 'StatisticFormatID']
df.drop(dropcolumn, axis=1, inplace=True)
cols = df.columns.tolist()
new_order = cols[-2:] + cols[:-2]
df = df[new_order]
df.columns = ['start_date', 'end_date', 'Drought Level', 'Abnormal Dry', 'Moderate Drought', 'Severe Drought',
              'Extreme Drought', 'Exceptional Drought']

# transform Date into datetime
df['start_date'], df['end_date'] = pd.to_datetime(df['start_date']), pd.to_datetime(df['end_date'])

# sort accroding to Date
df = df.sort_values(by='start_date', ascending=True).reset_index(drop=True)
# format transform
df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
print(df)

df.to_parquet('../../../dataset/Time-MMD/numerical/Climate/Climate.parquet')