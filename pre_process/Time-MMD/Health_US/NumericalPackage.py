import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/numerical/Health_US/Health_US.csv')
dropcolumn = ['date', 'REGION TYPE', 'REGION', 'YEAR', 'WEEK', 'YEAR_WEEK']
df.drop(dropcolumn, axis=1, inplace=True)
cols = df.columns.tolist()
df.columns = ['start_date', 'end_date', 'WEIGHTED ILI', 'Influenza Patients Proportion', 'AGE 0-4', 'AGE 25-49',
              'AGE 25-64', 'AGE 5-24', 'AGE 50-64', 'AGE 65', 'ILITOTAL', 'NUM OF PROVIDERS', 'TOTAL PATIENTS']
df = df.replace("X", 0).apply(pd.to_numeric, errors="ignore")

# transform Date into datetime
df['start_date'], df['end_date'] = pd.to_datetime(df['start_date']), pd.to_datetime(df['end_date'])

# sort according to Date
df = df.sort_values(by='start_date', ascending=True).reset_index(drop=True)

df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
print(df)

df.to_parquet('../../../dataset/Time-MMD/numerical/Health_US/Health_US.parquet')