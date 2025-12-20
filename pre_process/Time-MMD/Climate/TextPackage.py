import pandas as pd
df = pd.read_csv('../../../dataset/Time-MMD/textual/Climate/Climate_report.csv')
df.drop(df.columns[0], axis=1, inplace=True)
# concatenate Text along rows
#df['Text'] = df.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']} {row['preds']}", axis=1)
df['Text'] = df.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']}", axis=1)
df.drop(['end_date', 'fact', 'preds'], axis=1, inplace=True)
df.columns = ['Date', 'Text']

df_search = pd.read_csv('../../../dataset/Time-MMD/textual/Climate/Climate_search.csv')
df_search.drop(df_search.columns[0], axis=1, inplace=True)
#df_search['Text'] = df_search.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']} {row['preds']}", axis=1)
df_search['Text'] = df_search.apply(lambda row: f"From {row['start_date']} to {row['end_date']}: {row['fact']}", axis=1)
df_search.drop(['end_date', 'fact', 'preds'], axis=1, inplace=True)
df_search.columns = ['Date', 'Text']

df_combined = pd.concat([df, df_search], axis=0, ignore_index=True)
# concatenate the same rows in Date，Text concatenated with blank space
df_combined = df_combined.groupby('Date', as_index=False).agg({'Text': ' '.join})
# sort according to 'Date'
df_combined.sort_values(by='Date', inplace=True)

# if rerank：
df_combined.reset_index(drop=True, inplace=True)

print(df_combined)


df_combined.to_parquet('../../../dataset/Time-MMD/textual/Climate/Climate_combined.parquet')