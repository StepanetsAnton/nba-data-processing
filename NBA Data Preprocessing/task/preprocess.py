import pandas as pd
import os
import requests
from datetime import datetime
import re
# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


def clean_data(path):
    df = pd.read_csv(path)

    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y', errors='coerce')

    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y', errors='coerce')


    df['team'] = df['team'].fillna('No Team')

    def clean_weight(weight):
        if isinstance(weight, str):
            weight = weight.strip()

            try:
                if '/' in weight and 'kg' in weight:
                    kg_part = weight.split('/')[1].strip()
                    return float(
                        kg_part.replace('kg.', '').replace('kg', '').strip())
                else:
                    return None
            except ValueError:
                return None
        else:
            return None

    df['weight'] = df['weight'].apply(clean_weight)

    def clean_height(height):
        if isinstance(height, str):
            try:
                if '/' in height:
                    meters_part = height.split('/')[1].strip()
                    return float(meters_part)
                else:
                    return None
            except ValueError:
                return None
        return None

    df['height'] = df['height'].apply(clean_height)

    df['salary'] = df['salary'].replace(r'[\$,]', '', regex=True).astype(float)

    df['height'] = df['height'].astype(float)
    df['weight'] = df['weight'].astype(float)
    df['salary'] = df['salary'].astype(float)

    df['country'] = df['country'].apply(lambda x: 'USA' if x == 'USA' else 'Not-USA')

    df['draft_round'] = df['draft_round'].replace('Undrafted', '0')

    return df


def feature_data(df):
    df['version'] = df['version'].apply(
        lambda x: 2000 + int(re.search(r'\d{2}$', x).group()) if isinstance(x, str) else None)

    df['age'] = df['version'] - df['b_day'].dt.year

    df['draft_year'] = df['draft_year'].dt.year

    df['experience'] = df['version'] - df['draft_year']

    df['bmi'] = df.apply(
        lambda row: row['weight'] / (row['height'] ** 2) if pd.notnull(row['weight']) and pd.notnull(row['height']) and
                                                            row['height'] > 0 else None, axis=1)

    df = df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])

    high_cardinality_columns = [col for col in df.columns if df[col].nunique() >= 50 and col != "bmi" and col != "salary"]
    df = df.drop(columns=high_cardinality_columns)

    return df

def multicol_data(df):
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    corr_matrix = df_numeric.corr()


    to_drop = set()
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and i != 'salary' and j != 'salary' and abs(corr_matrix.loc[i, j]) > 0.5:

                if abs(corr_matrix.loc[i, 'salary']) < abs(corr_matrix.loc[j, 'salary']):
                    to_drop.add(i)
                else:
                    to_drop.add(j)


    df_numeric = df_numeric.drop(columns=to_drop)

    return df_numeric

cleaned_data = clean_data(data_path)
featured_data = feature_data(cleaned_data)
data_multicol = multicol_data(featured_data)
print(data_multicol.head())