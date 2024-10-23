import pandas as pd
import os
import requests

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


cleaned_data = clean_data(data_path)
print(cleaned_data[['b_day', 'team', 'height', 'weight', 'country', 'draft_round', 'draft_year', 'salary']].head())