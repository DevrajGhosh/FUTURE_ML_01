import pandas as pd


def load_and_clean(filepath):
    df = pd.read_csv(filepath, encoding='latin1')

    # Drop duplicates and rows missing Sales
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Sales'], inplace=True)

    # Parse date
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')

    # Extract time features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day'] = df['Order Date'].dt.day

    return df


def aggregate_monthly(df):
    monthly = (
        df.groupby(['Year', 'Month'])['Sales']
        .sum()
        .reset_index()
    )
    monthly['Date'] = pd.to_datetime(
        monthly[['Year', 'Month']].assign(Day=1)
    )
    monthly.sort_values('Date', inplace=True)
    monthly.reset_index(drop=True, inplace=True)
    return monthly
