import pandas as pd

def main() -> None:
    df = pd.read_csv('./data/insurance.csv')
    print(df.head())


main()
    
