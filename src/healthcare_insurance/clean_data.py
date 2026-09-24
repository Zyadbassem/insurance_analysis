import pandas as pd

'''
This function takes an input: DataFrame and returns a cleaned DataFrame
'''
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Dropping NA
    df = df.dropna()

    
    #Floating any data
    df = df.astype({'bmi': float, 'children': float, 'charges': float})

    # # Renaming the categorical variable
    df['sex'] = df['sex'].replace('male', '1')
    df['sex'] = df['sex'].replace('female', '0')

    df['smoker'] = df['smoker'].replace('yes', '1')
    df['smoker'] = df['smoker'].replace('no', '0')

    df = df.astype({'bmi': float, 'children': float, 'charges': float, 'smoker': int, 'sex': int})

    df = pd.concat([df, pd.get_dummies(df['region'], dtype=int)], axis=1)
    df = df.rename(columns={'sex': 'is_male', 'smoker': 'is_smoker'})

    return df
    
    
