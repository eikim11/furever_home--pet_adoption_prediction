import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from pylab import savefig


#datetime_type = year, month, dayofweek, or hour
def parse_date_cols(df, col: str, new_col: str, datetime_type: str):
    df[col] = pd.to_datetime(df[col])
    if datetime_type == 'year':
        df[new_col] = df[col].dt.year
    elif datetime_type == 'month':
        df[new_col] = df[col].dt.month
    elif datetime_type == 'dayofweek':
        df[new_col] = df[col].dt.dayofweek
    elif datetime_type == 'hour':
        df[new_col] = df[col].dt.hour
    else:
        print("wrong datetime type")
    return df

#convert col values dtype: int, float, string, etc
def convert_dtype(df, col: str, new_dtype: str):
    df[col] = df[col].astype(new_dtype)
    return df

#consolidate column values
def simplify_col_vals(df, col:str, value:list, replacement:list):
  # spayed --> female, neutered --> male, unknown remains as is
  df[col] = df[col].replace(value, replacement*len(value))

# Simplify 'Sex' column from 5 unique vals into 3 - Male, Female, Unknown
def replace_simplify_group_cols(df):

    # Create new column 'Fixed' that shows True if an animal is neutered or spayed
    df['Fixed'] = df['Sex'].apply(lambda x: True if x in ['Neutered', 'Spayed'] else False)

    # Create new column 'Has_Name' that shows True if an animal has a name
    df['Has_Name'] = df['Name'].notnull()

    # Have 'Sex' column only show male or female
    simplify_col_vals(df, 'Sex', ['Neutered'], ['Male'])
    simplify_col_vals(df, 'Sex', ['Spayed'], ['Female'])

    # Simplify 'Outcome Type' --> "good", "neutral", "bad"
    outcome_adopt = ['ADOPTION']
    outcome_return = ['RETURN TO OWNER', 'RTOS']
    outcome_transfer = ['TRANSFER']
    outcome_bad = ['DIED', 'DISPOSAL', 'EUTHANIZE', 'ESCAPED/STOLEN']

    #drop 93 rows with missing 'Outcome Type'
    df = df[df['Outcome Type'].notna()]
    simplify_col_vals(df, 'Outcome Type', outcome_bad, [0])
    simplify_col_vals(df, 'Outcome Type', outcome_transfer, [1])
    simplify_col_vals(df, 'Outcome Type', outcome_return, [2])
    simplify_col_vals(df, 'Outcome Type', outcome_adopt, [3])
    # dtype of this column is object -- might need to change to int later

    return df

def create_fill_age_col(df, method='median'):
    df['Age'] = df['Intake Date'] - df['Date Of Birth']
    df['Age'] = df['Age'].dt.days.astype(float)

    if method == 'median':
        df['Age'] = df['Age'].fillna(df['Age'].median())

    elif method == 'mean':
        df['Age'] = df['Age'].fillna(df['Age'].mean())

    elif method == '0':
        df['Age'] = df['Age'].fillna(0)

    #convert from age in Days to Years
    df['Age'] = round(df['Age'] / 365, 2)

    #drop DoB col
    df = df.drop(columns='Date Of Birth', inplace=True)

    return df

def create_subdfs(df):

    # Separated by Animal Type
    df_cat = df[df['Type'] == 'CAT']
    df_dog = df[df['Type'] == 'DOG']
    df_other = df[df['Type'] == 'OTHER']

    return df_cat, df_dog, df_other

def separate_breed(df):
    df['Breed'] = df['Breed'].str.replace('/MIX', '')
    new_val = df['Breed'].str.split('/', n=1, expand=True)

    df['Breed_1'] = new_val[0]
    df['Breed_2'] = new_val[1]

    df = df.drop(columns='Breed', inplace=True)

    return df

def separate_color(df):
    new_val = df['Color'].str.split('/', n=1, expand=True)

    df['Color_1'] = new_val[0]
    df['Color_2'] = new_val[1]

    df = df.drop(columns='Color', inplace=True)

    return df

def simplify_intake_condition(df):
    new_val = df['Intake Condition'].str.split('/', n=1, expand=True)
    df['Intake Condition'] = new_val[0]

    return df

def dummify_categ_cols(df):

    need_dummies = ['Type', 'Sex', 'Size', 'Intake Condition',
                'Breed_1', 'Breed_2', 'Color_1', 'Color_2']

    # Take care of NaNs before creating dummies
    df['Size'].fillna('Unknown', inplace=True)

    for col in need_dummies:

        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)

    # cols in need_dummies will be dropped in the drop_cols function

    return df

def drop_cols(df, model: str):
    '''
    Input: dataframe, model type = 'clf' or 'reg'
    'clf' model will drop 'Outcome Type' and leave 'Days in Shelter' to be the dependent variable
    'reg' model will drop 'Days in Shelter' and leave 'Outcome Type' to be the dependent variable

    Output:
    dataframe with dropped columns
    '''

    if model == 'reg':

        # Remove 'Outcome Type', keep 'Days in Shelter'
        drop_cols = ['Name', 'Impound Number', 'Kennel Number', 'Outcome Subtype',
                     'Intake Jurisdiction', 'Outcome Jurisdiction', 'Outcome Jurisdiction',
                     'Location', 'Count', 'Animal ID', 'Outcome_Year',
                     'Intake Type', 'Intake Subtype', 'Outcome Condition',
                     'Outcome Date', 'Outcome Zip Code', 'Outcome_Month', 'Outcome_dayofweek',
                    'Outcome Type', 'Type', 'Sex', 'Size', 'Intake Condition',
                    'Breed_1', 'Breed_2', 'Color_1', 'Color_2']

        df = df.drop(columns=drop_cols)

        return df

    elif model == 'clf':

        # Remove 'Days in Shelter', keep 'Outcome Type'
        drop_cols = ['Name', 'Impound Number', 'Kennel Number', 'Outcome Subtype',
                     'Intake Jurisdiction', 'Outcome Jurisdiction', 'Outcome Jurisdiction',
                     'Location', 'Count', 'Animal ID', 'Outcome_Year',
                     'Intake Type', 'Intake Subtype', 'Outcome Condition',
                     'Outcome Date', 'Outcome Zip Code', 'Outcome_Month', 'Outcome_dayofweek',
                    'Days in Shelter', 'Type', 'Sex', 'Size', 'Intake Condition',
                    'Breed_1', 'Breed_2', 'Color_1', 'Color_2']

        df = df.drop(columns=drop_cols)

        return df

    else:
        print("Enter Model: 'clf' or 'reg'.")



if __name__ == "__main__":

    #load raw data
    df = pd.read_csv('data/Animal_Shelter_Intake_and_Outcome.csv')

    #convert dtype in columns
    parse_date_cols(df, 'Outcome Date', '_', '_')
    parse_date_cols(df, 'Date Of Birth', '_', '_')
    convert_dtype(df, 'Days in Shelter', 'int')

    #Create month, dayofweek, year columns from intake date
    parse_date_cols(df, 'Intake Date', 'Intake_Month', datetime_type='month')
    parse_date_cols(df, 'Intake Date', 'Intake_dayofweek', datetime_type='dayofweek')
    parse_date_cols(df, 'Intake Date', 'Intake_Year', datetime_type='year')

    # Parse 'Outcome Date' for EDA, not for modeling - potential data leakage
    parse_date_cols(df, 'Outcome Date', 'Outcome_Month', datetime_type='month')
    parse_date_cols(df, 'Outcome Date', 'Outcome_dayofweek', datetime_type='dayofweek')
    parse_date_cols(df, 'Outcome Date', 'Outcome_Year', datetime_type='year')

    create_fill_age_col(df)

    df = replace_simplify_group_cols(df)

    separate_breed(df)

    separate_color(df)

    simplify_intake_condition(df)

    df = dummify_categ_cols(df)

    # Might not need these sub-dfs
    df_cat, df_dog, df_other = create_subdfs(df)

    # datasets to be plugged into regression model
    df_reg = drop_cols(df, 'reg')
    df_cat_reg = drop_cols(df_cat, 'reg')
    df_dog_reg = drop_cols(df_dog, 'reg')
    df_other_reg = drop_cols(df_other, 'reg')

    # datasets to be plugged into classification model
    df_clf = drop_cols(df, 'clf')
    df_cat_clf = drop_cols(df_cat, 'clf')
    df_dog_clf = drop_cols(df_cat, 'clf')
    df_other_clf = drop_cols(df_cat, 'clf')

    # Save cleaned dataframes as csv
    df_reg.to_csv('data/reg_data_cleaned.csv')
    df_cat_reg.to_csv('data/reg_cat_data.csv')
    df_dog_reg.to_csv('data/reg_dog_data.csv')
    df_other_reg.to_csv('data/reg_other_data.csv')

    df_clf.to_csv('data/clf_data_cleaned.csv')
    df_cat_clf.to_csv('data/clf_cat_data.csv')
    df_dog_clf.to_csv('data/clf_dog_data.csv')
    df_other_clf.to_csv('data/clf_other_data.csv')