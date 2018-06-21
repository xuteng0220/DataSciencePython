import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)
df.head()

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()

### Question 0 (Example)
def answer_zero():
    return df.iloc[0]

firstCountry = answer_zero() 
type(firstCountry)
firstCountry


### Question 1
import numpy as np

def answer_one():
    maxSG = np.max(df['Gold'])
#     return df.index[df['Gold'] == maxSG]
    return df[df['Gold'] == maxSG].index
a = answer_one()
a
type(a)

### Question 2
def answer_two():
    diffG = np.abs(df['Gold'] - df['Gold.1'])
    maxDiffG = np.max(diffG)
    return df[(df['Gold'] - df['Gold.1']) == maxDiffG].index
answer_two()

### Question 3
def answer_three():
    dfGold = df[(df['Gold'] > 0) & (df['Gold.1'] > 0)]
    relDiffG = np.max((dfGold['Gold'] - dfGold['Gold.1'])/dfGold['Gold.2'])
    return dfGold[(dfGold['Gold'] - dfGold['Gold.1'])/dfGold['Gold.2'] == relDiffG].index
answer_three()

### Question 4
def answer_four():
    df['Points'] = df['Gold.2'] * 3 + df['Silver.2'] * 2 + df['Bronze.2']
    return df['Points']
answer_four()



## Part 2
### Question 5
census_df = pd.read_csv('census.csv')
census_df.head()

# census_df.where(census_df['SUMLEV'] == 40).dropna() # SUMLEV == 40, COUNTY == 0

def answer_five1():
    countyNum = {}
    county_df = census_df[census_df['SUMLEV'] == 50]
    for state in county_df['STNAME'].unique():
        countyNum[state] = np.sum(county_df[county_df['STNAME'] == state].dropna()['COUNTY'])
    countyNum = pd.Series(countyNum)
    return countyNum.where(countyNum == np.max(countyNum)).dropna().index
answer_five1()

def answer_five2():
    countyNum = {}
    county_df = census_df[census_df['SUMLEV'] == 50]
    for group, frame in county_df.groupby('STNAME'):
        countySum = np.sum(frame['COUNTY'])
        countyNum[group] = countySum
    countyNum =pd.Series(countyNum)
    return countyNum.where(countyNum == np.max(countyNum)).dropna().index
answer_five2()


def answer_five3():
    county_df = census_df[census_df['SUMLEV'] == 50]
    countyNum = county_df.set_index('STNAME').groupby(level = 0)['COUNTY'].agg({'COUNTY': np.sum})
    return countyNum[countyNum['COUNTY'] == np.max(countyNum['COUNTY'])].index
answer_five3()


def answer_six():
    popNum = census_df.set_index('STNAME').groupby(level = 0)['CENSUS2010POP'].agg({'CENSUS2010POP': np.sum})
    return popNum.sort_values('CENSUS2010POP', ascending=False)[0:3].index