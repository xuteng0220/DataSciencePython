import pandas as pd

pd.read_excel?

allPositions = pd.read_excel('allPositions.xlxs', 'allPositions', skiprows = 1)
allPositions.head()

allPositions.columns

allPositionsAbbr = allPositions[['证券编码',  '证券名称', '数量', 'RM市值', 'Beta', '部门','产品型1', '资产类型', '行业分类一级']]

stocks = allPositionsAbbr[allPositionsAbbr['产品型1'] == '股票']
stocksA = stocks.dropna()



