import pandas as pd
import numpy as np

# pd.read_excel?



## beta
allPositions = pd.read_excel('allPositions.xlsx', 'allPositions', skiprows = 1) # 导入全头寸
allPositions.head()

# allPositions.columns

allPositionsAbbr = allPositions[['证券编码',  '证券名称', '数量', 'RM市值', 'Beta', '部门','产品型1', '资产类型', '行业分类一级']] # 筛选出需要的列

stocks = allPositionsAbbr[allPositionsAbbr['产品型1'] == '股票'] # 筛选出股票
stocksA = stocks.dropna() 


# 计算每一个行业的加权beta
stockBeta = stocksA.copy()
stockBeta['capBeta'] = stockBeta['Beta'] * stockBeta['RM市值']
companyBeta = stockBeta.set_index('行业分类一级').groupby(level = 0)['RM市值', 'capBeta'].agg({'RM市值': np.sum, 'capBeta': np.sum})


industryBeta = companyBeta.copy()
industryBeta['industryBeta'] = industryBeta['capBeta'] / industryBeta['RM市值']



## 成份股
stockA = allPositions[['证券编码',  '证券名称', '数量', 'RM市值', '产品型1', '行业分类一级']] # 筛选出需要的列
stockA = stockA[stockA['产品型1'] == '股票'].dropna()


## 久期评级
bond = allPositions[['证券编码',  '证券名称', '数量', 'RM市值', 'DV01', '久期', '资产类型']] # 筛选出需要的列

bond = bond[bond['资产类型'] == '债券'].dropna()



# 输出到excel文件：全头寸
writer = pd.ExcelWriter('allPositions.xlsx')
allPositions.to_excel(writer, sheet_name='allPositions', index=False, header=True)
## 输出到sheet'Beta'
industryBeta.to_excel(writer, sheet_name = 'Beta', index=True, header=True)
# 输出到sheet'成份股'
stockA.to_excel(writer, sheet_name = 'comStocks', index=False, header=True)
# 输出到sheet'久期'
bond.to_excel(writer, sheet_name = 'Duration', index=False, header=True)
# 输出到sheet'评级'
grade.to_excel(writer, sheet_name = '评级', index=False, header=True)