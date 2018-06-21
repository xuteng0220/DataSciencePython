from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  # display multiple statements


# Introduction to Data Science in Python

## The Python Programming language: Functions


def addNumber(x, y):
    return x + y

addNumber(1, 2)



def addNumbers(x, y, z = None): # default parameter
    if (z == None):
        return x + y
    else:
        return x + y + z

addNumbers(1, 2)
addNumbers(1, 2, 3)



def addNumbers(x, y, z = None, flag = False): # flag value, when it is true, invoke the function
    if (flag):
        print('flag is True')
    if z == None:
        return x + y
    else:
        return x + y + z
addNumbers(1, 2)
addNumbers(x = 1, y = 2) # when using functions, we can specify the parameter names, but cannot misuse them
addNumbers(1, 2, True) # True is treated as 1
addNumbers(1, 2, 3, True)

a = addNumbers # pass the function to a variable and use the variable to invoke the function
a(1, 2, 3, flag = True)

## The Python programming language: Types and Sequences

print('Hello World')

type('this is a string')

type(None)

type(1)

type(1.0)

type(addNumber)

### tuple & list & dictionary

x = (1, 'a', 2, 'b')
type(x) # tuple, items in which cannot be changed

i = 0
while ( i != len(x) ): # tuple, iterable
    print(x[i])
    i = i + 1

y = [1, 2, 3, 'a', 'b']
type(y) # list

y.append('c') # list, items in which can be changed
y

for yi in y: # list, iterable
    print(yi)

[1, 2] + [3, 4, 5]

[1] * 3 # * 3, repeat 3 times

1 in [3, 2, 1]

x = 'this is a string' # strings, treat it as list of characters
print(x[0])
print(x[3])
print(x[-7])
print(x[2:7])
print(x[:5])

firstname = 'teng'
lastname = 'xu'
print(firstname + ' ' + lastname)
print(firstname * 3)
print('eng' in firstname)

firstname = 'teng ryan oligen xu'.split()[0]
lastname = 'teng ryan oligen xu'.split()[-1]
splitResult = 'teng ryan oligen xu'.split()
print(splitResult)
print(firstname)
print(lastname)

'abc dbf gbi'.split('b')

'abc dbf gbi'.split()


x = ('teng', 'xu', 'wancial@126.com')
firstname, lastname, email = x # unpack
print(firstname)
print(lastname)
print(email)




x = {'7':'David Beckham', '11':'Ryan Giggs'} # key: value, key is the index of value
x['11']

x['18'] = 'Paul Schole'
x['0'] = None
x

for number in x:
    print(x[number])

for name in x.values():
    print(name)

for number, name in x.items():
    print(number)
    print(name)

## The Python programming language: more on strings

print('oligen', 7) 
print('oligen' + str(7))

salesInfo = {'price': 10,
            'numItem': 7,
            'customer': 'oligen'}

salesStatement = '{} bought {} item(s) at a price of {} each for a tatol of {}'

print(salesStatement.format(salesInfo['customer'],
                           salesInfo['numItem'],
                           salesInfo['price'],
                           salesInfo['price'] * salesInfo['numItem']))

## Data files and summary statistics
import csv

%precision 2

with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))


mpg[:3]


len(mpg)

mpg[0].keys()

sum(float(d['cty']) for d in mpg) / len(mpg)
sum(float(d['hwy']) for d in mpg) / len(mpg)


cylinders = set(d['cyl'] for d in mpg)
cylinders

ctyMpgByCyl = []
for c in cylinders:
    summpg = 0
    cyltypecount = 0
    for d in mpg:
        if d['cyl'] == c:
            summpg += float(d['cty'])
            cyltypecount += 1
    ctyMpgByCyl.append((c, summpg / cyltypecount))
ctyMpgByCyl

ctyMpgByCyl.sort(key = lambda x: x[0])
ctyMpgByCyl



vehicleclass = set(d['class'] for d in mpg)
vehicleclass

hwyMpgByClass = []
for t in vehicleclass:
    summpg = 0
    vclasscount = 0
    for d in mpg:
        if d['class'] == t:
            summpg += float(d['hwy'])
            vclasscount += 1
    hwyMpgByClass.append((t, summpg / vclasscount))

hwyMpgByClass.sort(key = lambda x: x[1])
hwyMpgByClass

## The Python programming language: dates and times

import datetime as dt
import time as tm

tm.time()

dtnow = dt.datetime.fromtimestamp(tm.time()) # time stamp
dtnow

dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second

delta = dt.timedelta(days = 100)
delta

today = dt.date.today()
today

today - delta
today > today - delta

weddingDay = dt.date(2018, 5, 26)
daysFromWedding = weddingDay - today
daysFromWedding





## The Python Programming language: objects & maps()

class Person: # Object, CamelCase, the 1st character is capitalized
    department = 'risk management'

    def setName(self, newName):
        self.name = newName
    def setLocation(self, newLocation):
        self.location = newLocation

person = Person()
person.setName('Teng Xu')
person.setLocation('Shanghai')
print('{} lives in {} and works in the department of {} in Orientsec'.
      format(person.name, person.location, person.department))

store1 = [29, 49, 72, 30]
store2 = [82, 20, 49, 61]
cheapest = map(min, store1, store2)
cheapest # map is an object, lazy evaluation which is iterable
# list(cheapest)

for price in cheapest:
    print(price)


people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

people = ('Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero')

def split_title_and_name(person):
    title = person.split()[0]
    lastName = person.split()[-1]
    return '{} {}'.format(title, lastName) # return person.split()[0] + ' ' + person.split()[-1]
list(map(split_title_and_name, people))




## The Python Programming Language: Lambda and list comprehension

my_function = lambda a, b, c: a + b  
# lambda, anonymous function, no default parameters, no complex logical inside the lambda
my_function(1, 2, 3)

my_list = []
for i in range(0, 100):
    if i % 2 == 0:
        my_list.append(i)
my_list[:5] # if the list is too long, it will be displayed just like a column

my_list = [i for i in range(0, 100) if i % 2 == 0] 
# list comprehension, more compact of a format, faster
my_list[:5]




##The Python Programming Language: Numpy

import numpy as np

### creat an array
my_list = [1, 2, 3]
x = np.array(my_list)
x

y = np.array([4, 5, 6])
y

z = np.array([[1, 2, 3], [4, 5, 6]])
z

z.shape # shape of an array

n = np.arange(0, 30, 2)
n
n = n.reshape(3, 5)
n

m = np.linspace(0, 6, 9)
m
m.resize(3, 3)
m

np.ones((2, 3))
np.zeros((3, 5))
np.eye(3) # identity matrix
np.diag(x) # construct a diagnal array
np.diag(z) # extract a diagnal

np.array([1, 2, 3] * 3)
np.repeat([1, 2, 3], 3)

old = np.array([[1, 1, 1],
                [1, 1, 1]])
new = old
new[0, :2] = 0 # old new指向同一个变量，new指向的变量改变，即old指向的变量改变
old

p = np.ones((2, 3), int)
p

### combining arrays
np.array([p, 2 * p]) # shape (2, 2, 3)
np.vstack([p, 2 * p]) # shape (4, 3), stack arrays in sequence vertically(row wise)
np.hstack([p, 2 * p])

### operations
x
y
x + y
x * y
x ** 2

x.dot(y) # inner product

z = np.array([y, y ** 2])
z

z.shape

z.T
z.T.shape

z.dtype

z = z.astype('f')
z.dtype



## the series data structure

import pandas as pd
pd.Series?

animals = ['tiger', 'bear', 'moose']
pd.Series(animals)

numbers = [1, 2, 3]
pd.Series(numbers)

animals1 = ['tiger', 'lion', None]
pd.Series(animals1)
numbers1 = [1, 2, None]
pd.Series(numbers1) #None => NaN, these two are almost equal

import numpy as np
np.nan == None
np.nan == np.nan
np.isnan(np.nan)

sports = {'kongfu': 'china',
         'soccer': 'england',
         'marathon': 'greek'}
s = pd.Series(sports)
s
s.index # index以dict的key首字母排序了

s = pd.Series(['tiger', 'bear', 'moose'], index = ['india', 'america', 'canada'])
s
s1 = pd.Series({'india':'tiger', 
                'america':'bear',
                'canada':'moose'},
               index = ['china', 'america', 'canada'])
s1



## Querying a Series

sports = {'kongfu': 'china',
         'soccer': 'england',
         'golf': 'scotland',
         'marathon': 'greek'}
s = pd.Series(sports)

# loc, iloc methods are in the library of pandas
s.iloc[2]  # query with number using iloc attribute
s.loc['soccer'] # query with index using loc attribute

s[2]  #same as s.iloc['soccer']
s['soccer']


sports = {99: 'Bhutan',
        100: 'scotland',
        101: 'Japan',
        102: 'South Korea'}
s = pd.Series(sports)
s.iloc[0]
s[0] #keyerror, keys in dict are integers


s = pd.Series([1.0, 7.0, 11.0, 17, 23])
s

sum = 0
for i in s:
    sum += i
print(sum)

import numpy as np
sum1 = np.sum(s)
print(sum1)


a = pd.Series(np.random.randint(0, 1000, 10000))
a.head()
len(a)

%%timeit -n 100
sum = 0
for i in a:
    sum += i

%%timeit -n 100
sum = np.sum(a) # vectorization, much faster

a += 2 # add 2 to each item in a using broadcasting
a.head()

for label, value in a.iteritems():
    a.set_value(label, value + 2)
a.head()  ### ?? output wared

%%timeit -n 10
s = pd.Series(np.random.randint(0, 1000, 10000))
for lable, value in s.iteritems():
    s.loc[lable] = value + 2
# s.index.head()  # label ?= index

%%timeit -n 10
s = pd.Series(np.random.randint(0, 1000, 10000))
s += 2

s = pd.Series([1, 2, 3])
s.loc['total'] = 6
s.loc['other'] = 'otherType'  # it will apeend items into a series
s

footballer = pd.Series({11:'ryan', 
                        7:'David',
                        18:'Paul',
                        2:'Garry',
                        3:'Fille'})
footballer

newPlayer = pd.Series(['Lukaku','Pogba','Rashford'],
                     index = [9, 6, 19])
allPlayer = footballer.append(newPlayer)
allPlayer

allPlayer.iloc[7]
allPlayer.loc[7]




## the dataFrame data structure
import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
    'Item Purchased': 'Dog Food',
    'Cost': 22.50}) # series will be aligned by the name of index
purchase_2 = pd.Series({'Name': 'Kevyn',
    'Item Purchased': 'Kitty Litter',
    'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
    'Item Purchased': 'Bird Seed',
    'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store1', 'Store1', 'Store2'])
# df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index = 'Name')  # right?
df.head()


df.loc['Store2']

type(df.loc['Store2'])

df.loc['Store1']

df.loc['Store1', 'Cost']

df.T

df.T.loc['Cost'] #dtype: object

df['Cost'] #dtype: float64

df.loc['Store1']['Cost'] # chaining, it is cost, a copy 

df.loc[:, ['Name', 'Cost']]

df.drop('Store1') # a copy of the original DF

df

copy_df = df.copy()
copy_df = copy_df.drop('Store1')
copy_df

copy_df.drop?

del copy_df['Name'] # delete a column
copy_df

df['Location'] = None # append a column
df



## DataFrame indexing and loading
costs = df['Cost']
costs

costs += 2
df

!cat olympics.csv # shell command, efficient in macOS/Linux

df = pd.read_csv('olympics.csv')
df.head()

df = pd.read_csv('olympics.csv', index_col = 0, skiprows = 1)
df.head()

df.columns

for col in df.columns:
    if col[:2] == '01':
        df.rename(columns = {col: 'Gold' + col[4:]}, inplace = True)  #再理解
    if col[:2] == '02':
        df.rename(columns = {col: 'Silver' + col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns = {col: 'Bronze' + col[4:]}, inplace = True)
    if col[:1] == 'No':
        df.rename(columns = {col: '#' + col[4:]}, inplace = True)
df.head()



## Querying a DataFrame
df['Gold'] > 0

only_gold = df.where(df['Gold'] > 0)
only_gold.head()

only_gold['Gold'].count()
df['Gold'].count()

only_gold = only_gold.dropna()
only_gold.head()

only_gold = df[df['Gold'] > 0]
only_gold.head()

len(df[(df['Gold'] > 0) | (df['Gold.1'] > 0)])

df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]



purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])

df = df.set_index([df.index, 'Name']) # multiple indices
df.index.names = ['Location', 'Name']
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))
df


## Indexing DataFrames
df.head()

df['country'] = df.index
df = df.set_index('Gold')
df.head()

df = df.reset_index()
df.head()

df = pd.read_csv('census.csv')
df.head()

df.['SUMLEV'].unique()

df = df[df['SUMLEV'] == 50]
df.head()

columns_to_keep = ['STNAME', 'CTYNAME', 'BIRTHS2010', 'BIRTHS2011', 'BIRTHS2012', 'BIRTHS2013', 'BIRTHS2014', 'BIRTHS2015', 'POPESTIMATE2010', 'POPEITIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014', 'POPESTIMATE2015']
df = df[columns_to_keep]
df.head()

df = df.set_index(['STNAME', 'CTYNAME'])
df.head()

df.loc['Michigan', 'Washtenaw Country']

df.loc[ [('Michigan', 'Washtenaw Country'), 
            ('Michigan', 'Wayne Country')] ]


## Missing Values
df = pd.read_csv('log.csv')
df

df.fillna?

df = df.set_index('time')
df = df.sort_index()
df

df = df.reset_index()
df = df.set_index(['time', 'user'])
df 

df = df.fillna(method = 'ffill')
df.head()



## Merging DataFrames
import pandas as pd

df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                    {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                    {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                    index = ['Store1', 'Store1', 'Store3'])
df

df['Date'] = ['December 1', 'January 1', 'mid-May']
df

df['Delivered'] = True
df

df['Feedback'] = ['Positive', None, 'Negative']
df

adf = df.reset_index() # the original index is set as a column
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
staff_df.head()
student_df.head()

pd.merge(staff_df, student_df, how = 'outer', left_index = True, right_index = True) # outer, inner, left, right as SQL Join

pd.merge(staff_df, student_df, how = 'inner', left_index = True, right_index = True) # left_index = True, left_index--the original index of staff_df

pd.merge(staff_df, student_df, how = 'left', left_index = True, right_index = True)

pd.merge(staff_df, student_df, how = 'right', left_index = True, right_index = True)

staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
pd.merge(staff_df, student_df, how = 'left', left_on = 'Name', right_on = 'Name')

staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
pd.merge(staff_df, student_df, how = 'left', left_on = 'Name', right_on = 'Name') # left_on, left_index--Name

staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how = 'inner', left_on = ['First Name', 'Last Name'], right_on = ['First Name', 'Last Name'])





# 2018-06-10 am
## Idiomatic Pandas: Making Code Pandorable
import pandas as pd
df = pd.read_csv('census.csv')
df

(df.where(df['SUMLEV'] == 50)
    .dropna()
    .set_index(['STNAME', 'CTYNAME'])
    .rename(columns = {'ESTIMATESBASE2010': 'Estimates Base 2010'}))

df = df[df['SUMLEV'] == 50]
df.set_index(['STNAME', 'CTYNAME'], inplace = True)
df.rename(columns = {'ESTIMATESBASE2010': 'Estimates Base 2010'})


import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']] # why [[]], row <- dataFrame, [[] <- list of columns ] <- index brackets
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

# a = df.loc['Alabama', 'Autauga County'][['POPESTIMATE2010', 'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014', 'POPESTIMATE2015']]
# np.min(a)
# np.max(a)
df.apply(min_max, axis = 1) 
    # 0 or 'index': apply function to each column
    # 1 or 'columns': apply function to each row


import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row
df.apply(min_max, axis = 1)

rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']
df.apply(lambda x: np.max(x[rows]), axis = 1)


## Group by
import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
df = df[df['SUMLEV'] == 50]
df

# %%timeit -n 10
for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME'] == state).dropna()['CENSUS2010POP'])
    print('Counties in state ' + state + ' have an average population of ' + str(avg))

# %%timeit -n 10
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))


df.head()

df = df.set_index('STNAME')
def fun(item):
    if item[0] < 'M':
        return 0
    if item[0] < 'Q':
        return 1
    return 2
for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')


df = pd.read_csv('census.csv')
df = df[df['SUMLEV'] == 50]
df.groupby('STNAME').agg({'CENSUS2010POP': np.average})


print(type(df.groupby(level=0)['POPESTIMATE2010', 'POPESTIMATE2011']))
print(type(df.groupby(level=0)['POPESTIMATE2010']))  # ?是什么


(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg':np.average, 'sum':np.sum}))

(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010', 'POPESTIMATE2011']
    .agg({'avg':np.average, 'sum':np.sum}))

(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010', 'POPESTIMATE2011']
    .agg({'POPESTIMATE2010':np.average, 'POPESTIMATE2011':np.sum}))




# week4
## Distributions in Pandas
import numpy as np
import pandas as pd


np.random.binomial(1, 0.5)
np.random.binomial(1000, 0.5)/1000

chanceOfTornado = 0.01/100
np.random.binomial(100000, chanceOfTornado)

chanceOfTornado = 0.01
tornadoEvents = np.random.binomial(1, chanceOfTornado, 1000000)
two_days_in_a_row = 0
for j in range(1, len(tornadoEvents) - 1):
    if tornadoEvents[j] == 1 and tornadoEvents[j-1] == 1:
        two_days_in_a_row += 1
print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 10000000/365))

np.random.uniform(0, 1)
np.random.normal(0.75)

Formula for standard deviation
$$\sqrt(\frac{1}{N} \sums_{i=1}^N (x_i-\bar(x))^2)$$


distribution = np.random.normal(0.75, size=1000)
np.sqrt(np.sum((np.mean(distribution) - distribution) ** 2) / len(distribution))
np.std(distribution)


import scipy.stats as stats
stats.kurtosis(distribution)
stats.skew(distribution)

chi_squares_df2 = np.random.chisquare(2, size = 10000)
stats.skew(chi_squares_df2)

chi_squares_df5 = np.random.chisquare(5, size = 10000)
stats.skew(chi_squares_df5)


$matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

output = plt.hist([chi_square_df2, chi_square_df5], bins=50, histtype='step', label=['2 degrees of freedom', '5 degrees of freedom'])
plt.legend(loc='upper right')


## Hypothesis Testing
df = pd.read_csv('grades.csv')
df.head()
len(df)

early = df[df['assignment1_submission'] <- '2015-12-31']
late = df[df['assignment_submission'] > '2015-12-31']

early.mean()
late.mean()


from scipy import stats
stats.ttest_ind?
stats.ttest_ind(early['assignment1_grade'], late['assignment1_grade'])

stats.ttest_ind(early['assignment2_grade'], late['assignment2_grade'])

stats.ttest_ind(early['assignment3_grade'], late['assignment3_grade'])















