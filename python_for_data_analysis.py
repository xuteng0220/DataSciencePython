# python data analysis
## appendix
### 变量和按引用传递
a = [1, 2, 3]  
# 变量赋值的过程可以理解为，1.创建一个对象，2.创建一个变量（名），3.将这个变量和对象绑定，变量指向该对象
# 变量使用，以参数形式传递给函数，传入一个引用，不是将对象复制到函数中去
b = a
a.append(4)
print(b)
# a, b是变量，是指向同一个对象的两个引用

### 类型
a = [1, 2, 3]
# a，一个变量，指向一个对象（list）
# [1, 2, 3]，一个（list）对象，包含该对象的类型信息等
a = 6
type(a)
a = 'abc'
type(a)
# 变量a不包含对象的类型信息，其指向的对象包含
c = 3.1415926
isinstance(c, (int, float)) # 判断变量（指向的对象）是否属于某个（些）类型（之一）

### 属性和方法
# attribute：存储在对象内部的其他python对象
# method：与对象    有关的能够访问其内部数据的函数
a = 'manunited'
a.<Tab> # 返回a的所有方法
getattr(a, 'split') # 返回特定方法是否属于该对象


### 引入模块、函数
import moduleName
a = moduleName.funName(...)

import moduleName as defNM
a = defNM.funName(...)

from moduleName import funName
a = funName(...)

from moduleName import funName as defFN
a = defFN(...)


### 二元运算、比较运算
a = [1, 2, 3]
b = a
c = list(a)

b is a # is 判断两个引用是否指向同一对象，is，is not常用来判断变量是否为None
c is a # list函数会创建新的列表

d = 7
e = 11
e // d # 取整除法
e ** d # 幂运算
e & d # and
e | d # or
e ^ d # xor
e | d # 


### 惰性
a = b = c = 5
d = a + b * c # python 急性子的语言，计算结果和表达式都是立即求值的，此处，先计算b * c的结果25，再加上a
# 利用iterator和generator等可以实现惰性/延迟运算，不会立即计算中间结果


### 可变、不可变对象
a_list = ['foo', 2, [4, 5]] # list可变
a_list[2] = (3, 4)
a_list

# 不可变的immutable，是指不能修改内存块的数据。即便修改了，实际是创建了一个信对象，并将其引用赋值给原变量
a_tuple = (3, 5, (3, 4)) # tuple 是不可变对象
a_tuple[1] = 'four'



### 标量类型
|类型|说明|
|--|--|
|None|null值|
|str|字符串|
|float|浮点型|
|bool|布尔型|
|int|整型（带符号整数）|
|long|长整型（带符号整数，任意精度）|

### 数值类型
ival = 123456789
ival ** 3

fval = 1.23456
fval1 = 1.23e-7

3 / 2
3 // 2

cval = 1 + 2j # j表示虚数
cval * (1 - 2j)
### 字符串
a = 'one way of writing a string'
b = "another way"

c = '''
this is a long string that
sapans multiple lines
'''
d = """
to write a multiple string 
in another way 
"""

e = 'string is immutable' # string不可变对象
e[7] = 7 # error
f = e.replace('string', 'longer string') # replace方法是创建了新的对象
f

g = 3.7
h = str(g)

s = 'python'
list(s)
s[:3]

s = '12\\34'  # backslash \, escape character
print(s)


s = r'this\has\no\special\characters' # r''
s

a = 'this is the first half'
b = 'and this is the second half'
a + b


template = '%.2f %s are worth $%d' # 字符串格式化输出
template % (4.567, 'Argentine Pesos', 1)


### Booleans 布尔值
True and True
False and True

a = [1, 2, 3]
if a:
    print('I found something!')

b = []
if not b:
    print('Empty!')

bool([]), bool([1, 2, 3])
bool('Hello World!'), bool('')
bool(0), bool(1)


### Type casting 类型转换
s = '3.14159'
fval = float(s)
type(fval)
int(fval)
bool(fval)


### None
# it’s worth bearing in mind that None is not a reserved keywordbut rather a unique instance of NoneType
a = None
a is None

b = 1
b is not None

def add_and_maybe_multiple(a, b, c=None): # None 作为参数默认值
    result = a + b
    if c is not None:
        result = result * c
    return result
add_and_maybe_multiple(1, 2, 3)

### Dates and Times
from datetime import datetime, date, time
dt = datetime(2018, 6, 22, 9, 45, 59)
dt.day
dt.minute
dt.date()
dt.time()

dt.strftime('%m%d%Y %H:%M')
dt.replace(minute = 0, second = 0)

datetime.strptime('20180202', '%Y%m%d')



dt2 = datetime(2018, 5, 26)
delta = dt2 - dt
delta
type(delta)

dt + delta

## 控制流
### 条件判断
`if elif else`

def equal0(x):
    if (x < 0):
        print('It\'s negative')
    elif (x == 0):
        print('equal to 0')
    else:
        print('positive')
equal0(7)


a = 5
b = 7
c = 8
d = 4
if a < b or c > d:  # c > d 不会被计算，python立即计算结果
    print('made it')


### 循环
`for`

seq = [1, 2, None, 4, None, 5]
total = 0
for value in seq:
    if value is None:
        continue
    total += value
total


seq = [1, 2, 0, 4, 6, 5, 2, 1]
total_til_5 = 0
for i in seq:
    if i == 5:
        break
    total_til_5 += i
total_til_5



`while`

x = 256
total = 0
while x > 0:
    if total > 500:
        break
    total += x
    x = x // 2
total

### 空语句
`pass`

def equal0(x):
    if x < 0:
        print('negative')
    elif x == 0:
        pass #空操作
    else:
        print('positive')
equal0(7)
equal0(0)
equal0(-7)



### 异常处理

float('3.1415')

float('something') # ValueError

# 处理 ValueError
def attempt_float(x):
    try:
        return float(x)
    except ValueError: # try语句发生异常时，执行except语句
        return x

attempt_float('3.1415')
attempt_float('something')


float((1, 2)) #TypeError
attempt_float((1, 2)) #TypeError

# 处理 ValueError TypeError
def attempt_float1(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return x

attempt_float1((1, 2))




f = open(path, 'w')
try:
    write_to_file(f)
finally: # 无论try语句成功与否，finally后的语句都执行
    f.close()


f = open(path, 'w')
try:
    write_to_file(f)
except:
    print('Failed')
else: # try语句成功时，执行else语句
    print('Succeeded')
finally:
    f.close()


range(10) # 返回一个用于逐个产生整数的迭代器
range(0, 20, 2)


seq = [1, 2, 3, 4, 5]
for i in range(len(seq)):
    val = seq[i]
    print(val)

sum = 0
for i in range(10000):
    if (i % 3 == 0) or (i % 5 == 0):
        sum += i
sum


### 三元表达式
x = 5
'Non_negative' if x > 0 else 'Negative' #将一个if-else块转化为一行


## 数据结构
### 元组 tuple
tup = (2, 3, 7, 11, 18)
nested_tup = ((2, 3, 7), (11, 18))
tuple([2, 3, 7])
a_tup = tuple('string')
a_tup[2]


tup = ('foo', [1, 2], True)
tup[2] = False # TypeErroe，tuple object dose not support item assignment
tup[1].append(3) # Q 怎么解释，A tup[1]指向了一个list，list不能变，list指向的元素可以变
tup

(3, None, 'foo') + (6, 0) + ('bar',) # ('bar') is a sting, ('bar',) is a tuple
('foo', 'bar') * 3


#### unpack
tup = (3, 11, 18)
gary, ryan, paul = tup
ryan

tup = (3, 7, (11, 18))
gary, david, (ryan, paul) = tup
ryan

a, b = (1, 2)
a
b
b, a = a, b # 交换变量名c = a, a = b, b =c
a
b

seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a, b, c in seq: # unpacking 长用于对tuple或list序列进行迭代
    sum = a + b + c
    print(sum)

#### tuple method
# 由于tuple的大小和内存不能修改，其方法很少
a = (1, 2, 3, 2, 2, 5, 5, 7)
a.count(2)


### list
a_list = [2, 3, 7, 'go']
tup = (1, 3, 'hello')
b_list = list(tup)
a_list
b_list

b_list[1] = 'oligen'
b_list

#### list method
b_list.append('world')
b_list

b_list.insert(1, 'ryan') #insert的计算量比append大
b_list

b_list.pop(2) # insert的逆运算
b_list

b_list.append('hello')
b_list.remove('hello') # 删除第一个hello
b_list

'hello' in b_list # 判断元素是不是在list中，python对list采用线性扫面，若判断元素是否在dict或set中，采用基于哈希表的方法，效率高

#### list 合并 排序
[2, 'gary'] + [7, 'david']
a = [2, 'gary']
a.extend([7, 'david']) #extend将元素附加到现有列表，比两个列表相加合并（创建新列表合并原有两个列表）的效率高

a = [3, 7, 18, 11, 20]
a.sort()
a

b = ['gary', 'david', 'ryan', 'paul', 'ole']
b.sort(key = len) #按字符串长度排序
b


#### 二分搜素
import bisect

c = [1, 2, 2, 3, 5, 5, 5, 7]
bisect.bisect(c, 2) # 返回插入到的位置
bisect.bisect(c, 5)

bisect.insort(c, 6) # 将元素插入到相应的位置
c

#### 索引
seq = [3, 2, 1, 27, 2, 6, 8, 85]
seq[3:9] # start:stop， start包含在内，stop不包含，元素个数为stop - start
seq[1:3] = ['a', 'b', 'c']
# seq[1:4] = ['a', 'b', 'c'] which one is correct
seq[:5]
seq[2:]
seq[-5:]
seq[-3:-1]

seq[::2] # start stop step
seq[::-1]


### 内置的序列函数
some_list = ['foo', 'bar', 'zip']
mapping = dict((v, i) for i, v in enumerate(some_list))
# enumerate 逐个返回序列的(index, value)元组
mapping

enumerate(some_list)
type(enumerate(some_list))
list(enumerate(some_list))


# sorted 排序
sorted([7, 1, 3, 9, 3, 6, 8])
sorted('horse race')
sorted(set('this is just some string'))

# zip 将多个序列中的元素按对组成tuple
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
a = zip(seq1, seq2)
type(a)
list(a)


seq3 = [True, False]
list(zip(seq1, seq2, seq3))

for i, (a, b) in enumerate(zip(seq1, seq2)):
    print('%d: %s, %s' % (i, a, b))

pitchers = [('ryan', 'giggs'), ('paul', 'scholes'), ('gary', 'nevil')]
firstName, lastName = zip(*pitchers)
firstName
lastName
# 将元组中的数unzip
# *的用法相当于zip(pitchers[0], piichers[1], ..., pitchers[len(seq) - 1])

list(range(10))
list(reversed(range(10))) # reversed 按逆序迭代序列中的元素


### dict
# 哈希映射hash map/相联数组associative array，是一种可变大小的键值对集
emptyDict = {}
d1 = {'a' : 'something', 'b' : [1, 2, 3]}
d1
d1[7] = 'integer'
d1
d1['b']

'b' in d1
d1[5] = 'some value'
d1['dummy'] = 'another value'
d1
del d1[5]    #关键字del，删除k-v
d1
ret = d1.pop('dummy') #方法pop，删除k-v
ret
d1

d1.keys() #返回key的iterator，无序
d1.values() #返回value的iterator

d1.update({'b' : 'foo', 'c' : 12}) #方法update将两个dict合并
d1

#### 元素两两配对，组成字典
# mapping = {}
# for key, value in zip(key_list, value_list):
#     mapping[key] = value
mapping = dict(zip(range(5), reversed(range(5))))
mapping



# if key in some_dict:
#     value = some_dict[key]
# else:
#     value = default_value
value = some_dict.get(key, default_value) #dict的方法get/pop可以接受一个可供返回的默认值

words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter: # key in a dict
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
by_letter

by_letter = {}
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word) # Q？
by_letter

# Q？
from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
by_letter



### set集合
set([2, 2, 2, 3, 1, 3, 3])
{2, 2, 2, 1, 3, 3}

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}
a | b
a.union(b)
a & b
a.intersection(b)
a - b
a.difference(b)
a ^ b #对称差，异或
a.symmetric_difference(b)

a.add(19)
a
a.remove(1)
a


a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)
a_set.issuperset({1, 2, 3})
{1, 2, 3} == {1, 2, 3}

a.isdisjoint(b) #a、b无公共元素，True















### 列表/字典/集合推导式
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]
# dict_comp = {key-expr : value-expr for value in collection if condition}
# set_comp = {expr for value in collection if condition}
unique_lengths = {len(x) for x in strings}
unique_lengths

loc_mapping = {val : index for index, val in enumerate(strings)}
loc_mapping
loc_mapping1 = dict((val, idx) for idx, val in enumerate(strings))
loc_mapping1

#### 嵌套列表推导式
all_data = [['tom', 'billy', 'jefferson', 'andrew', 'wesley', 'steven', 'joe'], ['susie', 'casey', 'jill', 'ana', 'eva', 'jennifer', 'stephanie']]
names_of_interest = []
for names in all_data:
    enough_es = [name for name in names if name.count('e') >= 2]
    names_of_interest.extend(enough_es)
names_of_interest

result = [name for names in all_data for name in names if name.count('e') >= 2]
result

some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened
# 等价于
# flattened = []
# for tup in some_tuples:
#     for x in tup:
#         flattened.extend(x)













# Chap3 IPython交互式计算开发环境
## IPython 基础
a = 5
a

import numpy as np
data = {i : np.random.randn() for i in range(7)}
data

an_apple = 27
an_example = 35
# an<Tab>，按下<Tab>键，自动补全
b = [1, 2, 3]
# b.<Tab>，按下<Tab>键，查看对象所含的方法和属性
# 以下划线开头的方法和属性，包括magic method默认不显示

import datetime
# datetime.<Tab>，查看模块所含的函数等














# chap4 Numpy
## ndarray
# ndarray n维数组对象；题哦同构数据多维容器，其中的元素必须是相同类型
import numpy as np
# np.random.seed(12345)
data = np.random.randn(2, 3)
data
data * 10
data + data
# shape dtype属性
data.shape
data.dtype

### 创建ndarray
# numpy中的array函数
data1 = [1, 2, 3, 5, 7]
arr1 = np.array(data1)
arr1

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2
arr2.ndim
arr2.shape


# np.array会自动为新建的数组推断一个较为合适的数据类型
arr1.dtype
arr2.dtype

# numpy中，创建特定类型的函数
np.zeros(10)
np.zeros((3, 6))
np.empty((3, 5, 2)) #以3×（5 × 2）显示；empty返回的不是0，而是垃圾值

a_list = [1, 2, 3, 4, 5]
a_array = np.asarray(a_list)
a_array

range(7)
np.arange(7) #类似于内置的range，但返回的是array，不是range

np.ones_like(arr2)
np.zeros_like(arr1)
np.eye(3)
np.identity(5)

### ndarray数据类型
arr1 = np.array([1, 2, 3], dtype = np.float64)
arr2 = np.array([1, 2, 3], dtype = np.int32)
arr1.dtype
arr2.dtype
# 当设计的程序涉及数据的存储读取速度时，再考虑数据类型的问题
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64) # astype转换类型，会创建一个新的数组
float_arr.dtype

numeric_strings = np.array(['1.23', '2.34', '3.45', '4.56', '5.67'], dtype = np.string_)
numeric_strings.dtype
numeric_strings.astype(float) #astype不改变原ndarray的数据类型，而是创建新的数据类型的ndarray。严格写法numeric_strings.astype(np.float64)，float是python的数据类型，astype函数能将它自动映射到相匹配的numpy数据类型
numeric_strings.dtype
numeric_float = numeric_strings.astype(float)
numeric_float.dtype

int_array = np.arange(3)
calibers = np.array([.13, .15, .17], dtype = np.float64) 
int_array.astype(calibers.dtype) 
empty_unit32 = np.empty(8, dtype = 'u4') # u4 代表无符号的32位（4字节）整型unit32
empty_unit32



### 数组和标量的运算
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr
1 / arr
arr ** 0.5

### 数组索引和切片（索引）
arr = np.arange(10)
arr
arr[5] # 索引
arr[5:8] # 切片（索引）
arr[5:8] = 12 
arr

arr_slice = arr[5:8]
arr_slice[1] = 12345
arr #数组的切片是原始的数组视图，数据不会被复制，任何修改都会直接反映到源数组上。numpy用于处理大量数据，切片作用于源数据不会因为复制而造成内存和性能的浪费    

arr_slice[:] = 63
arr

arr_slice_copy = arr[5:8].copy() #得到切片的一个副本
arr_slice_copy[:] = 7
arr

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2] #多维数组切片，得到低维数组

arr2d
arr2d[0][2] #递归索引
arr2d[0, 2] #多维索引
arr2d[:2]
arr2d[:2, 1:]
arr2d[1, :2]
arr2d[2, :1]
arr2d[:, :1] # : 表示选取整个轴
arr2d[:, 1] # 与前一个结果不同

arr2d[:2, 1:] = 0
arr2d

arr3d = np.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
arr3d

arr3d[0]
origin_value = arr3d[0].copy()

arr3d[0] = 42
arr3d

arr3d[0] = origin_value
arr3d

arr3d[1, 0]



### 布尔型索引
# 布尔型索引选取数组中的数据，总是创建数据的副本
names = np.array(['ryan', 'paul', 'david', 'gary', 'paul'])
names
data = np.random.randn(5, 7)
data

names == 'paul'
data[names == 'paul'] # 布尔型数组的长度需跟被索引的轴长度一致

data[names == 'paul', 2:]
data[names == 'paul', 3]
data[names == 'paul', 3:4] #有冒号表示选取轴

names != 'paul'
data[~(names == 'paul')] # ~ 等价于 !=

mask = (names == 'paul') | (names == 'ryan')
mask
data[mask]

data[data < 0] = 0 #通过布尔型数组赋值
data[names == 'david'] = 7
data


#### 其他索引方式
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr

arr[[3, 7, -1, -5, 6]] # 按指定的顺序索引

arr = np.arange(32).reshape((8, 4))
arr

arr[[1, 5, 7, 2], [0, 3, 1, 2]] # 得到一维数组，4个元素
arr[[1, 5, 7, 2]]
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]] # 得到一个4*4的二维数组
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])] # np.ix_函数将两个一维数组组成可以索引矩阵的索引器


#### 转置和轴对换
arr = np.arange(15).reshape((3, 5))
arr
arr.T

arr = np.random.randn(6, 3)
np.dot(arr.T, arr) #矩阵内积

arr = np.arange(16).reshape(2, 2, 4)
arr
arr.transpose((1, 0, 2)) # 高维数组的转置需要一个由轴编号组成的元组进行轴对换
arr.transpose((1, 2, 0))

arr.swapaxes(1, 2) # 进行轴对换，返回源数据的视图，不是创建一个新的数据

### 数组函数
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)
arr = np.random.randn(7) * 5
np.modf(arr) # 将小数的整数部分和小数部分分为两个数组


### 数组数据处理
point = np.arange(3)
x, y = np.meshgrid(point, point)
x
y
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
xs
ys

# Q 图如何显示
import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
plt.imshow(z, cmap = plt.cm.gray)
plt.colorbar()
plt.title("image plot of $\sqrt{x^2 + y^2}$ for a grid of values")




### 将条件逻辑表述为数组运算
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)] # if cond true xarr, else yarr
result
result = np.where(cond, xarr, yarr)
result

arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 1, -1)
np.where(arr > 0, 1, arr)

arr = np.arange(25).reshape(5, 5)
arr[np.where(arr > 7)] # Q np.where

cond1 = np.array([True, False, True, False])
cond2 = np.array([False, True, True, False])
result = []
for i in range(4):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)
result

result = np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))
result
result = 1 * (cond1 & ~cond2) + 2 * (cond2 & ~cond1) + 3 * ~(cond1 | cond2)
result




### 数学与统计方法
arr = np.random.randn(5, 4)
arr.mean()
np.mean(arr)
arr.sum()
np.sum(arr)
arr.mean(1) # 列
arr.sum(0) # 行

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)


### 用于布尔型数组的方法
arr = np.random.randn(100)
(arr > 0).sum()

bools = np.array([False, False, True, False])
bools.any() # any 是否存在True
bools.all() # all 是否全为True


### 排序
arr = np.random.randn(8)
arr

arr.sort()
arr

arr = np.random.randn(5, 3)
arr
arr.sort(1)
arr
arr.sort(0)
arr

arr = np.random.randn(1000)
arr.sort()
arr[int(0.05 * len(arr))] # 5%分位数


### unique及集合运算
names = np.array(['bob', 'joe', 'will', 'bob', 'will', 'joe', 'joe'])
np.unique(names)
set(names)

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

| 方法 | 说明 |
| -- | -- |
| unique(x) | 唯一值 |
| intersect1d(x, y) | 交 |
| union1d(x, y) | 并 |
| in1d(x, y) | x的元素是否包含于y |
| setdiff1d(x, y) | x - y |
| setxor1d(x, y) | 对称差 x+y-xy |


## 数组（文件）的输入输出
arr = np.arange(10)
np.save('some_array', arr) # 默认文件后缀 .npy
np.load('some_array.npy')

np.savez('array_achive.npz', a = arr, b = arr) # 将多个array保存到数组压缩文件中
arch = np.load('array_archive.npz')
arch['b']


## 线性代数
x = np.array([[1., 2., 3. ], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)
np.dot(x, np.ones(3))

from numpy.linalg import inv, qr
x = np.random.randn(5, 5)
mat = x.T.dot(x)
inv(mat) # 矩阵的逆
mat.dot(inv(mat))

q, r = qr(mat) # qr分解
r 

|numpy.linalg|说明|
|--|--|
|diag|对角线元素，或转化成对角矩阵|
|dot|内积|
|trace|迹|
|det|行列式|
|eig|特征值|
|inv|逆|
|pinv|Moore-Penrose逆|
|qr|qr分解|
|svd|奇异值分解|
|solve|解方程组|
|lstsq|Ax=b最小二乘解|


## 随机数
samples = np.random.normal(size=(3, 3))
samples

from random import normalvariate
N = 1000000
%timeit samples = [normalvariate(0, 1) for _ in range(N)]
%timeit np.random.normal(size=N)

|numpy.random函数|说明|
|--|--|
|seed|随机数生成器的种子|
|permutation|对一个序列进行随机排列|
|shuffle|对一个序列随机排列|
|rand|均匀分布|
|randint|整数均匀分布|
|randn|正态分布|
|binomail|二项分布|
|normal|正态分布|
|beta|Beta分布|
|chisquare|卡方分布|
|gamma|gamma分布|
|uniform|均匀分布|



## random walk
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
# Q 将随机游走画成图

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()

(np.abs(walk) >= 10).argmax() # 首次距离原点达到10所需的步数

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
draws.shape
np.shape(draws)
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks
walks.shape
walks.max()
walks.min()

hist30 = (np.abs(walks) >= 30).any(1)
len(hist30)
hist30.sum()

crossing_times = (np.abs(walks[hist30]) >= 30).argmax(1)
crossing_times.mean()






# ch4 Pandas
from pandas import Series, DataFrame
import pandas as pd

## Series
# series 由一组数据与一组与之相关联的数据标签（索引）组成
obj = Series([4, 7, -5, 3])
obj

obj.values
obj.index
obj.index = ['a', 'b', 'c', 'd'] # Series的索引可以通过赋值进行改变

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index

obj2['a']

obj2['b'] = 3
obj2[['a', 'b']]

obj2[obj2 > 1]
obj2 * 2
np.exp(obj2)

'b' in obj2
'e' in obj2


sdata = {'ohio': 35000, 'texas': 71000, 'oregon': 16000, 'utah': 5000}
obj3 = Series(sdata)
obj3

state = ['texas', 'ohio', 'utha', 'california']
obj4 = Series(sdata, index=state)


pd.isnull(obj4)
pd.notnull(obj4) # 检测缺失值
obj4.isnull()

obj3 + obj4 # Series数据在运算时能自动对齐索引

obj4.name = 'population'
obj4.index.name = 'state' # Series对象及其索引具有name属性
obj4


## DataFrame
data = {'state': ['ohio', 'ohio', 'ohio', 'nevada', 'nevada'], 'year': [2000, 2001, 2002, 2001, 2002], 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)

DataFrame(data, columns=['year', 'state', 'pop']) # columns可以指定列的顺序

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2 # NaN补齐无数据的column
frame2.columns

### 索引列，对列赋值，新增列，删除列
frame2['state'] #索引
frame2.year #属性的方式

frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(5.)
frame2

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2

frame2['eastern'] = frame2.state == 'ohio' # 创建新列
frame2


del frame2['eastern']  # 删除列
frame2.columns

### 索引行
frame2[3]
frame3.iloc[3]
frame2.loc['two']

pop = {'nevada': {2001: 2.4, 2002: 2,9}, 'ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop) # 外层字典的key作为column，内层字典的key作为index
frame3

frame3.T
DataFrame(pop, index=[2001, 2002, 2003])

frame4 = DataFrame({'ohio': frame3['ohio'][:-1], 'nevada': frame3['nevada'][:2]}) # dataFrame列的数据类型是Series

frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3

frame2.values # 以ndarray的形式返回数据
frame3.values # 数据类型不同


## Series & DataFrame的索引
obj = Series(range(3), index=['a', 'b', 'c'])
obj.index  # index是不可变的，不能对其赋值，obj.index[2] = 'd'，error
obj.index[1:] 

index = pd.index(np.arange(3))
obj2 = Series([1.5, 2.3, 3.7], index=index)
obj2

frame3
'ohio' in frame3.columns
2003 in frame3.index

### 重新索引
obj = Series([4.5, 7.2, 3.7, 9.1], index=['d', 'a', 'b', 'c'])
obj
obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj.reindex(['a', 'b', 'c', 'e']， fill_value=0)

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill') # 以上一个值补充index

|reindex插值|说明|
|--|--|
|ffill/pad|同前一个值|
|bfill/backfill|同后一个值|

frame = DataFrame(np.arange(9).reshape(3, 3), index=['a', 'c', 'd'], columns=['ohio', 'texas', 'california'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd']) # reindex默认对行
frame2

states = ['texas', 'utah', 'california']
frame.reindex(columns=states)
frame

frame.reindex(index=['a', 'b', 'c', 'd'], method=ffill, columns=states) # 可以同时对index和column进行reindex，填充空白值只能针对行

frame.loc[['a', 'c'], states]


### drop 丢弃 删除
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
newObj = obj.drop('c')
newObj
obj.drop(['d', 'c'])

data = DataFrame(np.arange(16).reshape((4, 4)), index=['ohio', 'utha', 'california', 'texas'], columns=['one', 'two', 'three', 'four'])
data.drop(['texas', 'ohio'])
data.drop('two', axis=1)
data.drop(['two', 'four'], axis=1)

### 索引 选取 过滤
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'c']]
obj[[1, 3]]
obj[obj < 2]

obj['b':'d'] # 利用标签进行切片，与python切片运算不同，其包含末端
obj['b':'c'] = 7 # 切片赋值

data = DataFrame(np.arange(16).reshape((4, 4)), index=['ohio', 'utha', 'california', 'texas'], columns=['one', 'two', 'three', 'four'])
data['two']
data[['three', 'one']]
data[:2]
data[data['three'] > 5]

data < 5
data[data < 5] = 0

#### loc iloc
data.loc['utha', ['two', 'three']]
data.loc[['texas', 'california'], [3, 0, 1]]
data.iloc[2]
data.loc[:'california', 'three']
data.loc[data.three > 5, :3]

#### index有重复值时的索引
obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique # index的属性is_unique
obj['a'] # type: Series
obj['c'] # type: value
frame = DataFrame(np.random.rand(4, 3), index=['a', 'a', 'b', 'b'])
df.loc['b']


### 算术运算和数据对齐
s1 = Series([7, 3, 5, 9], index=['a', 'b', 'c', 'd'])
s2 = Series([1, 2, 3, 4, 5], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2 # 按索引取并集，对称差部分以NaN补充

d1 = DataFrame(np.arange(9).reshape((3, 3)), columns=list('bcd'), index=['ohio', 'texas', 'utha'])
d2 = DataFrame(np.arange(12).reshape((4, 3)), columns=list('bde'), index=['ohio', 'texas', 'oregon', 'california'])
d1 + d2 # 取索引和列的并集

d1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
d2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
d1 + d2
d1.add(a2, fill_value=0)
d1.reindex(columns=df2.columns, fill_value=0)

|算术运算|说明|
|--|--|
|add|+|
|sub|-|
|div|/|
|mul|*|


### dataframe与series的运算
arr = np.arange(12.)reshape((3, 4))
arr
arr[0]
arr - arr[0]

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['utah', 'texas', 'ohio', 'oregon'])
series = frame.iloc[0]
frame
series

frame - series

series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2

series3 = frame['d']
frame.sub(series3, axis=0) # 每一列减去相应值


### apply map
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['texas', 'utha', 'ohio', 'oregon'])
np.abs(frame)

f = lambda x: x.max() - x.min()
frame.apple(f)
frame.apple(f, axis=1)

def f(x):
	return Series([x.min(), x.max()], index=['min', 'max'])
a = frame.apple(f)
type(a)
a.dtype
a

format = lambda x: '%.2f' % x
frame.applymap(format) # DataFrame应用applymap函数作用于每一个元素
frame['e'].map(format) # Series应用map函数作用于每一个元素


### 排序 排名

obj = Series(range(4), index=['d', 'a', 'c', 'b'])
obj.sort_index()

frame = DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'], columns=['d', 'c', 'a', 'b'])
frame.sort_index()
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending=False)

obj = Series([1.5, 6.2, 9.1, 2.7])
obj.order()

obj = Series([1.5, np.nan, 6.2, 9.1, np.nan, 2.7])
obj.order() # nan排序后放在末尾

frame = DataFrame({'b': [3, 7, 2, 1], 'a': [7, 9, 6, 3]})
frame.sort_index(by='b')
frame.sort_index(by=['a', 'b'])

obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank() # 给出排序值，rank的method有average(default)，min，max，first
obj.rank(method='first')
obj.rank(ascending=False, method='max')

frame = DataFrame({'b': [3, 7, 2, 1], 'a': [8, 9, 7, 2], 'c': [6, 9, 5, 4]})
frame.rank(axis=1)


### 描述性统计
df = DataFrame([1.5, np.nan], [1.7, 2.9], [np.nan, np.nan], [0.29, -3.8], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
df.sum() # na值被忽略，只对非na起作用
df.sum(axis=0)
df.sum(axis=1)

df.mean(axis=1, skipna=False)
df.idxmax()
df.idxmin()
df.cumsum() # Q默认axis是什么
df.describe()

obj = Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()

|描述性统计方法|说明|
|--|--|
|argmin/argmax|最小/大值的索引位置（整数）|
|idxmin/idxmax|最小/大值的索引值|
|mad|平均绝对离差|
|skew|偏度|
|kurt|峰度|
|diff|一阶差分|
|pct_change|百分比变化|

import pandas.io.data as web

all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')
price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
volume = DataFrame({tic: data['volume'] for tic, data in all_data.iteritems()})

returns = price.pct_change()
returns.tail()

returns.MSFT.corr(returns.IBM) # Series的corr
returns.MSFT.cov(returns.IBM)

returns.corr() # DataFrame的corr
returns.cov() # 协方差矩阵

returns.corrwith(returns.IBM)

returns.corrwith(volume)

obj = Series(['c', 'd', 'a', 'c', 'd', 'b', 'c'])
uniques = obj.uniques()
uniques.sort()
obj.value_counts() # 计数
pd.value_counts(obj.values, sort=False)

mask = obj.isin(['b', 'd'])
mask
obj[mask]

data = DataFrame({'Q1': [1, 3, 4, 3, 4], 'Q2': [2, 3, 1, 2, 3], 'Q3': [1, 5, 2, 4, 4]})
data.apply(pd.value_counts).fillna(0)

### 缺失值处理
string_data = Series(['aardvark', 'artichole', np.nan, 'avocado'])
string_data.isnull()
string_data[0] = None
string_data.isnull() # none也被当作na处理

from numpy import nan as NA
data = Series([1, NA, 3.5, NA, 7])
data.dropna()
data[data.notnull()]

data0 = DataFrame([1, 6.5, 3], [1, NA, NA], [NA, NA, 1], [3, 2, 7]) # 与下面的data有何区别
data = DataFrame([[1, 6.5, 3], [1, NA, NA], [NA, NA, 1], [3, 2, 7]])
data.dropna() # dropna默认丢弃任何含有缺失值的行
data.dropna(how='all') # 丢弃整行为NA的行
data.dropna(axis=1, how='all')

data[5] = NA # 增加一列，全为NA
data.loc[5] = NA # 增加一行 

df = DataFrame(np.random.randn(7, 3))
df.loc[:4, 1] = NA
df.loc[:2, 2] = NA
df
df.dropna(thresh=3) # thresh

### 填充缺失值
df.fillna(0)
df.fillna({1: 7, 3: 11}) # 参数为字典形式，对指定的列填充指定的数值

- = df.fillna(q, inplace=True) # fillna默认返回新对象，inplace对现有对象就地修改
df

df = DataFrame(np.random.randn(6, 3))
df.loc[2:, 1] = NA
df.loc[4:, 2] = NA
df
df.fillna(method='ffill')
df.fillna(method='ffill', limit=3)

data = Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())


### 层次化索引 hierarchical indexing
data = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b','c', 'c', 'd', 'd'], [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]]) # 带有multiindex的Series
data.index
data['b']
data['b': 'c']
data.loc[['b', 'd']]
data[:, 2]

data.unstack() # 将有多重索引的Series转化为DataFrame
data.unstack().stack()

frame = DataFrame(np.arange(12).reshape((4, 3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['ohio', 'ohio', 'california'], ['green', 'red', 'green']])
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame
frame['ohio']
MultiIndex.from_arrays([['ohio', 'ohio', 'california'], ['green', 'red', 'green'], names=['state', 'color']]) # Q?

frame.swaplevel('key1', 'key2') # 调换index的层级
frame.sortlevel(1) # Q？ 针对index，如何针对columns？
frame.swaplevel(0, 1).sortlevel(0)

frame.sum(level='key2')
frame.sum(level='color', axis=1)


### 列作为索引值
frame = DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'two', 'two', 'three', 'one', 'two', 'three'], 'd': [0, 1, 1, 1, 2, 2, 1]})
frame
frame2 = frame.set_index(['c', 'd']) # 将列的值作为索引
frame.set_index(['c', 'd'], drop=False) # 将列的值作为索引，同时保留列
frame2.reset_index()

### 其他
# 整数索引会产生歧义
s = Series(np.arange(3))
s[-1] # 对于未指定索引值的数据，默认索引值为整数，用整数进行索引时可能报错
s.iloc[-1]

s1 = Series(np.arange(3), index=['a', 'b', 'c'])
s1[-1]

s2 = Series(range(3), index=[-5, 1, 3])
s2.iget_value(2)

frame = DataFrame(np.arange(6).reshape((3, 2)), index=[2, 0, 1])
frame.irow(0)

### 面板数据
# panel
import pandas.io.data as web
pdata = pd.Panel(dict((stk, web.get_data_yahoo(stk, '1/1/2009', '6/1/2012')) for stk in ['APPL', 'GOOG', 'MSFT', 'DELL']))
pdata

pdata.swapaxes('item', 'minor')
pdata['Adj Close']
pdata.loc[:, '6/1/2012', :]
pdata.loc['Adj Close', '5/22/2012':, :]

stacked = pdata.loc[:, '5/30/2012', :].to_frame()
stacked

stacked.to_panel() # 将DataFrame转化成panel，to_frame是其逆







# 数据规整：清理、转换、合并、重塑
## 合并数据集
df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df1, df2) # 未指明的情况下，merge将重叠的列作为键进行合并
pd.merge(df1, df2, on='key')

df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df1, df2, left_on='lkey', right_on='rkey') # merge默认是inner连接
pd.merge(df1, df2, how='outer')

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})
pd.merge(df1, df2, on='key', how='left') # 多对多连接产生的是行的笛卡尔积
pd.merge(df1, df2, how='inner')

lef = DataFrame({'key1': ['foo', 'foo', 'bar'],
                 'key2': ['one', 'two', 'one'],
                 'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')

pd.merge(left, right, on='key1') # 被合并的数据，列名称重复
pd.merge(left, right, on='key1', suffixes=('_left', '_right')) # 参数suffixes，对重复列名称加上指定的后缀


## 索引作为链接键进行合并
left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                 'value': range(6)})
left2 = DataFrame({'group_val': [3.5, 7], index=['a', 'b']})
pd.merge(left1, right2, left_on='key', right_index=True) # left_on 左侧DataFrame中用作链接的键，right_index右侧DataFrame以其index作为链接的键
pd.merge(left1, right2, left_on='key', right_index=True, how='outer')

lefth = DataFrame({'key': ['ohio', 'ohio', 'ohio', 'nevada', 'nevada'], 'key2': [2000, 20001, 2002, 20001, 2002], 'data': np.arange(5.)})
righth = DataFrame({np.arange(12).reshape((6, 2)), index=[['nevada', 'nevada', 'ohio', 'ohio', 'ohio', 'ohio'], [2001, 2000, 2000, 2000, 2001, 2002]], columns=['event1', envent2]}) # 层次化索引，多个索引列
pd.merge(letfh, righth, left_on=['key1', 'key2'], right_index=True) 
pd.merge(letfh, righth, left_on=['key1', 'key2'], right_index=True, how='outer')

left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'], columns=['ohio', 'nevada'])
right2 = DataFrame([[7., 8.], [9., 20.], [11., 12.], [13., 14.]], index=['b', 'c', 'd', 'e'], columns=['missouri', 'alabama'])
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)

left2.join(right2, how='outer') # join方法用于合并两个数据框，默认以index作为链接键
left1.join(right1, on='key') # df1的index与df2的列key作为链接键

another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]], index=['a', 'c', 'e', 'f'], columns=['new york', 'oregon'])
left2.join([right2, another])
left2.join([right2, another], how='outer')

## 轴向链接
arr =np.arange(12).rehape((3, 4))
np.concatenate([arr, arr], axis=1) # numpy的concatenate函数，用于ndarray数组的合并

s1 = Series([0, 1], index=['a', 'b'])
s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = Series([5, 6], index=['f', 'g'])
pd.concat([s1, s2, s3]) # pandas函数concat将serieses合并，默认行合并
# [s1, s2, s3] # ？直接能合并吗
# s1 + s2 + s3 # ？
pd.concat([s1, s2, s3], axis=1)

s4 = pd.concat([s1 * 5, s3])
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join='inner')
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result
result.unstack()

pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

df1 = DataFrame(np.arange(6).reshape((3, 2)), index=['a', 'b', 'c'],columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape((2, 2)), index=['a', 'b', 'c'], columns=['one', 'two'])
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])

pd.concat({'level1': df1, 'level2': df2}, axis=1) # dict的key被当作合并时的keys
pd.concat([df1, df2], keys=['level1', 'level2'], names=['upper', 'lower'])

df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
pd.concat([df1, df2], ignore_index=True)


## 有重叠的数据集的合并
a = Series([np.nan, 2.3, np.nan, 3.3, 4.3, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
b = Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
np.where(pd.isnull(a), b, a)
b[:-2].combine_first(a[2:]) # Q b[:-2] a[2:]

df1 = DataFrame({'a': [1., np.nan, 5., np.nan], 'b': [np.nan, 2., np.nan, 6.], 'c': range(2, 18, 4)})
df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.], 'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)


## 数据集重构和数据透视表
data = DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['ohio', 'colorado'], namse='state'), columns=pd.Index=(['one', 'two', 'three']), name='number')
result = data.stack() # 将数据堆栈起来
result.unstack() # 外层index为行，内层index为列
result.unstack(0)
result.unstack('state')

















# 数据聚合和分组运算
# 分组运算的术语split apply combine（拆分 应用 合并）
df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'], 
                'key2': ['one', 'two', 'one', 'two', 'one'], 
                'data1': np.random.randn(5), 
                'data2': np.random.randn(5)})
df

grouped = df['data1'].groupby(df['key1']) # 以数据框中的列data1被分组对象，key1分组键
grouped
grouped.mean()

mean1 = df['data1'].groupby([df['key1'], df['key2']]).mean()
mean1.unstack()

states = np.array(['ohio', 'california', 'california', 'ohio', 'ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean() # 可以是任意长度适当的（无关）数据作为分组键

df.groupby('key1').mean() # 直接将数据框的列名作为参数，作为分组键
df.groupby(['key1', 'key2']).mean()

df.groupby(['key1', 'key2']).size() # 分组过程中，缺失值会被提出（未来版本的pandas中可能用NA代替）

## 对分组进行迭代
for name, group in df.groupby('key1'):
    print name
    print group
for (k1, k2), group in df.groupby(['key1', 'key2']):
    print k1, k2
    print group












































