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
# method：与对象	有关的能够访问其内部数据的函数
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

#### 数值类型
ival = 123456789
ival ** 3

fval = 1.23456
fval1 = 1.23e-7

3 / 2
3 // 2

cval = 1 + 2j # j表示虚数
cval * (1 - 2j)
#### 字符串
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


#### Booleans 布尔值
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


#### Type casting 类型转换
s = '3.14159'
fval = float(s)
type(fval)
int(fval)
bool(fval)


#### None
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


#### Dates and Times
from datetime import datetime, date, time
dt = datetime(2018, 02, 01, 22, 46, 59)
dt.day
dt.mintue
dt.date()
dt.time()

dt.strftime('%m%d%Y %H:%M')

datetime.strptime('20180202', '%Y%m%d')
datetime.datetime(2018, 02, 02, 12, 00)

dt.replace(minute = 0, second = 0)

dt2 = datetime(2018, 05, 26)
delta = dt2 - dt
delta
type(delta)

dt + delta

## 控制流
`if elif else`

if (x < 0):
	print('negative')

if (x < 0):
	print('It\'s negative')
elif (x == 0):
	print('equal to 0')
else:
	print('positive')

a = 5
b = 7
c = 8
d = 4
if a < b or c > d:  # c > d 不会被计算，python立即计算结果
	print('made it')



`for`

seq = [1, 2, None, 4, None, 5]
total = 0
for value in seq:
	if value is None:
		continue
	total += value

seq = [1, 2, 0, 4, 6, 5, 2, 1]
total_til_5 = 0
for i in seq:
	if i == 5:
		break
	total_til_5 += i


`while`

x = 256
total = 0
while x > 0:
	if total > 500:
		break
	total += x
	x = x // 2

`pass` # 空语句、空操作
if x < 0:
	print('negative')
elif x == 0:
	pass
else:
	print('positive')


### 异常处理
float('3.1415')


float('something') # ValueError

# 异常ValueError
def attempt_float(x):
	try:
		return float(x)
	except ValueError: # try语句发生异常时，执行except语句
		return x

attempt_float('3.1415')
attempt_float('something')


float((1, 2)) #TypeError
attempt_float((1, 2)) #TypeError

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

sum = 0
for i in range(10000): # 返回一个用于逐个产生整数的迭代器
	if x % 3 == or x % 5 == 0:
		sum += i



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

tup = tuple('foo', [1, 2], True)
tup[2] = False # TypeErroe，tuple object dose not support item assignment
tup[1].append(3) # ? 怎么解释

(3, None, 'foo') + (6, 0) + ('bar')
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
b_list
b_list[1, 2] = ['oligen', 'say']
b_list

#### list method
b_list.append('world')
b_list

b_list.inset(1, 'ryan') #insert的计算量比append大
b_list

b_list.pop(2) # insert的逆运算
b_list

b_list.append('hello')
b_list.remove('hello') # 删除第一个hello
b_list

'hello' in b_list # 判断元素是不是在list中，python对list采用线性扫面，若判断元素是否在dict或set中，采用基于哈希表的方法，效率高

#### list 合并 排序等
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
import bisec

c = [1, 2, 2, 3, 5, 5, 5, 7]

bisec.bisec(c, 2) #插入到的位置
bisec.bisec(c, 5)

bisec.insort(c, 6) # 插入相应的位置


### 索引
seq = [3， 2， 1, 27, 2, 6, 8, 85]
seq[3:9] # start:stop， start包含在内，stop不包含，元素个数为stop - start
seq[1:3] = ['a', 'b', 'c']
# seq[1:4] = ['a', 'b', 'c'] which one is correct
seq[:5]
seq[2:]
seq[-5:]
seq[-3:-1]

seq[::2] # start stop step
seq[::-1]


#### 内置的序列函数
some_list = ['foo', 'bar', 'zip']
mapping = dict((v, i) for i, v in enumerate(some_list))
# enumerate 逐个返回序列的(i, value)元组
mapping

# sorted 排序
sorted([7, 1, 3, 9, 3, 6, 8])
sorted('horse race')
sorted(set('this is just some string'))

# zip 将多个序列中的元素按对组成tuple
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
zip(seq1, seq2)

seq3 = [True, False]
zip(seq1, seq2, seq3)

for i, (a, b) in enumerate(zip(seq1, seq2)):
	print('%d: %s, %s' % (i, a, b))

pitchers = [('ryan', 'giggs'), ('paul', 'scholes'), ('gary', 'nevil')]
firstName, lastName = zip(*pitchers)
firstName
lastName
# 将元组中的数unzip
# *的用法相当于zip(seq[0], seq[1], ..., seq[len(seq) - 1])

list(reserves(range(10))) # reserved 按逆序迭代序列中的元素


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
del d1[5]	#关键字del，删除k-v
ret = d1.pop('dummy') #方法pop，删除k-v

d1.keys() #返回key的iterator，无序
d1.values() #返回value的iterator

d1.update({'b' : 'foo', 'c' : 12}) #方法update将两个dict合并

#### 元素两两配对，组成字典
# mapping = {}
# for key, value in zip(key_list, value_list):
# 	mapping[key] = value
mapping = dict(zip(range(5), reversed(range(5))))
mapping



# if key in some_dict:
# 	value = some_dict[key]
# else:
# 	value = default_value
value = some_dict.get(key, default_value) #dict的方法get/pop可以接受一个可供返回的默认值

words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = {}
for word in words:
	letter = word[0]
	if letter not in by_letter:
		by_letter[letter] = [word]
	else:
		by_letter[letter].append(word)

by_letter


by_letter = {}
for word in words:
	letter = word[0]
	by_letter.setdefault(letter, []).append(word)

by_letter



from collections import defaultdict
by_letter = defaultdict(list)
for word in words:
	by_letter[word[0]].append(word)









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
a.remove(1)


a_set = {1, 2, 3, 4, 5}
{1, 2, 3}.issubset(a_set)
a_set.issuperset({1, 2, 3})
{1, 2, 3} = {1, 2, 3}

a.isdisjoint(b) #a、b无公共元素，True


## 列表/字典/集合推导式
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]
# dict_comp = {key-expr : value-expr for value in collection if condition}
# set_comp = {expr for value in collection if condition}
unique_lengths = {len(x) for x in strings}
unique_lengths

loc_mapping = {val : index for index, val in enumerate(string)}
loc_mapping
loc_mapping1 = dict((val, idx) for idx, val in enumerate(strings))
loc_mapping1

### 嵌套列表推导式
all_data = [['tom', 'billy', 'jefferson', 'andrew', 'wesley', 'steven', 'joe'], ['susie', 'casey', 'jill', 'ana', 'eva', 'jennifer', 'stephanie']]
names_of_interest = []
for names in all_data:
	enough_es = [name for name in names if name.count('e') >= 2]
	names_of_interest.extend(enough_es)

result = [name for names in all_data for name in names if name.count('e') >= 2]

some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened
# 等价于
# flattened = []
# for tup in some_tuples:
# 	for x in tup:
# 		flattened.extend(x)







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

# np,array会自动为新建的数组推断一个较为合适的数据类型
arr1.dtype
arr2.dtype

# numpy中，创建特定类型的函数
np.zeros(10)
np.zeros(3, 6)
np.empty(3, 5, 2) #以3×（5 × 2）显示；empty返回的不是0，而是垃圾值

a_list = [1, 2, 3, 4, 5]
a_array = np.asarray(a_list)


np.arange(7) #类似于内置的range，但返回的是array，不是list

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
numeric_strings.astype(float) #严格写法numeric_strings.astype(np.float64)，float是python的数据类型，astype函数能将它自动映射到相匹配的numpy数据类型

int_array = np.arange(3)
calibers = np.array([.13, .15, .17], dtype = 'f8') # f8是float64的类型代码
int_array.astype(calibers) 





### 数组和标量的运算
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr
1 / arr
arr ** 0.5

### 数组索引和切片（索引）
arr = np.array(10)
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

arr[5:8].copy() #得到切片的一个副本


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2] #多维数组切片，得到低维数组

arr2d[0][2] #递归索引
arr2d[0, 2] #多维索引
arr2d[:2]
arr2d[:2, 1:]
arr2d[1, :2]
arr2d[2, :1]
arr2d[:, :1] # : 选取整个轴
arr2d[:, 1] # :1时切片索引，此处1是索引，得到的结果不同
arr2d[:2, 1:] = 0

arr3d = np.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
arr3d

arr3d[0]
origin_value = arr3d[0].copy()

arr3d[0] = 42
arr3d

arr3d[0] = origin_value
arr3d

arr3d[1, 0]



#### 布尔型索引
# 布尔型索引选取数组中的数据，总是创建数据的副本
names = np.array(['ryan', 'paul', 'davia', 'gary', 'paul'])
data = np.random.randn(5, 7)

names

data

names == 'paul'
data[names == 'paul'] # 布尔型数组的长度需跟被索引的轴长度一致

data[names == 'paul', 2:]
data[names == 'paul', 3]

names != 'paul'
data[-(names == 'paul')] # - <=> !=

mask = (names == 'paul') | (names == 'ryan')
mask
data[mask]


data[data < 0] = 0 #通过布尔型数组赋值
data[names == 'david'] = 7

#### 其他索引方式
arr = np.empty((8, 4))
for i in range(8):
	arr[i] = i
arr

arr[[3, 7, -1, -5, 6]] # 按指定的顺序索引

arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]] # 得到一维数组，4个元素

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]] # 得到一个4*4的二维数组
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])] # np.ix_函数将两个一维数组组成可以缩阴矩阵的索引器


#### 转置和轴兑换券
arr = np.arange(15).reshape((3, 5))
arr
arr.T

arr = np.random.randn(6, 3)
np.dot(arr.T, arr) #矩阵内积


arr = np.arange(16).reshape(2, 2, 4)
arr
arr.transpose((1, 0, 2)) # 高维数组的转置需要一个由轴编号组成的元组进行轴对换
arr.transpose((1, 2, 0))

arr.swapaxes(1, 2) # 进行轴对换，直接作用的源数据上

### 数组函数
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)


x = np.random.randn(8)
y = np.random.randn(8)
np.maximum(x, y)
arr = np.random.randn(7) * 5
np.modf(arr) # 将小数的整数部分和小数部分分为两个数组


## 数组数据处理
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
xs
ys

# 图如何显示
import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
plt.imshow(z, cmap = plt.cm.gray)
plt.colorbar()

plt.title("image plot of $\sqrt{x^2 + y^2}$ for a grid of values")




### 条件逻辑表述为数组运算
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)] # if cond true xarr, else yarr
result = np.where(cond, xarr, yarr)

arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 1, -1)
np.where(arr > 0, 1, arr)

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

result = np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond3, 2, 3)))

result = 1 * (cond1 - cond2) + 2 * (cond2 & -cond2) + 3 * -(cond1 | cond2)


