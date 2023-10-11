# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy import stats



def createArr(min, max,row,col):
	return np.random.randint(min,max,[row,col])

# arr 二维数组
def getRowArr(arr,row):
	return arr[row,:]

# arr 二维数组
def getColArr(arr,col):
	return arr[:,col]

# arr 一维数组 非重复集合
def getCollection(arr):
	return np.unique(arr)

# arr 一维数组 指定值寻找位置,返回数组
def getIndex(arr,x):
	return np.where(arr == x)[0]

# 第i个应用所占内存大小
def appOccupyMem(i):
	return FAPPSIZE+abs(math.sin(i))*math.pow( i, 0.98 )

# 多个应用所占内存大小
def appsOccupyMem(arr):
	total = 0 ;
	for i in arr :
		total = total + appOccupyMem(i)
	return total

# 第j台机器内存总大小
def linuxNodeMemSize(j):
	return FNODESIZE + math.floor(j/5)*2*1024

# 第j台机器内存总大小
def currentNodeTotalMem(arr):
	totalMem = 0 
	for value in arr:
		totalMem = totalMem + appOccupyMem(value)
	return totalMem

# arr向量里能否在i位置放j，即j机器能否再加第i个应用
def putAppLimit(arr,i,j):
	return linuxNodeMemSize(j)*0.8 >= (currentNodeTotalMem(getIndex(arr,j)) + appOccupyMem(i))

# 创建一组解向量
def createVector(IMAX,JMAX):
	arr = np.random.randint(1,size=IMAX)
	# arr = np.array([])
	index = 0 ;
	while index < IMAX:
		j = random.randint(1, JMAX)
		
		errNodeCollection = set()
		while not putAppLimit(arr,index,j):
			errNodeCollection.add(j)
			if len(errNodeCollection) == JMAX:
				raise ValueError("当前机器内存无法支撑现有应用")
			j = random.randint(1, JMAX)
			
		arr[index] = j	
		# arr = np.append(arr,int(j))
		index = index + 1

	# print(arr)
	return arr
	
# 返回每台node上剩余内存大小
def fitVector(arr):
	leftMemNodeSet = np.array([])
	jcollection = set(arr)
	for j in jcollection:
		icollection = getIndex(arr,j)
		leftMemNodeSet = np.append(leftMemNodeSet, linuxNodeMemSize(j) - appsOccupyMem(icollection))
	# print(leftMemNodeSet)
	return leftMemNodeSet



def initArr(M, IMAX, JMAX):
	arr = createArr(0,1,M,IMAX)
	for index in range(M):
		index = index - 1
		arr[index,:]=createVector(IMAX,JMAX)
	return  arr

def drawImage(leftMemNodeArr,title):
	xarr = np.arange(1, len(leftMemNodeArr) + 1, 1, dtype=int)
	plt.figure(facecolor='lightblue')
	plt.plot(xarr,leftMemNodeArr,color='orangered')
	ax=plt.gca()
	# ax.set_facecolor("yellowgreen")
	ax.set_title(title + "")
	plt.show()

def statistic(x):
    # Get only the `shapiro` statistic; ignore its p-value
    return stats.shapiro(x).statistic

# 二维数组 适应度评价
def fitArr(arr,IMAX,M,P):
	gaussSortDic = {}
	rows, cols = arr.shape
	for index in range(rows):
		index = index - 1
		if index <0 :
			pass

		# Shapiro-Wilk test
		#scipy.stats.shapiro适用于小样本数据，只能检查正态分布。
		leftMemNodeSet = fitVector(arr[index,:])
		s,p = stats.shapiro(leftMemNodeSet)
		# print(s,p)

		# 记录解Vector以及适应度评价p
		gaussSortDic[round(p, 6)] = {
			"leftMemNodeSet":leftMemNodeSet,"orginVector":arr[index,:]}
		# drawImage(leftMemNodeSet,str(index))


	# 轮盘赌算法 选择概率计算 f=fit/totalFit,f越大，选择概率越高
	# 按照正态分布p降序排列
	sortKeysList = sorted(gaussSortDic.keys(),reverse = True)
	print(sortKeysList[:int(math.floor(M*P))])

	vectorArr = np.arange(M*P*IMAX).reshape((M*P, IMAX))
	index = 0
	for key in sortKeysList[:int(math.floor(M*P))]:
		vectorArr[index,:] = gaussSortDic[key]["orginVector"]
		# drawImage(dictResult[key],str(key))
		index=index+1
	# print("vectorArr")
	# print(vectorArr)


	return vectorArr

# 获取二维数组的行数
def getRows(arr):
	if arr == None:
		return 0
	resultRows,cols = arr.shape
	return resultRows

# cross , 两个一维数组, 交叉概率
def crossTool(arr1,arr2,PC):
	arr3 = np.empty(len(arr1))
	arr4 = np.empty(len(arr2))
	for idx,value in enumerate(arr1):
		arr3[idx] = value
		arr4[idx] = arr2[idx]
		if random.random() <= PC:
			if putAppLimit(arr1,idx,arr2[idx]) and putAppLimit(arr2,idx,arr1[idx])  and value != arr2[idx]: 
				arr3[idx] = arr2[idx]
				arr4[idx] = value
	return arr3,arr4

#  交叉
def cross(arr,M,P,PC):
	# 结果
	rows,cols = arr.shape

	#
	newRows = M*(1-P)
	resultArr = np.empty([0,cols],dtype = int)

	# vector自上而下开始0-1，0-2，，，0-19； 1-2，1-3，，，，1-19；2-3，，，
	index = 0
	while (index + 1) < rows:
		j = index + 1
		while  j < rows:
			arr1,arr2 = crossTool(arr[index,:],arr[j,:],PC)
			if getRows(resultArr) <= (newRows-2):
				# print(arr[index,:])
				# print(arr[j,:])
				# print(arr1 ==arr[index,:])
				# print(arr2 == arr[j,:])
				resultArr = np.append(resultArr,[arr1],axis=0)
				resultArr = np.append(resultArr,[arr2],axis=0)
			else:
				break
			j = j + 1
		if (getRows(resultArr) >= newRows):
			break
		index = index - 1

	return resultArr

def checkConfig(M,P):
	top = M*P
	if (top/2*(top-1)+top) < M:
		raise ValueError("当前配置无法完成训练")



#  变异
def mutation(arr):
	return arr

 	

FAPPSIZE = 500 # app占用内存大小影响因子	
FNODESIZE = 8 * 1024 # node内存大小影响因子
IMAX = 360 # app数量
JMAX = 30 # node数量
M = 180	  # 种群大小
P=0.2 #选择基数率
PC = 0.55  # 交叉率
PM = 0.15  # 变异率
G = 100  # 终止迭代代数

checkConfig(M,P)
arr = initArr(M, IMAX, JMAX)
index = 0
while index <G:
	# 计算适应度 + 选择概率
	goodArr = fitArr(arr,IMAX,M,P)

	# 交叉
	arr = cross(goodArr,M,P,PC)

	# 变异
	mutation(arr)

	index = index +1 





# 1) 首先寻找一种对问题潜在解进行“数字化”编码的方案。（建立表现型和基因型的映射关系）

# 2) 随机初始化一个种群（那么第一批袋鼠就被随意地分散在山脉上），种群里面的个体就是这些数字化的编码。

# 3) 接下来，通过适当的解码过程之后（得到袋鼠的位置坐标）。

# 4) 用适应性函数对每一个基因个体作一次适应度评估（袋鼠爬得越高当然就越好，所以适应度相应越高）。

# 5) 用选择函数按照某种规定择优选择（每隔一段时间，射杀一些所在海拔较低的袋鼠，以保证袋鼠总体数目持平。）

# 6) 让个体基因变异（让袋鼠随机地跳一跳）。

# 7) 然后产生子代（希望存活下来的袋鼠是多产的，并在那里生儿育女）。



# print(getRowArr(arr,3))
# print(getColArr(arr,3))
# print(getCollection(getRowArr(arr,1)))
# print(getIndex(getRowArr(arr,1),5))
# print(linuxNodeMemSize(2))
# print(appOccupyMem(2))

