import matplotlib.pyplot as plt
import numpy as np
import math
import time
from collections import Counter
import cv2



class KNN :


	def __init__(self) :
		time.sleep(1)
		print ("Initializing K Nearest Neighbor model ...")
		print ("")
		time.sleep(2)



	def preparingTrainingData(self, df) :
		self.df = df
		print ("Preparing Training Data ...")
		indexes = []
		Numbers = {}
		for number in range(self.df.shape[0]) :
			nb = df.iloc[number]
			indexes.append(int(nb[0]))
			Numbers[number] = list(nb[1:])

		print ("Finish Preparing...")
		print ("")
		time.sleep(2)
		return Numbers, indexes


	def preparingTestData(self, df) :
		self.df =df
		print ("Preparing Testing Data ...")
		Numbers = {}
		for number in range(self.df.shape[0]) :
			nb = df.iloc[number]
			Numbers[number] = list(nb[:])
		print ("Finish Preparing...")
		print ("")
		time.sleep(2)
		return Numbers


	def visualizationDigits(self, numbers, index, flag = False) :
		self.numbers = numbers
		self.index = index
		self.flag = flag
		if self.flag == True :
			time.sleep(2)
			print ("Visualization...")
			digits = np.asarray(self.numbers[self.index]).reshape(28,28)
			plt.imshow(digits,cmap='gray')
			plt.show()

	def differenceBetweenFrames(self, trainingData, testData) :
		self.trainingData = trainingData
		self.testData = testData

		return math.sqrt(sum([(train - test)**2 for train, test in zip(self.trainingData, self.testData)]))


	def Test(self, TrainSet, TestSet, label, TrainSample=3000, TestSample =25, k=3) :

		time.sleep(2)
		self.TrainSet = TrainSet
		self.TestSet = TestSet
		self.label = label
		self.TrainSample = TrainSample
		self.TestSample = TestSample

		print("Initializing Testing")
		v = []
		vData = []
		for testing in range(self.TestSample) :
			for training in range(self.TrainSample) :
				v.append((self.differenceBetweenFrames(self.TrainSet[training], self.TestSet[testing]), self.label[training]))

			#visualizationDigits(TestSet, testing, True)
			v = sorted(v)
			#print("")
			#print("")
			v = [i for _, i in v ]
			v = v[:k]
			vote = Counter(v)
			#print(v)
			#print("")
			#print(vote)
			#print("")
			w, f = vote.most_common()[0]
			#print w
			vData.append(w)
			v=[]
		print("Finish")
		return vData
	






