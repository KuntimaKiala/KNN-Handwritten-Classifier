from KuntimaClassifiers import KNN
import pandas as pd
import os

print ("Kaggle Competition : Digit Recognizer\n\n ")
print ("Loading Data ...")
path = '../data'
dataTest = os.listdir(path)[0]
dataTrain = os.listdir(path)[2]
dfTraining = pd.read_csv(os.path.join(path,dataTrain ),skipinitialspace=True)
dfTest = pd.read_csv(os.path.join(path,dataTest ),skipinitialspace=True)



def KnnDetector(dfTraining, dfTest) :
    
    model = KNN()
    train, label = model.preparingTrainingData(dfTraining)
    test =  model.preparingTestData(dfTest)
    #model.Test(train,test,label, TrainSample=3000, TestSample = 25, k = 3)
    result = model.Test(train,test,label,TrainSample=5000, k=5)
    return result 

if __name__ == "__main__" :
    result = KnnDetector(dfTraining, dfTest)
    print result
