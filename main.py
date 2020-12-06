from data.data_loader import Load
from src.network import Network


LoadTrainingData = Load("images.gz", "labels.gz", data_amt=1000)
LoadTestingData = Load("training_images.gz", "training_labels.gz", data_amt=1000)

trainingData, testingData = LoadTrainingData.data, LoadTestingData.data

net = Network([784, 20, 20, 10])
net.BGD(trainingData, 0.2, 100)
rate = net.accuracy(testingData)
print(rate)

