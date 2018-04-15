'''import re

string="*****Epoch: 1, precision:  91.28%, recall:  93.19%, f1 score:  92.22%, total cost time: 119, epoch cost time: 55, total loss: 0.042762, char loss: 0.000000, word loss: 0.000000"

precisionRegex= re.compile(r'(?<=precision:\s\s)(\d+\.\d+)')
recallRegex= re.compile(r'(?<=recall:\s\s)(\d+\.\d+)')
f1ScoreRegex= re.compile(r'(?<=f1\sscore:\s\s)(\d+\.\d+)')
totalLossRegex= re.compile(r'(?<=total\sloss:\s)(\d+\.\d+)')

precision=float(precisionRegex.findall(string)[0])
recall=float(recallRegex.findall(string)[0])
f1Score=float(f1ScoreRegex.findall(string)[0])
totalLoss=float(totalLossRegex.findall(string)[0])

print(precision,recall,f1Score,totalLoss)'''

#generating statistics from experiments
import re
import numpy as np
regexTestF1=re.compile(r'(\d+\.\d+)(?=\-)')

#without Language Model
#LSTM
filepath="/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/lstm_vs_cnn/LSTM_experiments.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

lstmTestF1=[float(a) for a in regexTestF1.findall(data)]

#CNN
filepath="/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/lstm_vs_cnn/CNN_experiments.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

cnnTestF1=[float(a) for a in regexTestF1.findall(data)]

#with Language Model
#Language Model LSTM
filepath = "/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/lstm_vs_cnn/Language_Model_LSTM.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

lmLstmTestF1 = [float(a) for a in regexTestF1.findall(data)]


#Language Model CNN
filepath = "/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/lstm_vs_cnn/Language_Model_CNN.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

lmCnnTestF1 = [float(a) for a in regexTestF1.findall(data)]

lstmTestF1=np.asarray([lstmTestF1])
cnnTestF1=np.asarray([cnnTestF1])
lmLstmTestF1=np.asarray([lmLstmTestF1])
lmCnnTestF1=np.asarray([lmCnnTestF1])

print(lstmTestF1.shape)
print(cnnTestF1.shape)
print(lmLstmTestF1.shape)
print(lmCnnTestF1.shape)

print(np.std(lstmTestF1),np.mean(lstmTestF1))
print(np.std(cnnTestF1),np.mean(cnnTestF1))
print(np.std(lmLstmTestF1),np.mean(lmLstmTestF1))
print(np.std(lmCnnTestF1),np.mean(lmCnnTestF1))


