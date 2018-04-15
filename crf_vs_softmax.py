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
filepath="/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/crf_vs_softmax/crf-lstm.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

crfLstmTestF1=[float(a) for a in regexTestF1.findall(data)]

#CNN
filepath="/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/crf_vs_softmax/crf-cnn.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

crfCnnTestF1=[float(a) for a in regexTestF1.findall(data)]

#with Language Model
#Language Model LSTM
filepath = "/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/crf_vs_softmax/softmax-lstm.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

softmaxLstmTestF1 = [float(a) for a in regexTestF1.findall(data)]


#Language Model CNN
filepath = "/Users/jayavardhanreddy/Study_Slides/Second_Semester/5249_Inter-DB/Experiments/crf_vs_softmax/softmax-cnn.txt"

with open(filepath, 'r') as myfile:
    data = myfile.read()

softmaxCnnTestF1 = [float(a) for a in regexTestF1.findall(data)]

#combined
totalCrf=crfLstmTestF1+crfCnnTestF1
totalSoftmax=softmaxLstmTestF1+softmaxCnnTestF1

crfLstmTestF1=np.asarray([crfLstmTestF1])
crfCnnTestF1=np.asarray([crfCnnTestF1])
softmaxLstmTestF1=np.asarray([softmaxLstmTestF1])
softmaxCnnTestF1=np.asarray([softmaxCnnTestF1])
totalCrf=np.asarray([totalCrf])
totalSoftmax=np.asarray([totalSoftmax])

print(crfLstmTestF1.shape)
print(crfCnnTestF1.shape)
print(softmaxLstmTestF1.shape)
print(softmaxCnnTestF1.shape)

print('*********')
print('crf lstm vs softmax lstm')
print(np.std(crfLstmTestF1),np.mean(crfLstmTestF1),np.max(crfLstmTestF1))
print(np.std(softmaxLstmTestF1),np.mean(softmaxLstmTestF1),np.max(softmaxLstmTestF1))


print('*********')
print('crf cnn vs softmax cnn')
print(np.std(crfCnnTestF1),np.mean(crfCnnTestF1),np.max(crfCnnTestF1))
print(np.std(softmaxCnnTestF1),np.mean(softmaxCnnTestF1),np.max(softmaxCnnTestF1))


print('*********')
print('crf total vs softmax total')
print(np.std(totalCrf),np.mean(totalCrf),np.max(totalCrf))
print(np.std(totalSoftmax),np.mean(totalSoftmax),np.max(totalSoftmax))




