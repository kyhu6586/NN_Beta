import math
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from matplotlib import pyplot as plt
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 5, 21, 50, 1
#Change thissssss
currWeight=0.5
numTests=2000
proteinFile="COMB_angles.txt"
testFile="1IGD_angles.txt"
tvals=[]
MLloss=[]
Kloss=[]
MLloss1=[]
Kloss1=[]
tvals1=[]

def  weighted_mse_loss(inp, target, weight):
    return torch.mean(weight**-2*(inp-target)**2)
class Data:
    resName=""
    angle=0.0
    constant=0.0
    def __init__(self,n,a,c):
        if n!="":
            self.resName=n
            self.angle=a
            self.constant=c
    def __str__(self):
        return self.resName+" "+str(self.angle)

def file_len(fname):
    f= open(fname,"r")
    count=0
    for line in f:
        if line!="" and len(line.split())>1:
            count+=1

    return count+1

def give_file_nam(s,i):
    if os.path.isfile(s+".png"):       
        i+=1
        return give_file_nam(s[:-1]+str(i),i)
    else:
        print(s)
        return s
#Hashtable for residues and their corresponding indeces
dictionary={}
x="ALA"
dictionary[x]=0
x="GLY"
dictionary[x]=1
x="THR"
dictionary[x]=2
x="TYR"
dictionary[x]=3
x="VAL"
dictionary[x]=4
x="LEU"
dictionary[x]=5
x="ILE"
dictionary[x]=6
x="TRP"
dictionary[x]=7
x="GLU"
dictionary[x]=8
x="ASP"
dictionary[x]=9
x="SER"
dictionary[x]=10
x="ASN"
dictionary[x]=11
x="GLN"
dictionary[x]=12
x="PRO"
dictionary[x]=13
x="PHE"
dictionary[x]=14
x="ARG"
dictionary[x]=15
x="CYS"
dictionary[x]=16
x="HIS"
dictionary[x]=17
x="LYS"
dictionary[x]=18
x="MET"
dictionary[x]=19


class MyNeuralNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MyNeuralNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.linear4 = torch.nn.Linear(H,H)
#        self.linear4 = torch.nn.Linear(D_in, D_out, bias=False)
#        self.ELU = torch.nn.ELU()

    def forward(self, x):
#        predicted_force = self.linear4(x)
#        predicted_force = self.ELU(predicted_force)
        hidden_force = self.linear1(x)
        hidden_force2 = self.linear2(hidden_force).clamp(min=0)
        hidden_force3 = self.linear4(hidden_force2)
        predicted_force = self.linear3(hidden_force3)
        return predicted_force

#    def print_weight(self):
#        #print(self.linear4.weight.data.numpy())
#        return self.linear3.weight.data.numpy()
        

train_len=file_len(proteinFile)
test_len=file_len(testFile)
# construct model
import matplotlib as mpl

f=open(proteinFile,"r")
#read in training data as numpy arrays
in_train = np.zeros([train_len,1],dtype=np.float32)
out_train = np.zeros([train_len,1],dtype=np.float32)
karplus_train = np.zeros([train_len,3],dtype=np.float32)
residues_train = np.zeros([train_len,20],dtype=np.float32)
weights = np.ones([train_len,1],dtype=np.float32)
in_test = np.zeros([test_len,1],dtype=np.float32)
out_test = np.zeros([test_len,1],dtype=np.float32)
karplus_test = np.zeros([test_len,3],dtype=np.float32)
residues_test = np.zeros([test_len,20],dtype=np.float32)
index=0
f=open(proteinFile,"r")
for line in f:
    split=line.split()
    if len(split)>1: 
        residues_train[index][dictionary[split[0]]]=1
        residues_train[index][dictionary[split[1]]]=0.2
        #dihedral angle
        in_train[index]=float(split[2])
        #coupling constant
        out_train[index]=float(split[3])
        weights[index]=float(currWeight)
        #for karplus equation comparison
        karplus_train[index][0]=(float(1))
        karplus_train[index][1]=(math.cos(float(split[2])))
        karplus_train[index][2]=(math.cos(2*float(split[2])))
        index+=1
    elif len(split)==1:
        currWeight=split[0]

j=open(testFile,"r")
index=0
for line in j:
    split=line.split()
    if len(split)>1:
        residues_test[index][dictionary[split[0]]]=1
        residues_test[index][dictionary[split[1]]]=0.2
        #dihedral angle
        in_test[index]=float(split[2])
        #coupling constant
        out_test[index]=float(split[3])
        index+=1
wexact = np.linalg.pinv(karplus_train).dot(out_train)
A=wexact[0]
B=wexact[1]
C=wexact[2]
in_train_2 = np.concatenate((in_train,residues_train),axis=1)
in_test_2 = np.concatenate((in_test,residues_test),axis=1)
d_dim = in_train_2.shape[0]
print(weights)
# create NN object
model = MyNeuralNet(D_in, H, D_out)

# using a mean squared error residual
criterion = weighted_mse_loss
#torch.save(model.state_dict(),"~/pythonpractice"

#using the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, momentum=0.9 )
#Find the percent that the ML is better than Karplus
#Find the average distance from expected value for each
for t in range(10000):
    # This part of the code finds a random part of the training data to feed into the NN
    batch_idx = np.random.choice(d_dim, N, replace=False)
    x = torch.autograd.Variable(torch.from_numpy(in_train_2[batch_idx,:]))
    y = torch.autograd.Variable(torch.from_numpy(out_train[batch_idx]))
    w = torch.autograd.Variable(torch.from_numpy(weights[batch_idx]))
    # Run test data through NN
    y_pred = model(x)

    # Calculate Error of NN
    loss = criterion(y_pred, y, w)

    # Zero fill, and update NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print out loss
    tvals+=[t]
    MLloss+=[loss.item()]
    Kloss+=[((w.numpy()**-2)*((y.numpy()-(A + B * np.cos(x[:,0,np.newaxis].numpy())+ C * np.cos(2*x[:,0,np.newaxis].numpy())))**2)).mean()]
    print(t, loss.item(), Kloss[t])
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
d_dim=in_test_2.shape[0]
num=0.0
devML=0.0
devK=0.0
criterion = torch.nn.MSELoss(reduction="mean")
for t in range(numTests):
    batch_idx = np.random.choice(d_dim, N)
    x = torch.autograd.Variable(torch.from_numpy(in_test_2[batch_idx,:]))
    y = torch.autograd.Variable(torch.from_numpy(out_test[batch_idx]))
    # Run test data through NN
    y_pred = model(x)
    # Calculate Error of NN
    loss = criterion(y_pred, y)
    tvals1+=[t]
    MLloss1+=[loss.item()]
    Kloss1+=[((y.numpy()-(A + B * np.cos(x[:,0,np.newaxis].numpy())+ C * np.cos(2*x[:,0,np.newaxis].numpy())))**2).mean()]
    print(str(MLloss1[t])+" "+str(Kloss1[t]))
    if MLloss1[t]<Kloss1[t]:
        num+=1
    devML+=loss.item()
    devK+=((y.numpy()-(A+B*np.cos(x[:,0,np.newaxis].numpy())+C *np.cos(2*x[:,0,np.newaxis].numpy())))**2).mean()
    # Zero fill, and update NN
title="Protein: "+proteinFile[:4]+", Width="+str(H)+", Number of Input Layers="+str(D_in)+"Test on: "+testFile[:4]
filenam=proteinFile[:4]+"_"+str(H)+"_"+str(D_in)+"_"+"1"
filenam=give_file_nam(filenam,1)
testRes=(num/numTests)*100
devML/=numTests
devK/=numTests
print(testRes)
plt.figure(figsize=(20,10))
plt.title(title)
plt.xlabel("Neural Network Pass")
plt.ylabel("Mean Squared Loss per Batch")
plt.text(3000,4,"test results="+str(testRes)+"%")
plt.text(3000,3,str(devML)+", "+str(devK))
plt.ylim(bottom=0,top=20)
plt.plot(tvals,MLloss,'b-',linewidth=0.4)
plt.plot(tvals,Kloss,'r-',linewidth=0.4)
plt.savefig(filenam+".png")
plt.show()
# Things to consider changing
# 1. Different Activation functions. I am using linear ones right now
#  but we should see what others are using. This could result in a better
# model.
# 2. Different Optimizer function. I am using Adam right now, which seems
# to be what everyone else is using, but it could be interesting to see what
# else we could use.
# 3. Different objective functions. I think that MSE makes a lot of sense for
# my project and Kylie's project, but I think we should take a look at the other options.
# 4. Model Hyperparameters. Obviously, we should always try to find the best hyperparameters.
# 5. Different Biases. I am using clamp right now, but I am not sure that one is the best. 
