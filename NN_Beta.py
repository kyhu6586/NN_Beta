import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from matplotlib import pyplot as plt
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 10, 21, 75, 1
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
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
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
#        self.linear4 = torch.nn.Linear(D_in, D_out, bias=False)
#        self.ELU = torch.nn.ELU()

    def forward(self, x):
#        predicted_force = self.linear4(x)
#        predicted_force = self.ELU(predicted_force)
        hidden_force = self.linear1(x)
        hidden_force2 = self.linear2(hidden_force).clamp(min=0)
        predicted_force = self.linear3(hidden_force2)
        return predicted_force

#    def print_weight(self):
#        #print(self.linear4.weight.data.numpy())
#        return self.linear3.weight.data.numpy()
        


# construct model
import matplotlib as mpl


#read in training data as numpy arrays

length=file_len("1akq_angles+names.txt")
train_len=3*length/4
test_len=(length-(3*length/4))-1
in_train = np.zeros([train_len,1],dtype=np.float32)
out_train = np.zeros([train_len,1],dtype=np.float32)
karplus_train = np.zeros([train_len,3],dtype=np.float32)
residues_train = np.zeros([train_len,20],dtype=np.float32)
in_test = np.zeros([test_len,1],dtype=np.float32)
out_test = np.zeros([test_len,1],dtype=np.float32)
karplus_test = np.zeros([test_len,3],dtype=np.float32)
residues_test = np.zeros([test_len,20],dtype=np.float32)

#CHANGE THIS 
f=open("1akq_angles+names.txt","r")
left=[]
right=[]
mid=[]
for line in f:
    split=line.split()
    obj=Data(split[0],float(split[1]),float(split[2]))
    print(obj.angle)
    if float(obj.angle)<-1:
        left+=[obj]
    elif float(obj.angle)<1:
        mid+=[obj]
    elif float(obj.angle)<6:
        right+=[obj]
train_len=3*length/4
test_len=(length-(3*length/4))-1
data=[]
train_dat=[]
test_dat=[]
train_dat+=left[:2]
train_dat+=right[:2]
if len(mid)<1:
    #raise Exception("INVALID DATA INPUT. Make sure your data includes at least two dihedral angles between -1 and 1")
    data=left[2:]+right[2:]
elif len(mid)==1 or len(mid)==2:
    train_dat+=mid[:]
    data=left[2:]+right[2:]
else:
    train_dat+=mid[:2]
    data=left[2:]+right[2:]+mid[2:]

#Distribute the remaining data randomly 
random.shuffle(data)
train_len-=6
train_data = data[:train_len+1]
test_data = data[train_len+1:]
index=0
in_train = np.zeros([len(train_data),1],dtype=np.float32)
out_train = np.zeros([len(train_data),1],dtype=np.float32)
karplus_train = np.zeros([len(train_data),3],dtype=np.float32)
residues_train = np.zeros([len(train_data),20],dtype=np.float32)
in_test = np.zeros([len(test_data),1],dtype=np.float32)
out_test = np.zeros([len(test_data),1],dtype=np.float32)
karplus_test = np.zeros([len(test_data),3],dtype=np.float32)
residues_test = np.zeros([len(test_data),20],dtype=np.float32)

while index<len(train_data):
    #residue name
    print(train_data[index].resName)
    residues_train[index][dictionary[train_data[index].resName]]=1
    #dihedral angle
    in_train[index]=float(train_data[index].angle)
    #coupling constant
    out_train[index]=float(train_data[index].constant)
    #for karplus equation comparison
    karplus_train[index][0]=(float(1))
    karplus_train[index][1]=(float(math.cos(in_train[index])))
    karplus_train[index][2]=(float(math.cos(2*in_train[index])))
    index+=1
index=0
while index<test_len:
    #residue name
    residues_test[index][dictionary[test_data[index].resName]]=1
    #dihedral angle
    in_test[index]=float(test_data[index].angle)
    #coupling constant
    out_test[index]=float(test_data[index].constant)
    #for the karplus equation comparison
    karplus_test[index][0]=(float(1))
    karplus_test[index][1]=(float(math.cos(in_test[index])))
    karplus_test[index][2]=(float(math.cos(2*in_test[index])))
    index+=1
wexact = np.linalg.pinv(karplus_train).dot(out_train)

A=wexact[0]
B=wexact[1]
C=wexact[2]
in_train_2= np.concatenate((in_train,residues_train),axis=1)
in_test_2=np.concatenate((in_test,residues_test),axis=1)
print(in_train_2)
print(out_train)
print(in_test_2)
print(out_test)
d_dim = in_train_2.shape[0]


# create NN object
model = MyNeuralNet(D_in, H, D_out)

# using a mean squared error residual
criterion = torch.nn.MSELoss(reduction="mean")
#torch.save(model.state_dict(),"~/pythonpractice"

#using the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, momentum=0.9 )

for t in range(10000):

    # This part of the code finds a random part of the training data to feed into the NN
    batch_idx = np.random.choice(d_dim, N, replace=False)
    x = torch.autograd.Variable(torch.from_numpy(in_train_2[batch_idx,:]))
    y = torch.autograd.Variable(torch.from_numpy(out_train[batch_idx]))

    # Run test data through NN
    y_pred = model(x)

    # Calculate Error of NN
    loss = criterion(y_pred, y)

    # Zero fill, and update NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print out loss
    print(t, loss.item(),((y.numpy()-(A + B * np.cos(x[:,0,np.newaxis].numpy())+ C * np.cos(2*x[:,0,np.newaxis].numpy())))**2).mean())

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

d_dim = in_test_2.shape[0]
print(d_dim)
N=10
print(N)
print(length)
batch_idx = np.random.choice(d_dim, N)
x = torch.autograd.Variable(torch.from_numpy(in_test_2[batch_idx,:]))
y = torch.autograd.Variable(torch.from_numpy(out_test[batch_idx]))


    # Run test data through NN
y_pred = model(x)

    # Calculate Error of NN
loss = criterion(y_pred, y)
print(x[:,0,np.newaxis].numpy())
#print(y.numpy())
print(y.numpy())
    # Zero fill, and update NN
print(loss.item(),((y.numpy()-(A + B * np.cos(x[:,0,np.newaxis].numpy())+ C * np.cos(2*x[:,0,np.newaxis].numpy())))**2).mean())


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
