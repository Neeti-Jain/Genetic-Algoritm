# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:21:59 2018

@author: Neeti
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Importing Dataset
print("Genetic Algorithm for regression " )
print("Submitted By :: Neeti Jain - 18200683")

# Reading csv file thru Pandas library
dataframe = pd.read_csv("Project 1 - Dataset.csv")
s = dataframe.iloc[:,[0,1,2,3,4]]
y = dataframe.iloc[:,13]   

# Randomly split dataset 
X_train, X_test, Y_train, Y_test = train_test_split( s, y, test_size = 0.25)

# Weights
P=5
N=10 
weights=P*N

# – Normalise the training dataset ( 0 and 1 )

#Normalise Input Training dataset 
xmax, xmin = X_train.max(), X_train.min()
X_train = ( X_train - xmin ) / ( xmax - xmin )
X_train= np.matrix(X_train)

#Normalise Output from Training dataset
ymax, ymin = Y_train.max(), Y_train.min()
Y_train = ( Y_train - ymin ) / ( ymax - ymin )
Y_train = Y_train.reset_index(drop=True) 

#Normalise Input Testing dataset 
xmax, xmin = X_test.max(), X_test.min()
X_test = (X_test - xmin)/(xmax - xmin)
X_test_orig = X_test
X_test= np.matrix(X_test)

#Normalise Output Testing dataset 
ymax, ymin = Y_test.max(), Y_test.min()
Y_test = (Y_test - ymin)/(ymax - ymin)
Y_test_orig = Y_test
Y_test = Y_test.reset_index(drop=True)

# Valiable declaration
test = 63
yhat_sumlist=list()

global_test_fitness = list()
test_chrome_list1 = []
# Define Methods

# Calculate eq1
def calculate_eq1(temp):
    f_value = 0
    for i in range(N):
        f_value = f_value + (1 / ( 1 + np.exp(-temp.item(i)) ) ) 
    return f_value

# Calculate eq2 with the help of eq1  return Chromosome and fitness value
def eq1_eq2(weight):
 yhat_sum = 0
 yhat_sq = 0
 for i in range(train):
         temp = np.matmul (  X_train[[i],:]  ,  weight)
         yhat_sum = calculate_eq1(temp)
         yhat_sq = yhat_sq + np.square( yhat_sum - Y_train[i])
         
 wmax,wmin = weight.max(),weight.min()
 fitness_value = (1- (yhat_sq/train))*100
 normalise_w = weight
 normalise_w = (normalise_w - wmin)/(wmax-wmin)
 normalise_w= np.matrix(normalise_w)
 normalise_w = np.around(normalise_w*1000)
 c=normalise_w.reshape(1,-1)
 chrome = convert(c)
 return chrome,fitness_value
### Function Ends #####


def convert(c):
    chromosome=""
    for j in range(weights):
        # Binarise
         chr = bin( int (c.item(j)))[2:].zfill(10)
         chromosome=chromosome + str(chr)+""
    return chromosome
### Function Ends #####


# Do the crossover
def crossover(pr,cr):
    a = pr[0:250] + cr[250:]
    b = cr[0:250] + pr[250:]
    return a,b
### Function Ends #####

# Mutate 
def mutate(a,b):
     l1=list(a)
     l2=list(b)
     ninja=list()
     for x in range(int((P*N*10)* 0.05 )):
          ninja.append( random.randint(1,P*N*10 - 1))

     ninja.sort() 
     for i in ninja:
          if l1[i]=="0":
              l1[i]="1"
          else:
              l1[i]="0"
     for i in ninja:
          if l2[i]=="0":
              l2[i]="1"
          else:
              l2[i]="0"
     c = "".join(l1)
     d = "".join(l2)
     
     return c,d   
### Function Ends #####
     
 # Debinarization
def debinarization(c,d):
    X = []
    Y=[]
    for i in range(0,weights*N,N):
        x = int(c[i:i+10],2)/1000
        y = int(d[i:i+10],2)/1000
        X= np.append(X,x)
        Y= np.append(Y,y)
    lower, upper = -1, 1
    X_norm = [lower + (upper - lower) * x for x in X]  
    Y_norm = [lower + (upper - lower) * y for y in Y]  
    return  X_norm,Y_norm
 ### Function Ends #####


#  Calculate eq2 for Testing data
def test_eq1_eq2(w):
 yhat_sum = 0
 yhat_sumlist.clear()
 for i in range(test):
         temp = np.matmul (  X_test[[i],:]  ,  w)
         yhat_sum = calculate_eq1(temp)
         yhat_sumlist.append( yhat_sum)
         
### Function Ends ##### 
         
Npop =500
fitness_value=list()
fitness_value2=list()
wx=list()
Z_norm=list()
global_fitness = list()
learner_weight = list()
chrome_list=list()
chrome_list1 = list()
chrome_list2 = list()
train = 189
fittest = 0

for i in range(Npop):
    wx.append(np.matrix((np.random.uniform(low=-1,high=1,size=(P,N)))))

for u in range(Npop):
    chrome, f= eq1_eq2(np.array(wx[u]))
    fitness_value.append(f)
    if fittest == 0 or fittest < f:
            fittest = f 
            print("Parent( Fittest ) :: ",fittest)
            parent = wx[u]
            parent_chrome = chrome 
            global_best_weight = wx[u]
            global_fitness.append(fittest) 
    chrome_list1.append(chrome)        

#print("Size of chrome_list",len(chrome_list))
for u in range(Npop): 
    current_cr = chrome_list1[u] 
    
    # Crossover
    a,b = crossover(parent_chrome,current_cr)
    
    # Mutate
    c,d = mutate(a,b)
    
    # Debinarization
    X_norm,Y_norm = debinarization(c,d) 
    Y_norm = np.matrix(Y_norm)
    X_norm = np.matrix(X_norm)
    
    #Create Weight Matrix
    Z_norm.append(X_norm.reshape(P,N))
    Z_norm.append(Y_norm.reshape(P,N))


#  ************ Set loop size  ************
    # Change loop size if u need quick ans
    
loop  = 10
for ss in range(loop):
    print("Loop No " , ss+1 , " running out of " ,loop)
    # Clear All List
    fitness_value.clear()
    chrome_list.clear()
    chrome_list1.clear()
    chrome_list2.clear()
    fitness_value2.clear()
    learner_weight.clear()
    
    for u in range(len(Z_norm)):
        chrome, f= eq1_eq2(np.array(Z_norm[u]))
        fitness_value.append(f)
        if fittest == 0  or fittest < f:
            fittest = f 
            print("Parent( Fittest ) :: ",f)
            parent = Z_norm[u]
            parent_chrome = chrome 
            global_best_weight = Z_norm[u]
            global_fitness.append(fittest)
        chrome_list1.append(chrome) 
#        learner_weight.append(Z_norm[u])
    
    
    fitness_value2 = fitness_value   
    fitness_value2.sort()
    mid = fitness_value2[int(len(fitness_value2)/2)]

    chrome_list2.clear();
    
    for x in range(len(fitness_value)):
        if len(chrome_list2) >=len(fitness_value2)/2:
            break;
        if float(mid) <= float(fitness_value[x]):    
                chrome_list2.append(chrome_list1[x])
                
    Z_norm.clear()
    for u in range(len(chrome_list2)):
        current_cr = chrome_list2[u]
        a,b = crossover(parent_chrome,current_cr)
        c,d = mutate(a,b)
    # step 14 debinarization
        X_norm,Y_norm = debinarization(c,d) 
        Y_norm = np.matrix(Y_norm)
        X_norm = np.matrix(X_norm)
        Z_norm.append(X_norm.reshape(P,N))
        Z_norm.append(Y_norm.reshape(P,N))
        
 # Plot

plt.scatter(global_fitness,range(len(global_fitness)), marker="o")
plt.title("Scatter plot 2D ")
plt.ylabel("Iterations")
plt.xlabel("Fitness Value")
plt.show()       

   
test_eq1_eq2(global_best_weight)
    
X0_Input_weight = X_test_orig[["Weight lbs"]]
X1_Input_Height = X_test_orig[["Height inch"]]

df_f = pd.DataFrame({"fitness":yhat_sumlist})
yhat_sq_test = 0    
#plt.title("3D Scatter plot Project 1 ")

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("Weight")
ax.set_ylabel("Height")
ax.set_zlabel("Output Y")
ax.scatter(X0_Input_weight, X1_Input_Height, Y_test)
ax.scatter(X0_Input_weight, X1_Input_Height, df_f)
plt.title("3d Scatter Plot")
plt.show()

for i in range(test):
    yhat_sq_test += np.square( yhat_sumlist[i] - Y_test[i])

overall_error = yhat_sq_test/test
print( "The Overall Error Calculated is ::  " ,overall_error)
