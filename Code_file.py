# Importing all the required libraries
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import statistics as st
import pandas as pd
import math



#Getting the path of the python file
dirpath = os.path.dirname(__file__)
file_list=[]
file_name=[]

# Walking CSV file of the dirpath  and storing them as an array
for root,dirs,files in os.walk(dirpath):
    for file in files:
       if file.endswith(".csv"):
           len_filename=len(file)
           file_name.append(file[:len_filename-4])
           x= os.path.join(dirpath, file)
           y = np.genfromtxt(x, delimiter=",")
           file_list.append(y)


# Creating a dictionary which has key as a file names and value is the array of the file content
dic_files=dict(zip(file_name,file_list))

# print(dic_files.keys())


#Storing the 4 array's of each dataset(train,targetvalues of train,test, targetvalues of test)

#Dataset1: 10 Features and 100 Examples
k1=dic_files.get('train-100-10' )
k2=dic_files.get('trainR-100-10')
k3=dic_files.get('test-100-10')
k4=dic_files.get('testR-100-10')

#Dataset2:100 Features and 100 Examples
l1=dic_files.get('train-100-100' )
l2=dic_files.get('trainR-100-100')
l3=dic_files.get('test-100-100')
l4=dic_files.get('testR-100-100')

#Dataset3:100 Features and 1000 Examples
m1=dic_files.get('train-1000-100' )
m2=dic_files.get('trainR-1000-100')
m3=dic_files.get('test-1000-100')
m4=dic_files.get('testR-1000-100')

#Dataset4: Wine dataset
w1=dic_files.get('train-wine' )
w2=dic_files.get('trainR-wine')
w3=dic_files.get('test-wine')
w4=dic_files.get('testR-wine')



#Dataset5:Crime dataset
c1=dic_files.get('train-crime' )
c2=dic_files.get('trainR-crime')
c3=dic_files.get('test-crime')
c4=dic_files.get('testR-crime')


#Task1


# Defining the function which calculates array w for each lambda.
# The output of the function is a dictionary with lamba as key and w array as values.
def finding_w(train,tvalues):
    list1=[]
    list2=[]
    for i in range(151):
        x=np.transpose(train).dot(tvalues)
        w=np.array((np.linalg.inv(i*np.identity(train.shape[1])+np.transpose(train).dot(train))).dot(x))
        m=w.shape[0]
        list1.append(i)
        # list2.append(w.reshape(1,m))
        list2.append(w)
    w_dic=dict(zip(list1,list2))
    return w_dic


# Defining the MSE function which calculates the MSE for each lamba.
# The output of the function is a dictionary with lamba as key and MSE as values.
def MSE(train,tvalues,test,testtvalues):
    list1=[]
    list2=[]
    for k in finding_w(train,tvalues).keys():
        w=(finding_w(train,tvalues))
        sum=0
        N = len(testtvalues)
        for i in range(0,len(testtvalues)):
            MSE=np.square((np.transpose(test[i]).dot(w.get(k))-testtvalues[i]))
            sum+=MSE
        list1.append(k)
        list2.append(sum/N)
    MSE_dic=dict(zip(list1,list2))
    return(MSE_dic)

#Labelling for desired output
print("Observations for Task 1")
print("for 10 Features and 100 Examples")
# Calling the MSE function for train and test for Dataset1
Train_100_10=MSE(k1,k2,k1,k2)
Test_100_10=MSE(k1,k2,k3,k4)


#
# #Ploting the graph of MSE's for training and testing dataset by lambda
# plt.plot(Train_100_10.keys(),Train_100_10.values(),"r--",label='Train-100-10',marker='o')
# plt.plot(Test_100_10.keys(),Test_100_10.values(),"b--",label='Test-100-10',marker='o')
# plt.axhline(y=3.78, color='y', linestyle='-')
# plt.xlabel('lamda-regularization parameter')
# plt.ylabel('MSE')
# plt.title('Variation of MSE with lamda -10 Features and 100 Examples ')
# plt.legend()
# plt.show()
#
#
# # #Labelling for desired output
# print("for 100 Features and 100 Examples")
#
# # Calling the MSE function for train and test for Dataset2
# Train_100_100=MSE(l1,l2,l1,l2)
# Test_100_100=MSE(l1,l2,l3,l4)
#
# #Ploting the graph of MSE's for training and testing dataset by lambda
# plt.plot(Train_100_100.keys(),Train_100_100.values(),"r--",label='Train-100-100',marker='o')
# plt.plot(Test_100_100.keys(),Test_100_100.values(),"b--",label='Test-100-100',marker='o')
# plt.axhline(y=3.78, color='g', linestyle='-')
# plt.xlabel('lamda-regularization parameter')
# plt.ylabel('MSE')
# plt.title('Variation of MSE with lamda - 100 Features and 100 Examples ')
# plt.legend()
# plt.show()
#
# #Labelling for desired output
# print("for 100 Features and 1000 Examples")
#
# # Calling the MSE function for train and test for Dataset2
# Train_1000_100=MSE(m1,m2,m1,m2)
# Test_1000_100=MSE(m1,m2,m3,m4)
#
# #Ploting the graph of MSE's for training and testing dataset by lambda
# plt.plot(Train_1000_100.keys(),Train_1000_100.values(),"r--",label='Train-1000-100',marker='o')
# plt.plot(Test_1000_100.keys(),Test_1000_100.values(),"b--",label='Test-1000-100',marker='o')
# plt.axhline(y=4.015, color='g', linestyle='-')
# plt.xlabel('lamda-regularization parameter')
# plt.ylabel('MSE')
# plt.title('Variation of MSE with lamda-100 Features and 1000 Examples')
# plt.legend()
# plt.show()
#
#
# #Labelling for desired output
# print("for wine dataset")
#
#
# # Calling the MSE function for train and test for Dataset3
# Train_wine=MSE(w1,w2,w1,w2)
# Test_wine=MSE(w1,w2,w3,w4)
#
# #Ploting the graph of MSE's for training and testing dataset by lambda
# plt.plot(Train_wine.keys(),Train_wine.values(),"r--",label='Train-winedataset',marker='o')
# plt.plot(Test_wine.keys(),Test_wine.values(),"b--",label='Test-winedataset',marker='o')
# plt.xlabel('lamda-regularization parameter')
# plt.ylabel('MSE')
# plt.title('Variation of MSE with lamda -winedataset ')
# plt.legend()
# plt.show()
# #
# #
# #Labelling for desired output
# print("for crime dataset")
#
# # Calling the MSE function for train and test for Dataset4
# Train_crime=MSE(c1,c2,c1,c2)
# Test_crime=MSE(c1,c2,c3,c4)
#
# #Ploting the graph of MSE's for training and testing dataset by lambda
# plt.plot(Train_crime.keys(),Train_crime.values(),"r--",label='Train-crimedataset',marker='o')
# plt.plot(Test_crime.keys(),Test_crime.values(),"b--",label='Test-crimedataset',marker='o')
# plt.xlabel('lamda-regularization parameter')
# plt.ylabel('MSE')
# plt.title('Variation of MSE with lamda-Crimedataset')
# plt.legend()
# plt.show()

#########################################################################################


#Task2:
#Merging the Datset3 Train and targetvalues arrays columnwise.
#The motive of this step is if you pick the training set randomly then the target value should be
#picked of the same row of the randomly picked training set.

modified_m2=np.transpose(m2)

# print(m1.shape)
# print(modified_m2.shape)

trainwithtarget=np.concatenate((m1, modified_m2[:,None]), axis=1)


# Defining a function, which randomly selects the rows of given attribute size.
def randomsample(size):
     return trainwithtarget[np.random.choice(trainwithtarget.shape[0], size, replace=False)]


## Defining the function which calculates array w for a given bias_p.
## Attributes are
# train: which is array of traindataset (in this scenario we would be passing the random sample of training dataset)
#tvalues: the target or labels array for passed training set
#bias_p : regularization parameter
# The output of the function is a dictionary with bias_p as key and w array as values.
def q2_finding_w(train,tvalues,bias_p):
    list1=[]
    list2=[]
    x=np.transpose(train).dot(tvalues)
    w=np.array((np.linalg.inv(bias_p*np.identity(train.shape[1])+np.transpose(train).dot(train))).dot(x))
    list1.append(bias_p)
    list2.append(w)
    w_dic=dict(zip(list1,list2))
    return w_dic


# Defining the MSE function which calculates the MSE for each lamba.
# The output of the function is a dictionary with lamba as key and MSE as values.

def q2_MSE(train,tvalues,test,testtvalues,bias_p):
    list1=[]
    list2=[]
    for k in q2_finding_w(train,tvalues,bias_p).keys():
        w=(q2_finding_w(train,tvalues,bias_p))
        sum=0
        N = len(testtvalues)
        for i in range(0,len(testtvalues)):
            MSE=np.square((np.transpose(test[i]).dot(w.get(k))-testtvalues[i]))
            sum+=MSE
        list1.append(k)
        list2.append(sum/N)
    MSE_dic=dict(zip(list1,list2))
    return(MSE_dic)


## This part calculates the mean of Mean square error of the samples generated from each training set of
#sample size ranging from 10 to 800,  as sample for each sample size is drawn 10 time, mean of these 10 MSE
# from each samples are calculated.
# And this excercise is done for each regularization parameter 7 (very low), 30 (Where MSE is low from the above Task1 excercise
# 125 which is too high

list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
for b in [7,30,125]:
    for i in range(10,801,40):
        list0 = []
        for k in range(0,10):
            MSE = q2_MSE(randomsample(i)[:, 0:100], randomsample(i)[:, 100], m3, m4,b)
            list0.append(list(MSE.values()))
        list0 = [item for sublist in list0 for item in sublist]

        mean=st.mean(list0)

        std=st.pstdev(list0)
        list1.append(i)
        list2.append(mean)
        list3.append(std)
    MSE_mean=dict(zip(list1,list2))
    MSE_std=dict(zip(list1,list3))
    list4.append(b)
    list5.append(MSE_mean)
    list6.append(MSE_std)
mean=dict(zip(list4,list5))
std=dict(zip(list4,list6))


print("Observations for Task 2")

#Ploting the graph of Mean of MSE as a function of size of training set for each lamba

df_plot=pd.DataFrame(mean)
print(df_plot)
df_plot.plot(kind='line',colormap='jet', marker='.', markersize=10,)
plt.xlabel('training size sample')
plt.ylabel('Mean of MSE')
plt.title('MSE for 3 different lamda for different training sizes')

plt.show()

##########################################################################################
#Task 3


#This function takes the training set and target array as the input
# The function basically finds the model parameter alpha and beta in a iterative way still these
#converges to a value.
def obtain_modelparameter(train,tvalues):
    list1=[]
    list2=[]
    for i in range(1,10):
        alpha0 = rd.randrange(1, 10)
        beta0 = rd.randrange(1, 10)
        flag = 1
        while(flag):
            #alpha0=rd.randrange(1,10)
            #beta0=rd.randrange(1,10)
            x=beta0*(np.transpose(train).dot(train))
            sn = np.linalg.inv(alpha0*(np.identity(train.shape[1]))+x)
            mn= np.array(beta0*(sn.dot(np.transpose(train)).dot(tvalues)))
            eigenvalues = np.linalg.eig(x)[0]
            gamma = 0
            for i in range(len(eigenvalues)):
                gamma += eigenvalues[i] / (alpha0 + eigenvalues[i])

            alpha=gamma/(np.transpose(mn).dot(mn))
            # print(alpha)
            sum = 0
            N = len(tvalues)
            for i in range(0, len(tvalues)):

                y = np.square(tvalues[i]-(np.transpose(mn).dot(train[i])))
                sum += y
            beta=(N-gamma)/float(sum)
            # print(beta)
            if(round(alpha0-alpha,1)==0 and round(beta0-beta,1)==0):
                flag = 0
            else:
                alpha0 = alpha
                beta0 = beta
        list1.append(alpha)
        list2.append(beta)
    return min(list1),min(list2)


#This function obtains the MSE using the MAP estimate for W
def MSE_trainset(train,tvalues,test,testtvalue):
    alpha=obtain_modelparameter(train,tvalues)[0]
    beta=obtain_modelparameter(train,tvalues)[1]
    x = beta * (np.transpose(test).dot(test))
    sn = np.linalg.inv(alpha * (np.identity(test.shape[1])) + x)
    mn = np.array(beta * (sn.dot(np.transpose(test)).dot(testtvalue)))
    sum = 0
    N = len(testtvalue)
    for i in range(0,N):
        MSE=np.square((np.transpose(test[i]).dot(mn)-testtvalue[i]))
        sum+=MSE
    return sum/N


#calculating the MSE using mn for all the 5 datasets

print("Observations for Task 3")
print("MSE using MAP estimate for Dataset-10 Features and 100 Example:",MSE_trainset(k1,k2,k3,k4))
print("MSE using MAP estimate for Dataset-100 Features and 100 Example:",MSE_trainset(l1,l2,l3,l4))
print("MSE using MAP estimate for Dataset-100 Features and 1000 Example:",MSE_trainset(m1,m2,m3,m4))
print("MSE using MAP estimate for Wine Dataset-11 Features and 342 Example:",MSE_trainset(w1,w2,w3,w4))
print("MSE using MAP estimate for Crime Dataset-100 Features and 298 Example:",MSE_trainset(c1,c2,c3,c4))

###################################################################################################

#Task 4:

#Storing the 4 array's of each dataset(train,targetvalues of train,test, targetvalues of test)

#f3
q1=dic_files.get('train-f3' )
q2=dic_files.get('trainR-f3')
q3=dic_files.get('test-f3')
q4=dic_files.get('testR-f3')

#f5
r1=dic_files.get('train-f5' )
r2=dic_files.get('trainR-f5')
r3=dic_files.get('test-f5')
r4=dic_files.get('testR-f5')


# This function modifies the data in the training and testing set to a mentioned degree (1,x,x^degree)
def create_example(data,degree):
    p = np.ones((len(data)))
    q = data.tolist()
    r = np.ones((len(data))).tolist()
    for i in range(degree):
        r = np.multiply(r, q)
        p = np.column_stack((p, r))
    return p

#This function takes the training set and target array as the input
# The function basically finds the model parameter alpha and beta in a iterative way still these
#converges to a value.
def q4_obtain_modelparameter(train,tvalues):
    list1 = []
    list2 = []
    rd.seed(11)
    for i in range(1, 3):
        alpha0 = rd.randrange(1, 10)
        beta0 = rd.randrange(1, 10)
        flag = 1
        while (flag):
            # alpha0=rd.randrange(1,10)
            # beta0=rd.randrange(1,10)
            x = beta0 * (np.transpose(train).dot(train))
            sn = np.linalg.inv(alpha0 * (np.identity(train.shape[1])) + x)
            mn = np.array(beta0 * (sn.dot(np.transpose(train)).dot(tvalues)))
            eigenvalues = np.linalg.eig(x)[0]
            gamma = 0
            for i in range(len(eigenvalues)):
                gamma += eigenvalues[i] / (alpha0 + eigenvalues[i])

            alpha = gamma / (np.transpose(mn).dot(mn))
            # print(alpha)
            sum = 0
            N = len(tvalues)
            for i in range(0, len(tvalues)):
                y = np.square(tvalues[i] - (np.transpose(mn).dot(train[i])))
                sum += y
            beta = (N - gamma) / float(sum)
            # print(beta)
            if (round(alpha0 - alpha, 1) == 0 and round(beta0 - beta, 1) == 0):
                flag = 0
            else:
                alpha0 = alpha
                beta0 = beta
        list1.append(alpha)
        list2.append(beta)
    return min(list1), min(list2)

#This function calculates the evidence for a given training set
#Attributes are train which is example matrix and tvalues is target values array
def evidence(train,tvalues):
    alpha = q4_obtain_modelparameter(train, tvalues)[0]
    beta = q4_obtain_modelparameter(train, tvalues)[1]
    # alpha=1
    # beta=2
    N=len(tvalues)
    M=train.shape[1]-1
    A=(alpha * (np.identity(train.shape[1]))) + (beta * (np.transpose(train).dot(train)))
    B=np.linalg.inv(A)
    mn=beta * (B.dot(np.transpose(train)).dot(tvalues))
    Expectance=0.5*(alpha*(np.transpose(mn).dot(mn)))+0.5*beta*(np.square(np.linalg.norm(tvalues-train.dot(mn))))
    Evid=0.5*M*(math.log(alpha))+0.5*N*(math.log(beta))-Expectance-0.5*(math.log(np.linalg.det(A)))-0.5*N*(math.log(2*(math.pi)))
    # print(Expectance)
    return Evid




# This function calculates the MSE using MAP predictive method for testing set passed
# for given training set
def q4_MSE_MAP(train,tvalues,test,testtvalue):
    alpha=q4_obtain_modelparameter(train,tvalues)[0]
    beta=q4_obtain_modelparameter(train,tvalues)[1]
    x = beta * (np.transpose(test).dot(test))
    sn = np.linalg.inv(alpha * (np.identity(test.shape[1])) + x)
    mn = np.array(beta * (sn.dot(np.transpose(test)).dot(testtvalue)))
    sum = 0
    N = len(testtvalue)
    for i in range(0,N):
        MSE=np.square((np.transpose(test[i]).dot(mn)-testtvalue[i]))
        sum+=MSE
    return sum/N

# This function calculates the MSE using non-regularized approach for testing set passed
# for a given training set

def q4_MSE_NR(train,tvalues,test,testtvalues):
    x = np.transpose(train).dot(tvalues)
    w = np.array((np.linalg.inv(np.transpose(train).dot(train))).dot(x))
    sum=0
    N = len(testtvalues)
    for i in range(0,N):
        MSE=np.square((np.transpose(test[i]).dot(w)-testtvalues[i]))
        sum+=MSE
    return (sum/N)



# This step calculated the 2 MSE and log of evidence for the dataset f3 for
#degree randing from 1 to 10
list1=[]
list2=[]
list3=[]
list4=[]
for i in range(1,11):
    MSE_MAP=q4_MSE_MAP(create_example(q1,i),q2,create_example(q3,i),q4)
    MSE_NR=q4_MSE_NR(create_example(q1,i),q2,create_example(q3,i),q4)
    log_evidence=evidence(create_example(q1,i),q2)
    list1.append(i)
    list2.append(MSE_MAP)
    list3.append(MSE_NR)
    list4.append(log_evidence)
f3_MSE_MAP=dict(zip(list1,list2))
f3_MSE_NR=dict(zip(list1,list3))
f3_logevidence=dict(zip(list1,list4))

print("Observations for Task 4")
print("MSE using MAP estimate as a function of degree:",f3_MSE_MAP)
print("MSE using non regularized approach as a function of degree:",f3_MSE_NR)
print("log of evidence as a function of degree:",f3_logevidence)

# Ploting the 2MSE's and log evidence as a function of degree
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(f3_MSE_MAP.keys(), f3_MSE_MAP.values(), 'r-',label='MSE-MAP',marker='o')
ax1.plot(f3_MSE_MAP.keys(),f3_MSE_NR.values(), 'g-',label='MSE-Non Regularized',marker='o')
ax2.plot(f3_MSE_MAP.keys(), f3_logevidence.values(), 'b-')

ax1.set_xlabel('Degree')
ax1.set_ylabel('MSE')
ax2.set_ylabel('Log Evidence', color='b')
plt.show()


# This step calculated the 2 MSE and log of evidence for the dataset f5 for
# degree randing from 1 to 10

list1=[]
list2=[]
list3=[]
list4=[]
for i in range(1, 11):
    MSE_MAP = q4_MSE_MAP(create_example(r1, i), r2, create_example(r3, i), r4)
    MSE_NR = q4_MSE_NR(create_example(r1, i), r2, create_example(r3, i), r4)
    log_evidence = evidence(create_example(r1, i), r2)
    list1.append(i)
    list2.append(MSE_MAP)
    list3.append(MSE_NR)
    list4.append(log_evidence)
f5_MSE_MAP = dict(zip(list1, list2))
f5_MSE_NR = dict(zip(list1, list3))
f5_logevidence = dict(zip(list1, list4))


print("MSE using MAP estimate as a function of degree:",f5_MSE_MAP)
print("MSE using non regularized approach as a function of degree:",f5_MSE_NR)
print("log of evidence as a function of degree:",f5_logevidence)
#
# # Ploting the 2MSE's and log evidence as a function of degree
#
#
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(f5_MSE_MAP.keys(), f5_MSE_MAP.values(), 'r-',label='MSE-MAP',marker='o')
ax1.plot(f5_MSE_MAP.keys(),f5_MSE_NR.values(), 'g-',label='MSE-Non Regularized',marker='o')
ax2.plot(f5_MSE_MAP.keys(), f5_logevidence.values(), 'b-')

ax1.set_xlabel('Degree')
ax1.set_ylabel('MSE')
ax2.set_ylabel('Log Evidence', color='b')
plt.show()















































