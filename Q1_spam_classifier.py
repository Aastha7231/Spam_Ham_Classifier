import pandas as pd
import numpy as np
import os
import csv
import nltk
from nltk.corpus import words

# reading the training data named as spam_ham_dataset
df = pd.read_csv('spam_ham_dataset.csv',header=None)
# using the nltk dictionary to create my own dictionary
nltk.download('words')
w=set(words.words())
df = np.array(df)
m = df.shape[0]
n = df.shape[1]

# dictonary is created using training data and nltk words
dictonary = []

# take the path of current working directory and store it in a csv file
path1=os.getcwd()
test_data="emails.csv"
t=[]
# accessing the fiven folder named test and reading the input email txt files to create test data
path=os.getcwd()
path=path+'\\test'
os.chdir(path)
# reading the test emails
for file in os.listdir():
  if file.endswith(".txt"):
    file_path=f"{path}\{file}"
    l1=[]
    with open(file_path,'r') as f:
      l1.append(f.read())
      t.append(l1)  

# emails.csv contains all the emails stored as rows in a csv file
os.chdir(path1)
with open(test_data,'w') as csvfile:
  write=csv.writer(csvfile)
  for i in range(len(t)):
    write.writerow(t[i])

# function to create dictionary for the training dataset
def make_dictonary():
  for i in range(m):
    mail=df[i][0].split()
    for j in mail:
      if j.lower() in w and j.lower() not in dictonary:
        dictonary.append(j)

# calling the make_dictonary function to create a dictonary
make_dictonary()

# function to return the index of the dictionary element
def list_index(i):
  index=dictonary.index(i)
  return index

# function to create the dataset for the training data
def initialize_x():
  x=[[0 for i in range(len(dictonary))] for i in range(m)]
  for i in range(m):
    mail=df[i][0].split()
    for j in mail:
      if j.lower() in dictonary:
        index=list_index(j.lower())
        x[i][index]=1
  return x

# function to store the value of y for the training data
def initialize_y():
  y=[[1 for i in range(1)] for i in range(m)]
  for i in range(m):
    y[i]=df[i][1]
  return y
# function call create the features dataset
x=initialize_x()

# function to compute the p_hat using the total number of spam emails in the training dataset
def compute_p():
  count=0
  for i in range(m):
    if y[i]:
      count+=1
  return count


y=initialize_y()
count_spam=compute_p()
count_nonspam=m-count_spam
spam=count_spam/m
print(spam)
non_spam=1-spam
print(np.log(spam/non_spam))

# function to create features matrix corresponding to each each element in dictionary for spam and nonspam category
def compute_p_y():
  p_i=[[1 for i in range(2)] for i in range(len(dictonary))]
  for i in range(len(dictonary)):
    count_0=0
    count_1=0
    for j in range(m):
      if y[j]==0:
        if x[j][i]==1:
          count_0+=1
      if y[j]==1:
        if x[j][i]==1:
          count_1+=1
    p_i[i][0]=count_0/count_nonspam
    p_i[i][1]=count_1/count_spam
  return p_i

# function call to compute the p matrix for training data
p_i=compute_p_y()

# reading the test data csv file created above
d=pd.read_csv('emails.csv',header=None)
d=np.array(d)

# function to initialize the test dataset for each word in dictionary
def initialize_x_test():
  m_test=d.shape[0]
  x_test=[[0 for i in range(len(dictonary))] for i in range(m_test)]
  for i in range(m_test):
    mail=d[i][0].split()
    for j in mail:
      if j.lower() in dictonary:
        index=list_index(j.lower())
        x_test[i][index]=1
  return x_test

x_test=initialize_x_test()

# function to compute y_predicted labels for test dataset
def compute_y_pred():
  m_test=d.shape[0]
  y_pred=[[0 for i in range(1)] for i in range(m_test)]
  for i in range(m_test):
    val=0
    for j in range(len(dictonary)):
      A=0
      B=0
      C=0
      if p_i[j][1]!=0 and p_i[j][0]!=0 and x_test[i][j]!=0:
        A=x_test[i][j]*np.log(p_i[j][1]/p_i[j][0])
      if (p_i[j][1])!=1 and (p_i[j][0])!=1 and x_test[i][j]!=1:
        B=(1-x_test[i][j])*np.log((1-p_i[j][1])/(1-p_i[j][0]))
      C=np.log(spam/non_spam)
      val+=A+B
    # print(val)
    if val>=0:
      y_pred[i]=1
    else:
      y_pred[i]=0
  return y_pred

y_pred=compute_y_pred()
# printing the final predicted labels for test dataset
print(y_pred)