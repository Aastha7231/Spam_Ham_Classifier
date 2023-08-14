import pandas as pd
import numpy as np
import os
import csv
import nltk
from nltk.corpus import words

p=os.getcwd()
p=p+'\\Solutions_CS22M005\\spam_ham_dataset.csv'
df = pd.read_csv(p,header=None)
nltk.download('words')
w=set(words.words())
df = np.array(df)
m = df.shape[0]
n = df.shape[1]
dictonary = []
print(m)

test_data=open('emails.csv','w+',newline='')
t=[]
path=os.getcwd()
path=path+'\\Solutions_CS22M005\\test'
os.chdir(path)
for file in os.listdir():
  if file.endswith(".txt"):
    file_path=f"{path}\{file}"
    l1=[]
    with open(file_path,'r') as f:
      l1.append(f.read())
      t.append(l1)  

print(len(t[0]))
print(len(t))
write=csv.writer(test_data)
write.writerows(t)



def make_dictonary():
  for i in range(m):
    mail=df[i][0].split()
    for j in mail:
      if j.lower() in w and j.lower() not in dictonary:
        dictonary.append(j)

make_dictonary()
print(len(dictonary))

def list_index(i):
  index=dictonary.index(i)
  return index

def initialize_x():
  x=[[0 for i in range(len(dictonary))] for i in range(m)]
  for i in range(m):
    mail=df[i][0].split()
    for j in mail:
      if j.lower() in dictonary:
        index=list_index(j.lower())
        x[i][index]=1
  return x

def initialize_y():
  y=[[1 for i in range(1)] for i in range(m)]
  for i in range(m):
    y[i]=df[i][1]
  return y

x=initialize_x()
print(x[5])

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

p_i=compute_p_y()

d= pd.read_csv('emails.csv',header=None)
d=np.array(d)
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