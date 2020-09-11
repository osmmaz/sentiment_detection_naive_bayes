import sklearn
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import sklearn.preprocessing  as pre
from sklearn.metrics import accuracy_score

def mrg(X1,X2):
    xx=[]
    for i in range(len(X1)):
        xx.append([X1[i],X2[i]])
    return xx
def spl(X):
    G=[]
    B=[]
    for i in range(len(X)):
        if(X[i][1]==0):
            for w in X[i][0]:
                B.append(w)
        else:
            for w in X[i][0]:
                G.append(w)
    return G,B
def allwords(X):
    words = []
    for i in range(len(X)):
        for w in X[i][0]:
            words.append(w)
    return words

def cpt(cls):
    c=0
    for m in cls:
        c=c+cls.get(m)
    return c

df = pd.read_csv("texts.csv") # Replace text with the path to your texts file

X, y = df.iloc[:, 3].values, df.iloc[:, 1].values # Replace the 3 with the column of your text and 1 for the target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0) # You can change the parameters accoriding to your choice

for p in range(len(X)):
    X[p]=X[p].split()

Z=[]
Z=mrg(X,y)
Y=Counter(y)
G=[]
B=[]
G,B=spl(Z)
Words=allwords(Z)
words=Counter(Words)
good=Counter(G)
bad=Counter(B)

def pred(p,good,bad,words,cls):
    p_g=1
    for m in p:
        prob=good.get(m)
        if(prob==None):prob=0
        p_g=p_g*((prob+1)/(len(words)+cpt(good)))
    p_g=p_g*(cls.get(1)/(cls.get(1)+cls.get(0)))
    p_b=1
    for m in p:
        prob = bad.get(m)
        if (prob == None): prob = 0
        p_b = p_b * ((prob + 1) / (len(words) + cpt(bad)))
    p_b=p_b*(cls.get(0)/(cls.get(1)+cls.get(0)))
    if(p_g>p_b):return 1
    else:return 0


print("what is your text : ")
p=input()
p=p.split()
res=pred(p,good,bad,words,Y)
if res == 1: res_sent="good"
else: res_sent="bad"
print("your feeling is :",res_sent)


def predict(X_t):
    p=[]
    for i in X_t:
        p.append(pred(i,good,bad,words,Y))
    return p

y_pred=predict(X_test)
acc = accuracy_score(y_pred, y_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("acc:",acc)
