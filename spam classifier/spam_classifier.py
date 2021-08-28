# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:20:42 2021

@author: chitv
"""

import pandas as pd

sms= pd.read_csv(r"C:\Users\chitv\Desktop\kaggle\SpamClassifier-master\SpamClassifier-master\smsspamcollection\SMSSpamCollection",sep="\t", names=["label","message"])
sms.head()
#%%
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

corpus=[]
for i in range(0,len(sms)):
    review= re.sub("[^a-zA-Z]"," ", sms.message[i])
    review=review.lower()
    review= review.split()
    
    review=[ ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)
    #%%
    from sklearn.feature_extraction.text import CountVectorizer
    cv= CountVectorizer(max_features=5000)
    X= cv.fit_transform(corpus).toarray()
    
    y=pd.get_dummies(sms.label)
    y=y.iloc[:,1].values
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=1)
    #%%
    
    from sklearn.naive_bayes import MultinomialNB
    spam_model= MultinomialNB().fit(X_train,y_train)
    
    y_pred= spam_model.predict(X_test)
    #%%
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    cm= confusion_matrix( y_test, y_pred)
    print (cm)
    acc= accuracy_score(y_test,y_pred)
    print(acc)