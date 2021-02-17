#import Library

import pandas as pd
import numpy as np
import seaborn as sns




df=pd.read_csv('emails.csv')  #dataset Import dateset={emails.csv}





df  #dataset check 





df['spam'].value_counts() # Count spam value.





df.drop_duplicates(inplace =True) # duplicate value drop





df  #again check the datset





df['spam'].value_counts()   # again count spam value into dataset 





df.isnull().sum() # null value check for missing mail.





x=df.text.values  # x means text of mail into dataset; x actualy independent value





y=df.spam.values # y means spam value  into dataset; y actualy dependent value





from sklearn.model_selection import train_test_split # import dataset split library from sklearn





xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)  # dataset split because xtrain data and ytrain data do the separate.Caz by train data dataset train and by test data complete the dataset model. 





from sklearn.feature_extraction.text import CountVectorizer # import CountVectorizer from sklearn.





cv=CountVectorizer()                     # object cteate for CountVectorizer
x_train=cv.fit_transform(xtrain)





x_train.toarray()  # find preporcessor dataset array.when x_train preprocessor.





from sklearn.naive_bayes import MultinomialNB # import machine learning algorithm Naive Bayes





model=MultinomialNB()
model.fit(x_train,ytrain)      # set the xtrain and ytrain.





x_test = cv.transform(xtest) 


x_test.toarray() #x_test check where (xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2))[size 0.2 means 20% data check


:


model.score(x_test,ytest) # model Accuracy check..





emails =['Congratulations for joining authlab','Hay you win iphone'] #emails=['Congratulations for joining authlab' (part 1),'Hay you win iphone'(part2)] .





cv_emails = cv.transform(emails)  #CountVectorizer do the preprocess mail  





model.predict(cv_emails)  # And then predict emails ham or spam.





