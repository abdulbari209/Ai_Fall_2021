REPORT
MULTINOMIAL NAIVE BAYES 
Because Multinomial Naive Bayes algorithm is a probabilistic learning method
LOGISTIC REGRESSION
Because Logistic regression is a statistical analysis method used to predict a data value based on prior observations of a data set
GAUSIAN NAIVE BAYES
It is a simple classification technique.They find use when the dimensionality of the inputs is high. Complex classification problems can also be implemented by using Naive Bayes Classifier.
PROBLEMS FACED
Problem in a sense huge amount of data thats why faced laptop performance issued 


import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#For unnormalized data
YTrain = trainData.Cover_Type;
trainData.drop('Cover_Type',inplace=True,axis=1)
X_train = trainData

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(testData)

print(trainData.shape)
print(testData.shape)

mnb = linear_model.Lasso(alpha=1)
mnb.fit(trainData,YTrain)
# Predictions
predictions = mnb.predict(testData)
print(predictions.shape)

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, YTrain)
Y_pred = mnb.predict(X_test)
acc_mnb = round(mnb.score(X_train, YTrain) * 100, 2)
print("Multinomial Naive Bayes accuracy =",round(acc_mnb,2,), "%")
print(Y_pred.shape)

submission = pd.DataFrame({
        "Id": testData["Id"],
        "Cover_Type": Y_pred
    })
submission.to_csv('Mul_submission.csv', index=False)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, YTrain)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, YTrain) * 100, 2)
print("Logistic Regression accuracy =",round(acc_log,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "Id": testData["Id"],
        "Cover_Type": Y_pred
    })
submission.to_csv('LR_submission.csv', index=False)
# Gaussian Naive Bayes
mnb = GaussianNB()
mnb.fit(X_train, YTrain)
Y_pred = mnb.predict(X_test)
acc_mnb = round(mnb.score(X_train, YTrain) * 100, 2)
print("Multinomial Naive Bayes accuracy =",round(acc_mnb,2,), "%")
print(Y_pred.shape)

submission = pd.DataFrame({
        "Id": testData["Id"],
        "Cover_Type": Y_pred
    })
submission.to_csv('Gau_submission.csv', index=False)
![image](https://user-images.githubusercontent.com/60986760/147825327-62afdf48-2d6d-4615-b3cc-e7bc09884103.png)




