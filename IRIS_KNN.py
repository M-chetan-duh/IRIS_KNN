import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("/content/datasets_17860_23404_IRIS.csv")
data.head()
data["sepal_length"]
data.shape
from sklearn.model_selection import train_test_split
x = data.iloc[:,[0,1,2,3]]
y = data.iloc[:,4]
x
y
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)
x_train.shape
y_train.shape
model = KNN(n_neighbors=5,metric='euclidean')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
y_test
accuracy = accuracy_score(y_test,y_pred)*100
print("The accuracy without scaling is",accuracy)
confusion_matrix(y_test,y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
scalar = MinMaxScaler()
x_train1= scalar.fit_transform(x_train)
x_test1= scalar.transform(x_test)
model.fit(x_train1,y_train)
new_ypred = model.predict(x_test1)
accuracy_after_scaling= accuracy_score(y_test,new_ypred)*100
print("The accuarcy after scaling is",accuracy_after_scaling)
no_neighbors= np.arange(1,8)
print(no_neighbors)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))
for i, k in enumerate(no_neighbors):
    
    knn = KNN(n_neighbors=k)
  
    knn.fit(x_train,y_train)
    
    
    train_accuracy[i] = knn.score(x_train, y_train)

    
    test_accuracy[i] = knn.score(x_test, y_test)

plt.plot(no_neighbors,train_accuracy,label = "training accuracy")
plt.plot(no_neighbors,test_accuracy,label  = "testing accuracy")
plt.title("KNN accuracy plot")
plt.legend()
plt.xlabel("number of neighbours")
plt.ylabel("accuracy")
plt.show()
