from django.shortcuts import render
from django.http import HttpResponse
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
# Create your views here.

def predict(request):
	if request.method =="POST":
		S_data=pd.read_csv('https://raw.githubusercontent.com/vijaykumar10022/MahineLearning-With-Syllabus-wise/master/s_data.csv')
		x=S_data[['Height','Weight']]
		y=S_data['T-Shirt Size'].values.tolist()
		lableencoder_Y=LabelEncoder()
		Y=lableencoder_Y.fit_transform(y)
		X_train,X_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.3,random_state=3)
		knn=KNeighborsClassifier(n_neighbors=5)
		knn.fit(X_train,Y_train)
		y_pred=knn.predict(X_test)
		print(int(request.POST['height']),int(request.POST['weight']))
		data=knn.predict([[int(request.POST['height']),int(request.POST['weight'])]])
		accuracy=accuracy_score(Y_test,y_pred)*100
		print("--------------->",data,accuracy)
		data=str(data)
		return render(request,'T_shirt/result.html',{"data":data,"accuracy":accuracy})
	return render(request,'T_shirt/index.html')