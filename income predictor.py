import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import csv
#LOading the Training Data
zip1= pd.read_csv('C:/Users/CHANDY/Desktop/ML Project/tcd ml 2019-20 income prediction training (with labels).csv')
#Filling the empty values with fillna 
zip1=zip1.fillna(method='ffill')
print(len(zip1))
#LOading the data that needs to be predicted 
zip2= pd.read_csv('C:/Users/CHANDY/Desktop/ML Project/tcd ml 2019-20 income prediction test (without labels).csv')
zip2=zip2.fillna(method='ffill')
print(len(zip2))
#Assigning the values from 89594 to the end for validation 
validation_zip=zip1.iloc[89594:,:]
training_zip=zip1.iloc[:89594,:]
predict_zip=zip2
#22399
#111993
#pad 73230


#COnnecting all the data sets into a single variables before one hat encoding
df_row_reindex = pd.concat([training_zip, validation_zip,predict_zip])

print(df_row_reindex.head())

X = df_row_reindex[['Country','Age','Profession','University Degree','Gender','Year of Record','Hair Color','Wears Glasses','Body Height [cm]']]


#Performing label encoding
Y = df_row_reindex[['Income in EUR']]
print(Y)
le=LabelEncoder()
X['Country']=le.fit_transform(X['Country'])
X['Profession']=le.fit_transform(X['Profession'])
X['University Degree']=le.fit_transform(X['University Degree'])
X['Gender']=le.fit_transform(X['Gender'])
X['Hair Color']=le.fit_transform(X['Hair Color'])
#performing one hat encoding
ohe=OneHotEncoder(categorical_features=[0])
X=ohe.fit_transform(X).toarray()
training_data=X[:111993,:]
print(len(training_data))
training_data_o=Y.iloc[:111993,:]
print(training_data_o)
from sklearn.ensemble import RandomForestRegressor
#Usinf the RandomForestRegressor on the traning data
model=rf = RandomForestRegressor()

result=model.fit(training_data,training_data_o)

#validation_data=X[89594:111993,:]
#validation_data_o=Y.iloc[89594:111993,:]
#print(model.score(validation_data,validation_data_o))
predict_data=X[111993:,:]
#Using the predict function to predict
lis1=model.predict(predict_data)
print(lis1)
#Writing the files into CSV
with open('new.txt','w') as f:
    csv_writer=csv.writer(f,delimiter='\n')
    csv_writer.writerow(lis1)

