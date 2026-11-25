import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
 
from sklearn.metrics import accuracy_score 
  
  

data = pd.read_csv("data/student.csv")
print(data.columns.tolist())
  

features = [ 

    "average_grade", 
    "attendance_rate", 

    "absences", 

    "assignments_completed" 

] 

X = data[features] 

y = data["success "]
  

X_train, X_test, y_train, y_test = train_test_split( 

    X, y, test_size=0.3, random_state=42, stratify=y 

) 
  

model = RandomForestClassifier(random_state=42) 

model.fit(X_train, y_train) 
  

predictions = model.predict(X_test) 

print("Accuracy :", accuracy_score(y_test, predictions))