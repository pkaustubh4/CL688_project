import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pickle

# loading data
data = pd.read_csv("Concrete_Data.csv", delimiter=",", skiprows=0, nrows=1127)

# converting data types
types = {
    'Type_of_Coarse_Aggregate': str,
    'Type_of_Fine_Aggregate': str,
    'Max_Size_of_Coarse_Aggregate(mm)': int,
    'Passing_0.6_mm_sieve(%)': float,
    'Target_Mean_Strength(MPa)': float,
    'Cement_O.P.C.(kg/m3)': int,
    'W/C_Ratio': float,                     
    'Water_Content(kg/m3)': int,           
    'Total_Aggregate(kg/m3)': int,         
    'Fine_Aggregate(kg/m3)': int,           
    'Coarse_Aggregate(kg/m3)': int,        
    'Workability_Slump(mm)': int,          
    'Hardened_Concrete_Density_(avg.)': float,
    '7_day_str(MPa)': float,                 
    '28_day_str(MPa)': float,               
    'Admix_1': str,                        
    'Admix_2': str,                         
    'Dos_1(lit)': float,                    
    'Dos_2(lit)': float
}

data = data.astype(types)

# choosing features for classification
features = ['Passing_0.6_mm_sieve(%)', 'Cement_O.P.C.(kg/m3)', 'W/C_Ratio', 
            'Fine_Aggregate(kg/m3)', 'Coarse_Aggregate(kg/m3)', 'Workability_Slump(mm)', 
            'Hardened_Concrete_Density_(avg.)', 'Dos_1(lit)', 'Dos_2(lit)']

train_target = '7_day_str(MPa)'

# splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data[features], 
    (data[train_target] > data[train_target].mean()).astype(int),
    test_size=0.2, 
    random_state=42
)

# scaling data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# building model
model = DecisionTreeClassifier(random_state=42)

# training model
model.fit(x_train, y_train)

# making prediction
y_pred = model.predict(x_test)

# evaluating model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# plotting roc curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()

# plotting feature importance
feature_importance = model.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, data[features].columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.tight_layout()
plt.show()

# saving model
pickle.dump(model, open('class.pkl', 'wb'))