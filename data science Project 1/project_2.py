#  Please find the attachment for the project.
# # Try to create an instance in mongodb and store the data set and call it into python-
# 1. Problem Statement.: Business Understand
# 2. Data Cleaning
# 3. EDA -Exploratary data Ananlysis (give me stats insight)
# 4. Model selection
# 5. Evaluation
# 6. Deployment - Flask

# import Required Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve

# from ipywidgets import Output
# Problem Statement:
# 1. Clearly define the problem you are trying to solve using your dataset. This will guide your analysis and model development.
# Load and read The Data Set
df=pd.read_excel(r"C:\Users\DELL\Desktop\data science Project 1\Attrition Rate-Dataset.xlsx")
Output = pd.read_excel(r"C:\Users\DELL\Desktop\data science Project 1\Attrition Rate-Dataset.xlsx")
df

# 2. Data Cleaning:
# Clean the dataset by handling missing values, removing duplicates, correcting inconsistencies, and formatting data appropriately.
# Testing For Null Values

df.isnull().sum()

df.info()

df.value_counts()

# # Describe the Summary 
df.describe()

# # 3. drop the unwanted columns : "employee Name" and "Employee ID":
DataFrame = df.drop(["EmployeeName","EmployeeID"], axis = 1)
DataFrame

# #  converting categorical values into numerical values using Label Encoding : 
from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
data = DataFrame.apply(LabelEncoder.fit_transform)
data

# # 4.EDA - Exploratory Data Analysis:
# # Perform exploratory data analysis to gain insights into the dataset. This can involve statistical summaries, 
# # data visualization,(plotting) and identifying patterns or correlations within the data.
data.head()

data.describe()

# # Split the data into y value - (dependent variable or Target Variable) - dependent Variable(y = mx+c)
X = data.drop(['Attrition'],axis = 1)
y = data.Attrition

X.head()

# ## Variance and Standard deviation
data.Attrition.var()

# # Standard deviation :

data.Attrition.std()

# ## Skewness 

data.Attrition.skew()

# # kurtossis:

data.Attrition.kurt()



# #  Third moment : bussiness understanding
# ## Univariant plot
plt.figure(figsize=(8,6))
plt.hist(data.Designation)
plt.xlabel("designation")
plt.ylabel("Traininghours")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(data.PercentSalaryHike)
plt.xlabel("designation")
plt.ylabel("PercentSalaryHike")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(data.TraininginHours)
plt.xlabel("designation")
plt.ylabel("TraininginHours")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(data.MonthlySalary)
plt.xlabel("designation")
plt.ylabel("MonthlySalary")
plt.show()

plt.hist(data.PercentSalaryHike)
plt.show()

plt.boxplot(data.Tenure)
plt.show()

plt.boxplot(data.Designation)
plt.show()

plt.boxplot(data.Attrition)
plt.show()

plt.boxplot(data.PercentSalaryHike)
plt.show()

plt.boxplot(data.TraininginHours)
plt.show()

## Data Destribution :
from scipy import stats
from matplotlib import pylab, mlab, pyplot
stats.probplot(data.Designation, plot=pylab)

stats.probplot(data.PercentSalaryHike, dist="norm",plot=pylab)

stats.probplot(data.TraininginHours, dist="norm",plot=pylab)

stats.probplot(data.TraininginHours, dist="norm",plot=pylab)

## Heatmap for the attrition data
sns.heatmap(data.corr(), annot = True, fmt = '.0%')

## Splitting the data into two types train data  and test data.
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf.fit(train_x,train_y)
predictions=clf.predict(test_x)
print("Accuracy of the model using Decision Tree Clasifier on test Data:")
print(accuracy_score(test_y,predictions))
predictions = clf.predict(train_x)
print("Accuracy of the model using Decision Tree Clasifier on train Data:")
print(accuracy_score(train_y,predictions))

from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier()

y1 = np.mean(cross_val_score(clf,test_x, test_y, cv=10))
y2 = np.mean(cross_val_score(clf,train_x, train_y, cv=10))

print("Trian Data - RF with CV ", y2*100)
print("Test Data - RF with CV", y1*100)

import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(train_x,train_y)
predictions = model.predict(test_x)
print("Accuaracy of the model using XG Classifier on test data:")
print(accuracy_score(test_y,predictions))

from sklearn.model_selection import cross_val_score

clf = xgb.XGBClassifier()

y1 = np.mean(cross_val_score(clf,test_x, test_y, cv=10))
y2 = np.mean(cross_val_score(clf,train_x, train_y, cv=10))

print("Trian Data - RF with CV ", y2*100)
print("Test Data - RF with CV", y1*100)

## Building the model using RandomForestClassifier :
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")
## model fitting on the train data
rf.fit(train_x, train_y)

## Predicting on the test data
pred_test_rf = rf.predict(test_x)

## fpr, tpr and theshold values
# from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
fpr_test_rf, tpr_test_rf, thresholds_test_rf = roc_curve(test_y, pred_test_rf)

## Accuracy for test data
accuracy_test_RF = np.mean(pred_test_rf == test_y)
accuracy_test_RF

pred_train_rf = rf.predict(train_x)
## Accuracy for train data
accuracy_train_RF = np.mean(pred_train_rf == train_y)
accuracy_train_RF

# fpr, tpr and threshold values Index:
import pylab as pl
i = np.arange(len(tpr_test_rf))
roc_test_rf = pd.DataFrame({'fpr_test_rf' : pd.Series(fpr_test_rf, index=i),'tpr_test_rf' : pd.Series(tpr_test_rf, index = i), '1-fpr_test_rf' : pd.Series(1-fpr_test_rf, index = i), 'tf_test_rf' : pd.Series(tpr_test_rf - (1-fpr_test_rf), index = i), 'thresholds_test_rf' : pd.Series(thresholds_test_rf, index = i)})
roc_test_rf.iloc[(roc_test_rf['tf_test_rf']-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc_test_rf['tpr_test_rf'], color = 'red')
# pl.plot(roc_test_rf['1-fpr_test_rf'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])
## Areaunder curve
roc_auc = auc(fpr_test_rf, tpr_test_rf)
print("Area under the ROC curve : %f" % roc_auc)

## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test_rf = confusion_matrix(pred_test_rf, test_y)
cm_test_rf

## Sencitivity(True positive rate)
Sensitivity = cm_test_rf[0,0]/(cm_test_rf[0,0] + cm_test_rf[0,1])
print('sensitivity:', Sensitivity)

# ## Specificity(True nagative)
specitivity = cm_test_rf[1,1]/(cm_test_rf[1,0] + cm_test_rf[1,1])
print('Specificity:', specitivity)

## Model predict on train data
pred_train_rf = rf.predict(train_x)

from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier()

y1 = np.mean(cross_val_score(clf,test_x, test_y, cv=10))
y2 = np.mean(cross_val_score(clf,train_x, train_y, cv=10))

print("Trian Data - RF with CV ", y2*100)
print("Test Data - RF with CV", y1*100)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf.fit(train_x,train_y)
# making prediction with our model 
predictions = clf.predict(train_x)



# ## For Building model obeject ( UI ) #############

import pickle
train_x['pred'] = y2
test_x['pred'] = y1
newTable =  pd.concat([train_x,test_x],axis=0)        
df4 = pd.merge(newTable, Output[['EmployeeID','EmployeeName']], left_index=True, right_index=True)
with open('finalModel_randForest.pkl', 'wb') as f:
    pickle.dump(df4, f)