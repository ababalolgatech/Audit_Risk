"""
@author: Dr.Ayodeji Babalola
"""

#                              IMPORT LIBS
#------------------------------------------------------------------
import pandas as pd
import numpy as np
import warnings  
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as  metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import seaborn as sb

plt.close('all')
plotting_flag = False


   
#                               Read the Data 
#------------------------------------------------------------------
data  = pd.read_csv('audit_risk.csv')


#                              Exploratory data Analysis
#------------------------------------------------------------------
data.head()
data.tail()

del_cols =['LOCATION_ID']  # not useful in the modeling process
data.drop(del_cols, axis=1, inplace=True)


print(data.isna().sum())  # checking for null values
data['Money_Value'].fillna((data['Money_Value'].mean()), inplace=True)  # test median

correlations_data = data.corr()['Risk'].sort_values()
df = data[correlations_data[18:-1].keys()]
if (plotting_flag == True):
    sb.pairplot(df, hue = 'Risk', diag_kind = 'hist',
                 plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                 size = 4)
corr = data.corr()
if (plotting_flag == True):
    sb.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,annot=True)

    X=data.drop(['Risk'],axis=1)

if (plotting_flag == True):
    plt.figure()
    plt.hist(data['CONTROL_RISK'])
    plt.xlabel('CONTROL_RISK)')
    plt.ylabel('Frequency)')
    plt.title('CONTROL_RISK')
    plt.figure()
    plt.hist(np.log(data['CONTROL_RISK']))
    plt.xlabel('CONTROL_RISK')
    plt.ylabel('Frequency)')
    plt.title('LOG of CONTROL_RISK')
    
    
data_skew = data.skew()

Y=data['Risk']

#           Train Test Split
#------------------------------------------------------------------
from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,stratify=Y, random_state = 50)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_scaled = pd.DataFrame(sc_X.fit_transform(X_train))
X_test_scaled = pd.DataFrame(sc_X.transform(X_test))


#                   Applying Base Model : Logistic Regression
#------------------------------------------------------------------
model_logistic = LogisticRegression();
model_logistic.fit(X_train_scaled, y_train)

#                       Cross Validation : Logistic Regression
#------------------------------------------------------------------
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'

acc_logi = cross_val_score(estimator = model_logistic, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)
acc_logi.mean()


#               Model Evaluation : Logistic Regression
#------------------------------------------------------------------
y_predict_logi = model_logistic.predict(X_test_scaled)
acc  = metrics.accuracy_score(y_test, y_predict_logi)
roc  = metrics.roc_auc_score(y_test, y_predict_logi)
prec = metrics.precision_score(y_test, y_predict_logi)
rec  = metrics.recall_score(y_test, y_predict_logi)
f1   = metrics.f1_score(y_test, y_predict_logi)

model_eval_logi_regress = pd.DataFrame([['Logistic Regression',acc, acc_logi.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])



#                       Applying Random Forest
#------------------------------------------------------------------
model_random_forest = RandomForestClassifier(n_estimators = 100, oob_score = True,criterion='entropy', random_state = 45)
model_random_forest.fit(X_train_scaled, y_train)  # train the model
print('Score: ', model_random_forest.score(X_train, y_train))


#               Model Evaluation : Random Forest
#------------------------------------------------------------------
acc_rande = cross_val_score(estimator = model_random_forest, X = X_train_scaled, y = y_train, cv = kfold, scoring=scoring)
acc_rande.mean()
y_predict_r = model_random_forest.predict(X_test_scaled)
roc  = metrics.roc_auc_score(y_test, y_predict_r)
acc  = metrics.accuracy_score(y_test, y_predict_r)
prec = metrics.precision_score(y_test, y_predict_r)
rec  = metrics.recall_score(y_test, y_predict_r)
f1   = metrics.f1_score(y_test, y_predict_r)

model_eval_random_forest = pd.DataFrame([['Random Forest',acc, acc_rande.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
model_evals = model_eval_logi_regress.append(model_eval_random_forest, ignore_index = True)


#                       CHOOSING THE BEST CLASSIFIER
#------------------------------------------------------------------

# Confusion Matrix
if (plotting_flag == True):
    plt.figure()
    cm_logi = metrics.confusion_matrix(y_test, y_predict_logi)
    plt.title('Confusion matrix (Logistic classifier)')
    sb.heatmap(cm_logi,annot=True,fmt="d")
    plt.show()
    
    plt.figure()
    cm_r = metrics.confusion_matrix(y_test, y_predict_r)
    plt.title('Confusion matrix (Random Forest classifier)')
    sb.heatmap(cm_r,annot=True,fmt="d")
    plt.show()


#                       ROC Curve  : 'Logistic Regression'
#------------------------------------------------------------------
model_random_forest.fit(X_train_scaled, y_train) # train the model
y_pred= model_logistic.predict(X_test_scaled) # predict the test data
# Compute False postive rate, and True positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test, model_logistic.predict_proba(X_test_scaled)[:,1])
# Calculate Area under the curve to display on the plot
auc = metrics.roc_auc_score(y_test,model_logistic.predict(X_test_scaled))
if (plotting_flag == True):
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('Logistic Regression', auc))
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

model_random_forest.fit(X_train_scaled, y_train) # train the model
y_pred= model_random_forest.predict(X_test_scaled) # predict the test data
# Compute False postive rate, and True positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test, model_random_forest.predict_proba(X_test_scaled)[:,1])
# Calculate Area under the curve to display on the plot
auc = metrics.roc_auc_score(y_test,model_random_forest.predict(X_test_scaled))
# Now, plot the computed values
if (plotting_flag == True):
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('Random Forest Entropy', auc))
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
