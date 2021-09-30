# Start by emporting all libriaries used to analyze, wrangle & split data. Create, optimize, hypertune and select models

!pip3 install category_encoders==2.
!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
!pip install eli5
!pip install shap
import pandas_profiling
import pandas as pd
import numpy as np
import seaborn as sns
import category_encoders as ce
import plotly.express as px
import eli5
import shap
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from eli5.sklearn import PermutationImportance
from xgboost import XGBClassifier



# Exploratory data analysis: 

df = pd.read_csv('Desktop/Datasets/churn_dataset_train.csv')
# df_test = pd.read_csv('/content/drive/MyDrive/churn_dataset_test.csv')

df.head()
df.info()
df.describe().T
df.describe(exclude=np.number).T

missing = df.isnull().sum()
missing.sort_values()



## BASELINE Statisitic then LogisitcRegression then shallow decision tree!
 
# creating a simple base line/ logisitic regression should be normal baseline after EDA is finsihed!
df['churn'].value_counts(normalize=True)

target = 'churn'
y_train = df[target]
y_train.value_counts(normalize=True)


# Training accuracy of majority class baseline = 
# frequency of majority class (aka base rate)
# since its a classification im using the mode
majority_class = y_train.mode()[0]
y_pred = [majority_class] * len(y_train)
accuracy_score(y_train, y_pred)

### Overview of correlation 1

pd.crosstab(df['number_customer_service_calls'], df['churn']).plot(kind='bar');

pd.crosstab(df['churn'], df['international_plan'])

### Overview of correlation 2
profile_report = df.profile_report()

profile_report


# Machine Learning:

# we only have 0.7 person of data mission and its missing from two columns that could be imputed easily.
# There is couple ways to approach this and since the data set is fairly small, randomized cv for validation would be more appropriate, 
# I will still use a 3 way split to allow me to have a test set kept in the vault for my last model test and will use val to tune and hyperparamter.
# I will use val also to run my precision and recall test and only use test to adjust recall in this situation 
# because of the type of problem we want to detect churn before it happens but we don't want to be over spending in marketing or some other channel
# By focusing on recall we could get close to perfection in predicting churn but we'll loose on signficantly in precision
# Balance will be key. 



# Spliting data 3ways: 

train, val = train_test_split(df, train_size=0.70, test_size=0.30, random_state=42)
train, test = train_test_split(train, train_size=0.80, test_size=0.20, random_state=42)

train.shape, val.shape, test.shape

# Creating my feature matrix and y vector (target and covareates)

target = 'churn'
features = train.columns.drop([target])
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

# MODEL 1 Randon Forest Classifer:
# (for reference I talked about a logistic regression in my pdf file but I only included its coefficient since it scored very low)
# I'm using a pipeline to encode, transform, reguleraize (for logistic regression only), fit and predict.

# Increase "minimun node size"=min_samples_leaf, to reduce model complexity
pipeline = make_pipeline(
    ce.OrdinalEncoder(),   #fit_tranfrom
    SimpleImputer(strategy='median'),  #fit_tranform 
    RandomForestClassifier(max_features=0.54, n_estimators=470, max_depth=10,  n_jobs=-1, random_state=42)  #fit.(ensemble ~150 trees)
)

pipeline.fit(X_train, y_train)

print('Train Accuracy', pipeline.score(X_train, y_train))
print('Validation Accuracy', pipeline.score(X_val, y_val))
# y_pred = pipeline.predict(X_val)
print('Test Accuracy', accuracy_score(y_test, y_test))


# Train Accuracy 0.9775510204081632
# Validation Accuracy 0.9485714285714286
# Test Accuracy 0.9571428571428572


# Quick confusion Matrix:

# used encoded columns because encoder changes dimensiality (used one hotencoder in a previous model)
y_pred = pipeline.predict(X_val)
confusion_matrix(y_val, y_pred)

# array([[890,   9],
#        [ 45, 106]])

# Quick confusion Matrix for y_test:
# array([[424,   7],
#        [ 14,  45]])


# feature Importances in a data farme graphed horizantaly

import matplotlib.pyplot as plt
# Get feature importances
rf = pipeline.named_steps['randomforestclassifier']
importances = pd.Series(rf.feature_importances_, X_val.columns)

# Plot top n feature importances
n = 25
plt.figure(figsize=(10,n/2))
plt.title(f'Top {n} features')
importances.sort_values()[-n:].plot.barh(color='grey');



# chekcing if labels are correct


from sklearn.utils.multiclass import unique_labels
unique_labels(y_val)

def plot_confusion_matrix(y_true, y_pred):
  labels = unique_labels(y_true)
  columns = [f'Predicted {label}' for label in labels]
  index = [f'Actual {label}' for label in labels]
  return columns, index

plot_confusion_matrix(y_val, y_pred)



# making a pandas data frame

def plot_confusion_matrix(y_true, y_pred):
  labels = unique_labels(y_true)
  columns = [f'Predicted {label}' for label in labels]
  index = [f'Actual {label}' for label in labels]
  table = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=columns, index=index)
  return table

plot_confusion_matrix(y_val, y_pred)



# Predicted no	Predicted yes
# Actual no	890	9
# Actual yes	45	106

#  plot a heatmap showcasing a more readable version of cm


def plot_confusion_matrix(y_true, y_pred):
  labels = unique_labels(y_true)
  columns = [f'Predicted {label}' for label in labels]
  index = [f'Actual {label}' for label in labels]
  table = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=columns, index=index)
  return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')

plot_confusion_matrix(y_val, y_pred);



#  Classification report

print(classification_report(y_val, y_pred))



#              precision    recall  f1-score   support

#           no       0.95      0.99      0.97       899
#          yes       0.92      0.70      0.80       151

#     accuracy                           0.95      1050
#    macro avg       0.94      0.85      0.88      1050
# weighted avg       0.95      0.95      0.95      1050


threshold = 0.5
pipeline.predict_proba(X_val)[:, 1] >threshold


y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
sns.distplot(y_pred_proba);


threshold = 0.6

y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
y_pred = y_pred_proba > threshold

ax = sns.distplot(y_pred_proba)
ax.axvline(threshold, color='red')

pd.Series(y_pred).value_counts()


roc_auc_score(y_val, y_pred_proba)


pipeline.predict_proba(X_val)


# PERMUATION IMPORTANCES AS A PRCOSEEING STEP; shuffling features's relation to the target 

# Using eli5 for permuation testing:

transformers = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='median')
)


X_train_transformed = transformers.fit_transform(X_train)
X_val_transformed = transformers.transform(X_val)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_transformed, y_train)


### 1. Calculate permutation importances
permuter = PermutationImportance(
    model, 
    scoring='accuracy', 
    n_iter=5, 
    random_state=42
)

permuter.fit(X_val_transformed, y_val)



feature_names = X_val.columns.tolist()
pd.Series(permuter.feature_importances_, feature_names).sort_values()


### 2. Display permutation importances

# area_code                       -0.001143
# total_day_calls                 -0.000952
# total_eve_calls                  0.000381
# account_length                   0.000571
# state                            0.001524
# total_night_calls                0.002095
# number_vmail_messages            0.002286
# total_night_minutes              0.002857
# total_night_charge               0.003810
# total_intl_minutes               0.006667
# total_eve_minutes                0.010857
# total_eve_charge                 0.011619
# total_intl_charge                0.016000
# voice_mail_plan                  0.019810
# total_day_charge                 0.020190
# total_intl_calls                 0.021714
# number_customer_service_calls    0.037714
# international_plan               0.048952
# total_day_minutes                0.054286
# dtype: float64



# Weight	Feature
# 0.0543 ± 0.0050	total_day_minutes
# 0.0490 ± 0.0057	international_plan
# 0.0377 ± 0.0076	number_customer_service_calls
# 0.0217 ± 0.0052	total_intl_calls
# 0.0202 ± 0.0070	total_day_charge
# 0.0198 ± 0.0022	voice_mail_plan
# 0.0160 ± 0.0030	total_intl_charge
# 0.0116 ± 0.0044	total_eve_charge
# 0.0109 ± 0.0023	total_eve_minutes
# 0.0067 ± 0.0017	total_intl_minutes
# 0.0038 ± 0.0034	total_night_charge
# 0.0029 ± 0.0040	total_night_minutes
# 0.0023 ± 0.0059	number_vmail_messages
# 0.0021 ± 0.0028	total_night_calls
# 0.0015 ± 0.0009	state
# 0.0006 ± 0.0015	account_length
# 0.0004 ± 0.0019	total_eve_calls
# -0.0010 ± 0.0032	total_day_calls
# -0.0011 ± 0.0008	area_code




# MODEL2


from xgboost import XGBClassifier


pipeline = make_pipeline(
    ce.OrdinalEncoder(), 
    XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)

pipeline.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = pipeline.predict(X_val)
print('Validation Accuracy', accuracy_score(y_val, y_pred))


# Validation Accuracy 0.9466666666666667


### fit_transfom on train, transform on val
encoder = ce.OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)

model = XGBClassifier(scale_pos_weight=0.7, #helped with the inbalanced classes
    n_estimators=1000,  # <= 1000 trees, depends on early stopping
    max_depth=5,        # try deeper trees because of one high cardinality categoricals(st)
    learning_rate=0.8,  # try higher learning rate
    n_jobs=-1
)

eval_set = [(X_train_encoded, y_train), 
            (X_val_encoded, y_val)]

model.fit( X_train_encoded, y_train, 
          eval_set=eval_set,
          eval_metric='error', 
          early_stopping_rounds=50) # Stop if the score hasn't improved in 50 rounds


# Stopping. Best iteration:
# [4]	validation_0-error:0.029082	validation_1-error:0.039048



results = model.evals_result()
train_error = results['validation_0']['error']
val_error = results['validation_1']['error']
epoch = range(1, len(train_error)+1)
plt.plot(epoch, train_error, label='Train')
plt.plot(epoch, val_error, label='Validation')
plt.ylabel('Classification Error')
plt.xlabel('Model Complexity (n_estimators)')
# plt.xlim(())
plt.ylim((-0.04, 0.08)) # Zoom in
plt.legend();

y_pred = model.predict(X_val_encoded)
print(classification_report(y_val, y_pred))



#               precision    recall  f1-score   support

#           no       0.96      0.99      0.98       899
#          yes       0.96      0.76      0.85       151

#     accuracy                           0.96      1050
#    macro avg       0.96      0.88      0.91      1050
# weighted avg       0.96      0.96      0.96      1050


from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(model, ax=ax)



# 3 Testing on test and creating csv file for submission: 

# Exporting the model for testing then production

from joblib import dump
dump(pipeline, 'pipeline.joblib', compress=True)


# Reloading model:

from joblib import load
joblibed = load('/content/pipeline.joblib')

churn_dataset_test = pd.read_csv('/content/drive/MyDrive/churn_dataset_test.csv')
churn_dataset_test.head()
churn_dataset_test.shape


# Removing the extra column to align with input size:
churn_dataset_test = churn_dataset_test.drop('churn', axis=1)
churn_dataset_test = churn_dataset_test.drop('index', axis=1)

# Checking for missing values even it does not matter since we have an automated inputer for new data coming in, but since
# model was trained with .7% of missing values.. for consistency purposes i wanted to keep same ratio by checking:

churn_dataset_test.isnull().sum()

# Fitting model into unseen test data provided
joblibed.predict(churn_dataset_test)

# Creating a data frame from predictions and Stitching columns back together

final_predictions = joblibed.predict(churn_dataset_test)
final_predictions = pd.DataFrame(final_predictions)

#  Quickly checking for how many yes!
final_predictions.value_counts(normalize=True)

churn_dataset_test['churn'] = final_predictions
churn_dataset_test.head(10)

# Exporting csv file with filled predictions
from google.colab import files
churn_dataset_test.to_csv('churn_dataset_test.csv') 
files.download('churn_dataset_test.csv')

# Retesting for safety:)

retesting = pd.read_csv('/content/churn_dataset_test.csv')
retesting.head(10)


# Using Shap for a deeper analysis of feature selection..
# I want it to be a dataframe thats why the double brakcet..
churn_dataset_test.iloc[[0]]

row = churn_dataset_test.iloc[[0]]

y_test.iloc[[0]]

joblibed.predict(row)

# it predicted NO but why, now lets crack open the black box:

rf = pipeline.named_steps['randomforestclassifier']
encoded_row = encoder.transform(row)
explainer = shap.TreeExplainer(rf)
explainer.shap_values(encoded_row)

shap_values = explainer.shap_values(encoded_row)

shap.initjs()
shap.force_plot(base_value = explainer.expected_value[0], 
                shap_values = shap_values[0],
                features = row,
                link = 'logit') #for classification this shows pred proba


#  Thank you for you time reading this! 