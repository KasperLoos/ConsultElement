import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint

#print(startup.isnull().sum())
#load data
url = 'https://raw.githubusercontent.com/KasperLoos/ConsultElement/refs/heads/main/data/bronze_data/startup_failures.csv'
startup = pd.read_csv(url)
startup['loan_approval'] = startup['status'].map({'operating':1, 'acquired' : 1, 'ipo' : 1, 'closed' : 0 })
startup['funding_total_usd'] = pd.to_numeric(startup['funding_total_usd'], errors='coerce')
startup['funding_total_usd'].fillna(0, inplace = True)

# Convert date columns
startup['founded_at'] = pd.to_datetime(startup['founded_at'], errors='coerce')
startup['first_funding_at'] = pd.to_datetime(startup['first_funding_at'], errors='coerce')
startup['last_funding_at'] = pd.to_datetime(startup['last_funding_at'], errors = 'coerce')

#drop missing values + weird dates -> less than 5% of data in this column
startup = startup.dropna(subset=['first_funding_at']).reset_index(drop=True)
start_date = '1970-01-01'
end_date = '2016-01-01'
startup = startup[startup['last_funding_at'].between(start_date, end_date)].reset_index(drop = True)

#convert to meaningfull columns
startup['founded_year'] = startup['founded_at'].dt.year
startup['founded_month'] = startup['founded_at'].dt.month
startup['funding_duration'] = (startup['last_funding_at'] - startup['first_funding_at']).dt.days
startup['funding_delay'] = (startup['first_funding_at'] - startup['founded_at']).dt.days
startup['mean_funding_per_round'] = startup['funding_total_usd'] / startup['funding_rounds']    

# Fill missing values with values of first funding date
startup['founded_year'] = startup['founded_year'].fillna(startup['first_funding_at'].dt.year)
startup['founded_month'] = startup['founded_month'].fillna(startup['first_funding_at'].dt.month)

#fill missing values with KNN imputer
imputer = KNNImputer(n_neighbors=10, weights='distance')
num_cols = startup.select_dtypes(include=['number']).columns
startup["funding_delay"] = imputer.fit_transform(startup[num_cols])[:, num_cols.get_loc("funding_delay")]

#make x and y variavbles
X_startup = startup.drop('loan_approval', axis='columns')
y_startup = startup['loan_approval']

#split data in training and test: 80 - 20
x_train, x_test, y_train,y_test = train_test_split(X_startup, y_startup, random_state=1, test_size = 0.2, train_size = 0.8)


CT_XGB = ColumnTransformer ([
    ('drop_columns', 'drop', ['status','permalink', 'homepage_url', 'category_list', 'state_code', 'region', 'city', 'name' ,'founded_at', 'first_funding_at', 'last_funding_at']),
    ('onehot', OneHotEncoder(handle_unknown='ignore'), [ 'country_code'])
])

#use CT on the train and test data
x_train_XGB = CT_XGB.fit_transform(x_train)
x_test_XGB = CT_XGB.transform(x_test)

#Use SMOTE procedure to correct for imbalanced data
smote = SMOTE()
x_train_XGB_resampled, y_train_XGB_resampled = smote.fit_resample(x_train_XGB, y_train)


#baseline XGB model
XGB = XGBClassifier(n_estimators=100, learning_rate=0.1,use_label_encoder=False, eval_metric='logloss')
XGB.fit(x_train_XGB_resampled, y_train_XGB_resampled)
preds2 = XGB.predict(x_test_XGB)
print("XGBoost Accuracy:", accuracy_score(y_test, preds2))
XGB_report = classification_report(y_test, preds2)

#parameter tuning for XGB model
param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 0.3),
    'lambda': uniform(0.5, 2)
}

random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_distributions=param_dist,
    n_iter=50,  # Try 50 different combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(x_train_XGB_resampled, y_train_XGB_resampled)

print("Best Parameters:", random_search.best_params_)

# Use best model
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(x_test_XGB)

print("Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred))
XGB_tuned_report = classification_report(y_test, y_pred)

#see results for 3 models
print(XGB_report, XGB_tuned_report)

