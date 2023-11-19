import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score,confusion_matrix
from sklearn.metrics import  recall_score, f1_score, accuracy_score
from sklearn.model_selection import  cross_val_score, cross_val_predict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE

path='C:/Users/name/'

data_trn= pd.read_csv(path+"glitch_trn_data.csv")
print('Before removing duplicates,shape:',data_trn.shape)

#Dropping Duplicates
data_trn=data_trn.drop_duplicates()
print('After removing duplicates,shape:',data_trn.shape)
#Number of unique values for every feature
def Num_uniques(df):
    
    output_data = []

    for col in df.columns:
            # Append the column name, number of unique values and the data type to the output data
            output_data.append([col ,df.loc[:,col].isnull().sum(), 
                                df.loc[:, col].nunique(),
                                  df.loc[:, col].dtype])
    output_df = pd.DataFrame(output_data, columns=['Column Name','Null Values',
                                                    'Number of Unique Values',
                                                     'Data Type'])

    return output_df
Num_uniques(data_trn)
data_trn.describe().T
label_trn=pd.read_csv(path+"glitch_trn_class_labels.csv", header=None)
#print(label_trn.head())
label_count= label_trn[1].value_counts()
print(label_count)
print('\n\nNumber of target classes:',len(label_count))
y_drop_train=label_trn.drop(columns=[0])
#print(y_drop_train.head())
data_tot=data_trn.copy()
data_tot['Target']=y_drop_train
data_tot=data_tot.drop(['id','GPStime','ifo'],axis=1)
# Pair plot with color-coded classes
sns.pairplot(data_tot, hue='Target', markers=['o'], palette='husl')
plt.show()
#Catagorical to numarical
catagorical=['ifo']
x_en_train= data_trn
x_en_test= pd.read_csv(path+'glitch_tst_data.csv')
list_data=[x_en_train,x_en_test]
mylist=[]

for data in list_data:
    for x in catagorical:
        x = pd.get_dummies(data, columns=[f'{x}'], prefix=[f'{x}'])
    x[x==True]=1
    x[x==False]=0
    mylist.append(x)

[x_en_train,x_en_test]=mylist

#print(x_en_train.head())
'''We can ignore the id and GPStime of the gravitational wave dataset 
because they do not directly contribute to glitch classification'''

x_drop_train=x_en_train.drop(columns=['id','GPStime'])
x_drop_test=x_en_test.drop(columns=['id','GPStime'])
#print(nwlist)
print(x_drop_train.head())


#Defining Class weights
unique_classes=np.unique(y_drop_train)
class_weights = compute_class_weight('balanced', classes=unique_classes,
                                      y=np.array(y_drop_train).reshape(-1))

### Dictionary for the class weight of each classes
dict_weight=dict(zip(unique_classes,class_weights))

weight_info=pd.DataFrame(label_count)
weight_info['weights']=weight_info.index.map(dict_weight)
print(weight_info)
#These would be used in every model

# Specifying which features to standardize and which ones to keep unchanged
features_to_standardize = ['centralFreq','peakFreq','snr','bandwidth','duration']

# Create a ColumnTransformer for feature transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('std_scaler', StandardScaler(), features_to_standardize)
    ],
    remainder='drop'
)
# Define scoring functions for cross-validation
scoring = {
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'f1_macro': make_scorer(f1_score, average='macro')
}
#Defining evaluation function
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)

class eval : 
    def __init__(self,X, y,model,model_name):
        self.X=X
        self.y=y
        self.precisions = cross_val_score(model, X, y, cv=skf, scoring=make_scorer(precision_score, average='macro'), n_jobs=-1, error_score='raise')
        self.recalls = cross_val_score(model, X, y, cv=skf, scoring=make_scorer(recall_score, average='macro'), n_jobs=-1, error_score='raise')
        self.f1s = cross_val_score(model, X, y, cv=skf, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1, error_score='raise')
        self.model_name=model_name
        self.accuracy=cross_val_score(model, X, y, cv=skf, scoring=make_scorer(accuracy_score), n_jobs=-1, error_score='raise')
        self.model=model

    def print_eval(self):
        print(f' For {self.model_name}')
        print("Accuracy Score(CV):",np.mean(self.accuracy))
        print("Macro-Averaged Precision (CV):", np.mean(self.precisions))
        print("Macro-Averaged Recall (CV):", np.mean(self.recalls))
        print("Macro-Averaged F1-Score (CV):", np.mean(self.f1s))

    def confusion(self):
        self.predicted = cross_val_predict(self.model, self.X, self.y, cv=skf)  # Replace X and y with your features and labels
        self.cm = confusion_matrix(self.y, self.predicted, normalize='true')
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.cm, annot=True, fmt="0.2f", cmap="Blues",
            xticklabels=unique_classes, yticklabels=unique_classes,cbar=False)
        plt.title(f'Normalized Confusion Matrix for {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
# Creating a pipeline with a preprocessor and Logistic Regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Standardizing numerical features
    ('classifier', LogisticRegression(solver= 'lbfgs',random_state=25))  # Logistic Regression
])


param_grid = {
    'classifier__C': [0.001, 0.01, 1, 10],
    'classifier__penalty': [None,'l1', 'l2'],
    'classifier__max_iter':[500,1000],
    'classifier__class_weight':[dict_weight,None]
}

cv_logistic = GridSearchCV(pipeline, param_grid,
                                cv=5, scoring=scoring,
                                refit='f1_macro',n_jobs=-1,
                                )
cv_logistic.fit(x_drop_train, y_drop_train)


best_logistic=cv_logistic.best_estimator_
logistic=eval(x_drop_train,y_drop_train,best_logistic,
               'Logistic Regression')
print(best_logistic)
logistic.print_eval()
logistic.confusion()

# Define a pipeline that includes data preprocessing (e.g., StandardScaler) and the classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Data preprocessing (optional)
    ('rf', RandomForestClassifier(random_state=25))
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'rf__class_weight':[dict_weight,None],
    'rf__n_estimators': [ 200, 300],
    'rf__max_depth': [20, 30, None],
    'rf__criterion' : ["gini", "entropy"],
    'rf__min_samples_split': [3, 5, 8,],
    'rf__max_features': ['sqrt', 'log2',None]
    }


# Set up GridSearchCV for hyperparameter tuning
model_rf = GridSearchCV(pipeline, param_grid, cv=5,
                                   scoring=scoring, 
                                   refit='f1_macro',
                                   error_score='raise',verbose=3,
                                   n_jobs=-1)

# Fit the RandomizedSearchCV to the training data
model_rf.fit(x_drop_train, y_drop_train)

# Extract the best estimator from the RandomizedSearchCV
best_rf = model_rf.best_estimator_
print(best_rf)
#Evaluation Score
rand_f_eval=eval(x_drop_train,y_drop_train,best_rf,
               'Random Forest')
rand_f_eval.print_eval()
rand_f_eval.confusion()
# Create a pipeline with StandardScaler and SVC
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('svc', SVC(random_state=25))
])

# Define hyperparameter grid for RandomizedSearchCV
param_dist = {
    'svc__class_weight':[None,dict_weight],
    'svc__probability':[True,False],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

# Perform RandomizedSearchCV with cross-validation
model_svc = GridSearchCV(
    pipeline,    
    param_grid=param_dist,
    scoring=scoring,
    cv=5,  # Number of cross-validation folds
    refit='f1_macro',
    error_score='raise',
    n_jobs=-1 , # Use all available CPU cores
    verbose=5
)

# Fit the random search on the training data
model_svc.fit(x_drop_train, y_drop_train)

best_svc=model_svc.best_estimator_
print(best_svc)
svc=eval(x_drop_train,y_drop_train,best_svc,
               'Support Vector Classifier')

svc.print_eval()
svc.confusion()
# Create a pipeline with a scaler and KNN classifier
pipeline_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier())
])

# Define the hyperparameter grid for KNN
param_knn = {
    'knn__n_neighbors': np.arange(1, 11),  # Number of neighbors
    'knn__weights': ['uniform', 'distance'],  # Weighting of neighbors
    'knn__p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Use GridSearchCV to search for the best hyperparameters
model_knn = GridSearchCV(pipeline_knn, param_grid=param_knn,
                          cv=5, scoring='f1_macro', n_jobs=-1,
                          error_score='raise',verbose=4)

# Fit the model
model_knn.fit(x_drop_train, y_drop_train)

# Get the best parameters
best_knn = model_knn.best_estimator_

# Print the best parameters
print("Best Hyperparameters:", best_knn)
knn=eval(x_drop_train,y_drop_train,best_knn,
               'K Nearest Neighbour Classifier')
knn.print_eval()
knn.confusion()
pipeline_dt = Pipeline([
    ('preprocessor', preprocessor),
    ('decision_tree', DecisionTreeClassifier(random_state=25))
])
# Define the hyperparameter grid for Decision Tree
param_dt = {
    'decision_tree__criterion': ['gini', 'entropy'],  # Criterion for the Decision Tree
    'decision_tree__max_depth': np.arange(1, 10),  # Maximum depth of the tree
    'decision_tree__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'decision_tree__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'decision_tree__class_weight':[dict_weight,None],
    'decision_tree__max_features':['auto', 'sqrt', 'log2',None]
}

# Use GridSearchCV to search for the best hyperparameters
model_dt = GridSearchCV(pipeline_dt, param_grid=param_dt,
                         cv=5, scoring='f1_macro',verbose=4)

# Fit the model
model_dt.fit(x_drop_train, y_drop_train)
# Get the best parameters
best_dt = model_dt.best_estimator_

# Print the best parameters
print("Best Hyperparameters:", best_dt)
dt=eval(x_drop_train,y_drop_train,best_dt,
               'Decision Tree Classifier')
dt.print_eval()
dt.confusion()
be_dt=DecisionTreeClassifier(criterion='entropy', max_depth=9,
                                        min_samples_split=10, random_state=25)

#AdaBoost for Random Forest
pipeline_ad=Pipeline([
    ('preprocessor', preprocessor),
    ('boost', AdaBoostClassifier(random_state=25))
    ])

param_ad = {
            'boost__base_estimator':[be_dt],
            'boost__n_estimators':[50, 100, 200],
            'boost__learning_rate': [0.01, 0.1, 1.0]
            }

# Use GridSearchCV for AdaBoosting
grid_ad = GridSearchCV(pipeline_ad, param_grid=param_ad,
                               cv=5, scoring='f1_macro',n_jobs=-1,
                               error_score='raise')
grid_ad.fit(x_drop_train, y_drop_train)


best_ad=grid_ad.best_estimator_
print(best_ad)
ad=eval(x_drop_train,y_drop_train,best_ad,'Adaboost')
ad.print_eval()
ad.confusion()
# All the models with best hyperparameter

be_log=LogisticRegression(C=0.001,class_weight=dict_weight,
                                    max_iter=1000, penalty=None,random_state=25)

be_svc=SVC(C=10,class_weight=dict_weight,
                     kernel='linear', probability=True, random_state=25)

be_rf=RandomForestClassifier(class_weight=dict_weight,
                                        max_depth=20, max_features='log2',
                                        min_samples_split=8,
                                        n_estimators=200,random_state=25)

be_dt=DecisionTreeClassifier(criterion='entropy', max_depth=9,
                                        min_samples_split=10, random_state=25)

be_knn=KNeighborsClassifier(n_neighbors=4, p=1, weights='distance')
be_ad=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                          max_depth=9,
                                                                          min_samples_split=10),
                                    n_estimators=200)
# Initialize RFE
rfe = RFE(estimator=be_rf)

# Define the parameter grid for grid search
param_fet = {'n_features_to_select': np.arange(1, x_drop_train.shape[1] + 1)}

# Initialize GridSearchCV
grid_fet = GridSearchCV(estimator=rfe, param_grid=param_fet, scoring='accuracy', cv=5)

# Fit GridSearchCV to the training data
grid_fet.fit(x_drop_train, y_drop_train)

be_ft=grid_fet.best_estimator_
print(be_ft)
eval_ft=eval(x_drop_train,y_drop_train,be_ft,'Random Forest of Selected Features')
eval_ft.print_eval()
# Access the selected features from the best_estimator_
selected_features = x_drop_train.columns[grid_fet.best_estimator_.support_]
ig=[]
for i in list(x_drop_train.columns):
    if i not in list(selected_features):
        ig.append(i)
print(f'Ignored feature(s):{ig}')
eval_ft.confusion()
be_rf.fit(x_drop_train,y_drop_train)
pred_rf=be_rf.predict(x_drop_test)
be_ft.fit(x_drop_train,y_drop_train)
pred_rf_ft=be_ft.predict(x_drop_test)
pd.DataFrame(pred_rf_ft).to_csv('test_labels_21051.csv',index=False,header=False)
pd.DataFrame(pred_rf_ft).value_counts()
