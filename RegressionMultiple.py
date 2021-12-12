import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import chi2, SelectKBest, f_regression
from sklearn.linear_model import Ridge, MultiTaskLasso, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from featuresAnalyser import make_df


####### Data #######

#df_train = make_df("train")
df_train = pd.read_csv('results.csv',index_col='lang').sort_values(by=['uas'])
#df_test = make_df("test")
X_train = df_train[["taille_vocab","tailles_phrase","longueur_mots","taux_non_projectivite","nombre_moyen_dep","longueur_moyenne_dep","longueur_moyenne_max_dep"]]
y_train = df_train[["uas"]]
#X_test = df_test["taille_vocab","tailles_phrase","longueur_mots","taux_non_projectivite","nombre_moyen_dep","longueur_moyenne_dep","longueur_moyenne_max_dep"]
#y_test = df_test["uas"]

# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(X_train,y_train)



####### Preprocessing #######

# 1. Nan ? Normalement non

# 2. Suppression des outiers
# model = IsolationForest(contamination=0.02)
# model.fit(X_train)
# outliers = model.predict(X_train)
# Appliquer la liste outliers à X_train

# 3. Standardisation
#scalerStd = StandardScaler() #Utiliser RobustScaler s'il y a des outliers
#X_train_std = scalerStd.fit_transform(X_train)

# 4. Test de dépendance :
# X_train.var(axis=0)
# chi2(X_train,y_train) #Renvoie quelles features sont les plus importantes
# f_regression(X_train,y_train)


####### Models #######

# 1. Test de plusieurs model
# models = {Ridge(), MultiTaskLasso(), Lasso()}
# scores = []

# for model in models:
#    model.fit(X_train,y_train)
#    score = model.score(X_test, y_test)
#    scores.append(score)

# 2. GridSearch
# param = {} # Pamètres à remplir
#grid = GridSearchCV(MeilleursModels(),param_grid,cv=StratifiedKFold(5))
#grid.fit(X_train,y_train)
#print(grid.best_params_)
#print(grid.best_score_)
