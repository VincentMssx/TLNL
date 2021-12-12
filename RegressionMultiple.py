import numpy as np
import pandas as pd
from featuresAnalyser import make_df
import matplotlib.pyplot as plt
import seaborn as sns

### Plot ###
def my_plot(df,x_value,y):
    y = y["las"].values.tolist()
    x = df[x_value].values.tolist()

    plt.figure(figsize=(7, 4))
    sns.set_style("whitegrid")
    ax = sns.lineplot(x=x, y=y, color='red')
    sns.despine(left=True)
    ax.axhline(0, color='grey')
    # ax.set(xlabel='Année', ylabel="Taux (%)")
    # ax.legend(['Arabie Saoudite','Moyen-Orient et Afrique du Nord'])
    plt.show()

#### Data ####

#df_train = make_df("train")
#df_train.sort_values(by=['uas'], inplace=True)
#df_test = make_df("dev")
#df_test.sort_values(by=['uas'], inplace=True)

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

df_train = pd.read_csv('results_train.csv',index_col='lang').sort_values(by=['uas'])
df_test = pd.read_csv('results_dev.csv',index_col='lang').sort_values(by=['uas'])

X_train = df_train[["taille_vocab" , "tailles_phrase" , "longueur_mots" , "taux_non_projectivite" , "nombre_moyen_dep" , "longueur_moyenne_dep" , "longueur_moyenne_max_dep", "Score"]]
X_train.fillna(X_train.mean(), inplace=True)
X_train = mean_norm(X_train)
y_train = df_train[["las"]]
X_test = df_test[["taille_vocab" , "tailles_phrase" , "longueur_mots" , "taux_non_projectivite" , "nombre_moyen_dep" , "longueur_moyenne_dep" , "longueur_moyenne_max_dep", "Score"]]
X_test.fillna(X_test.mean(), inplace=True)
X_test = mean_norm(X_test)
y_test = df_test[["las"]]

### Corrélations ###

corr_df = X_train.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.tight_layout()
plt.show()

### Regression ###

# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(X_train,y_train)

print(LR.intercept_)
print(LR.coef_)

y_prediction =  LR.predict(X_test)
y_prediction

# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)

print('r2 score is ', score)
print('MSE is ',mean_squared_error(y_test,y_prediction))
print('RMS error of is ',np.sqrt(mean_squared_error(y_test,y_prediction)))

for var in ["taille_vocab" , "tailles_phrase" , "longueur_mots" , "taux_non_projectivite" , "nombre_moyen_dep" , "longueur_moyenne_dep" , "longueur_moyenne_max_dep", "Score"]:
    my_plot(X_train,x_value=var,y=y_train)






