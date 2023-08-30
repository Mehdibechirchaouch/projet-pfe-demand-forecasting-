#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
vente_S30=pd.read_csv("S30_data.csv")


# In[2]:


vente_S30


# In[3]:


columns=['Code_Site','Date_comptable','Code_Article','Code_Fam_Stat_Article_1','Code_Fam_Stat_Article_2','Code_Fam_Stat_Article_3',
'Code_Categorie_Article',
'Affaire',
'Mt_Ligne_HT',
'Qté_facturée']


# In[4]:


vente_S30 = vente_S30.loc[:,columns]


# In[5]:


vente_S30


# In[6]:


vente_S30['Code_Fam_Stat_Article_1'] = pd.factorize(vente_S30['Code_Fam_Stat_Article_1'])[0]
vente_S30['Date_comptable'] = pd.to_datetime(vente_S30['Date_comptable'])
vente_S30['Code_Fam_Stat_Article_2'] = pd.factorize(vente_S30['Code_Fam_Stat_Article_2'])[0]
vente_S30['Code_Fam_Stat_Article_3'] = pd.factorize(vente_S30['Code_Fam_Stat_Article_3'])[0]


# In[7]:


# Groupement des données par catégorie d'article
groupes = vente_S30.groupby('Code_Categorie_Article')

# Itération sur les groupes et enregistrement dans des fichiers CSV
for categorie, groupe in groupes:
    nom_fichier = f"categorie_S30{categorie}.csv"  # Nom du fichier CSV
    groupe.to_csv(nom_fichier, index=False)


# In[8]:


SRVV=pd.read_csv("categorie_S30SRVV.csv")


# In[9]:


SRVV


# In[10]:


SRVV['Date_comptable'] = pd.to_datetime(SRVV['Date_comptable'])


# In[11]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = SRVV['Code_Site'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
SRVV['Code_Site'] = SRVV['Code_Site'].map(mapping)

# Vérifier les modifications
print(SRVV['Code_Site'])


# In[12]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = SRVV['Code_Article'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
SRVV['Code_Article'] = SRVV['Code_Article'].map(mapping)

# Vérifier les modifications
print(SRVV['Code_Article'])


# In[13]:


colonn='Code_Categorie_Article'
SRVV=SRVV.drop(colonn,axis=1)


# In[14]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = SRVV['Affaire'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
SRVV['Affaire'] = SRVV['Affaire'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[15]:


SRVV_1 = SRVV.sort_values('Date_comptable')


# In[16]:


SRVV_1.to_csv('SRVV_S30_Date.csv')


# In[17]:


srvv=pd.read_csv('SRVV_S30_Date.csv')


# In[18]:


srvv


# In[19]:


col='Unnamed: 0'


# In[20]:


srvv=srvv.drop(col,axis=1)


# In[21]:


srvv['Date_comptable'] = pd.to_datetime(srvv['Date_comptable'])
srvv.info()


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = srvv[features]
y = srvv['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle des forêts aléatoires
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
print('Quantité prédite:', y_pred)


# In[23]:


from sklearn.metrics import r2_score


# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Afficher le coefficient de détermination (R²)
print("Coefficient de détermination (R²) :", r2)


# In[24]:


from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à optimiser
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Créer le modèle de forêts aléatoires
model = RandomForestRegressor(random_state=42)

# Appliquer la recherche par grille avec validation croisée
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres et le meilleur score
print("Meilleurs hyperparamètres:", grid_search.best_params_)
print("Meilleur score de validation croisée:", grid_search.best_score_)


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = srvv[features]
y = srvv['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle des forêts aléatoires
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
print('Quantité prédite:', y_pred)


# In[26]:


from sklearn.metrics import r2_score


# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Afficher le coefficient de détermination (R²)
print("Coefficient de détermination (R²) :", r2)


# In[27]:


tolerance = 100  # Tolérance ou marge d'erreur

# Convertir les valeurs prédites et les valeurs réelles en classes binaires
y_pred_binary = abs(y_pred - y_test) <= tolerance

# Calculer l'accuracy
accuracy = sum(y_pred_binary) / len(y_pred_binary)

# Afficher l'accuracy
print("Accuracy:", accuracy)


# In[28]:


import numpy as np
import matplotlib.pyplot as plt


# Calculate the absolute difference between y_test and y_pred
diff = np.abs(y_test - y_pred)

# Create a colormap for the points based on the difference
cmap = plt.cm.get_cmap('cool')  # Choose a colormap, e.g., 'cool'

# Create a scatter plot with colored points based on the difference
plt.scatter(y_test, y_pred, c=diff, cmap=cmap)

# Add a colorbar to indicate the color scale
cbar = plt.colorbar()
cbar.set_label('Absolute Difference')

# Add labels and title
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Comparison of Actual and Predicted Values ")

# Show the plot
plt.show()


# In[29]:


autre methode pour la comparaision


# In[ ]:





# In[ ]:





# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

# Créer un DataFrame pour comparer les valeurs réelles et prédites
resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test, 'Valeurs_Prédites': y_pred})

# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
resultats_comparaison['Date_comptable'] = srvv.loc[X_test.index, 'Date_comptable']

# Assurez-vous que "Date_comptable" est au format de date
resultats_comparaison['Date_comptable'] = pd.to_datetime(resultats_comparaison['Date_comptable'])

# Trier les données par date pour le tracé
resultats_comparaison.sort_values(by='Date_comptable', inplace=True)

# Créer une figure et des axes pour le graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Tracer les valeurs réelles et prédites en fonction du temps
ax.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Réelles'], label='Valeurs Réelles', marker='o')
ax.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Prédites'], label='Valeurs Prédites', marker='x')

# Ajouter une légende, un titre et des labels d'axe
ax.legend()
ax.set_xlabel("Date comptable")
ax.set_ylabel("Quantité")
ax.set_title("Comparaison des valeurs réelles et prédites en fonction du temps")

# Ajouter une grille pour faciliter la lecture du graphique
ax.grid(True)


# Faire pivoter les étiquettes de l'axe des x pour une meilleure lisibilité
plt.xticks(rotation=45)

# Afficher le graphique
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




