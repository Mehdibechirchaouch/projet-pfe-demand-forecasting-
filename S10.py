#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
vente_S10=pd.read_csv("S10_data.csv")


# In[2]:


vente_S10


# In[3]:


columns=['Code_Site','Date_comptable','Code_Article','Code_Fam_Stat_Article_1','Code_Fam_Stat_Article_2','Code_Fam_Stat_Article_3',
'Code_Categorie_Article',
'Affaire',
'Mt_Ligne_HT',
'Qté_facturée']


# In[4]:


vente_S10 = vente_S10.loc[:,columns]


# In[5]:


vente_S10


# In[6]:


vente_S10['Code_Fam_Stat_Article_1'] = pd.factorize(vente_S10['Code_Fam_Stat_Article_1'])[0]
vente_S10['Date_comptable'] = pd.to_datetime(vente_S10['Date_comptable'])
vente_S10['Code_Fam_Stat_Article_2'] = pd.factorize(vente_S10['Code_Fam_Stat_Article_2'])[0]
vente_S10['Code_Fam_Stat_Article_3'] = pd.factorize(vente_S10['Code_Fam_Stat_Article_3'])[0]


# In[7]:


# Groupement des données par catégorie d'article
groupes = vente_S10.groupby('Code_Categorie_Article')

# Itération sur les groupes et enregistrement dans des fichiers CSV
for categorie, groupe in groupes:
    nom_fichier = f"categorie_S10{categorie}.csv"  # Nom du fichier CSV
    groupe.to_csv(nom_fichier, index=False)


# In[8]:


Palim=pd.read_csv("categorie_S10PALIM.csv")


# In[9]:


Palim


# In[10]:


Palim['Date_comptable'] = pd.to_datetime(Palim['Date_comptable'])


# In[46]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Code_Site'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Code_Site'] = Palim['Code_Site'].map(mapping)

# Vérifier les modifications
print(Palim['Code_Site'])
valeurs_uniques


# In[12]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Code_Article'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Code_Article'] = Palim['Code_Article'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[13]:


colonn='Code_Categorie_Article'
Palim=Palim.drop(colonn,axis=1)


# In[14]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Affaire'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Affaire'] = Palim['Affaire'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[15]:


Palim_1 = Palim.sort_values('Date_comptable')


# In[16]:


Palim_1.to_csv('Palim_S10_Date.csv')


# In[17]:


palim=pd.read_csv('Palim_S10_Date.csv')


# In[18]:


palim


# In[19]:


col='Unnamed: 0'


# In[20]:


palim=palim.drop(col,axis=1)


# In[21]:


palim['Date_comptable'] = pd.to_datetime(palim['Date_comptable'])
palim.info()


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = palim[features]
y = palim['Qté_facturée']

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
X = palim[features]
y = palim['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle des forêts aléatoires
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
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


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')

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


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

# Créer un DataFrame pour comparer les valeurs réelles et prédites
resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test, 'Valeurs_Prédites': y_pred})

# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
resultats_comparaison['Date_comptable'] = palim.loc[X_test.index, 'Date_comptable']

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


# In[31]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = palim[features]
y = palim['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train1, y_train1)

# Faire des prédictions sur l'ensemble de test
y_pred1 = model.predict(X_test1)

# Calculer le RMSE
rmse = mean_squared_error(y_test1, y_pred1, squared=False)
print('RMSE:', rmse)


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = palim[features]
y = palim['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir les hyperparamètres à tester
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.05, 0.01]
}

# Créer l'estimateur du modèle
model = GradientBoostingRegressor(random_state=42)

# Utiliser GridSearchCV pour rechercher les meilleurs hyperparamètres en utilisant la validation croisée
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres et le meilleur score RMSE
print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleur score RMSE :", (-grid_search.best_score_) ** 0.5)

# Entraîner le modèle avec les meilleurs hyperparamètres
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculer le RMSE pour le meilleur modèle
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE (Meilleur modèle) :', rmse)


# In[33]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = palim[features]
y = palim['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=150, max_depth=7, random_state=42)
model.fit(X_train1, y_train1)

# Faire des prédictions sur l'ensemble de test
y_pred1 = model.predict(X_test1)

# Calculer le RMSE
rmse = mean_squared_error(y_test1, y_pred1, squared=False)
print('RMSE:', rmse)


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

# Créer un DataFrame pour comparer les valeurs réelles et prédites
resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test1, 'Valeurs_Prédites': y_pred1})

# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
resultats_comparaison['Date_comptable'] = palim.loc[X_test1.index, 'Date_comptable']

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


# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error

# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = palim[features]
y = palim['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=150, max_depth=7, random_state=42)
gb_model.fit(X_train, y_train)

# Créer et entraîner le modèle RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test avec les deux modèles
y_pred_gb = gb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Calculer les métriques d'évaluation pour chaque modèle
rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

r2_gb = r2_score(y_test, y_pred_gb)
r2_rf = r2_score(y_test, y_pred_rf)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
# Afficher les résultats
print("Résultats du modèle Gradient Boosting :")
print("RMSE :", rmse_gb)
print("R² :", r2_gb)
print("MAE :", mae_gb)

print("\nRésultats du modèle Random Forest :")
print("RMSE :", rmse_rf)
print("R² :", r2_rf)
print("MAE :", mae_rf)


# In[36]:


import matplotlib.pyplot as plt

# Faire un graphique de dispersion pour le modèle Gradient Boosting
plt.scatter(y_test, y_pred_gb, label='Prédiction GB')

# Faire un graphique de dispersion pour le modèle Random Forest
plt.scatter(y_test, y_pred_rf, label='Prédiction RF')

# Tracer la ligne y=x (ligne de référence)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray')

# Ajouter des labels et un titre
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédictions")

# Ajouter une légende
plt.legend()

# Afficher le graphique
plt.show()


# In[37]:


# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
X_test_with_date = X_test.copy()
X_test_with_date['Date_comptable'] = palim.loc[X_test.index, 'Date_comptable']

# Assurez-vous que "Date_comptable" est au format de date
X_test_with_date['Date_comptable'] = pd.to_datetime(X_test_with_date['Date_comptable'])

# Trier les données par date pour le tracé
X_test_with_date.sort_values(by='Date_comptable', inplace=True)

# Créer une figure et des axes pour le graphique
fig, ax = plt.subplots(figsize=(12, 6))

# Tracer les prédictions du modèle Gradient Boosting
ax.plot(X_test_with_date['Date_comptable'], y_pred_gb, label='Prédictions Gradient Boosting', marker='o')

# Tracer les prédictions du modèle Random Forest
ax.plot(X_test_with_date['Date_comptable'], y_pred_rf, label='Prédictions Random Forest', marker='x')

# Tracer les valeurs réelles
ax.plot(X_test_with_date['Date_comptable'], y_test, label='Valeurs Réelles', marker='s')

# Ajouter une légende, un titre et des labels d'axe
ax.legend()
ax.set_xlabel("Date comptable")
ax.set_ylabel("Quantité")
ax.set_title("Comparaison des prédictions des modèles en fonction du temps")

# Ajouter une grille pour faciliter la lecture du graphique
ax.grid(True)

# Faire pivoter les étiquettes de l'axe des x pour une meilleure lisibilité
plt.xticks(rotation=45)

# Afficher le graphique
plt.tight_layout()
plt.show()


# In[38]:


# Tracer les résidus pour les deux modèles
residuals_gb = y_test - y_pred_gb
residuals_rf = y_test - y_pred_rf

plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals_gb, label="Gradient Boosting", alpha=0.7)
plt.scatter(y_test, residuals_rf, label="Random Forest", alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Valeurs Réelles")
plt.ylabel("Résidus")
plt.title("Comparaison des résidus pour les deux modèles")
plt.legend()
plt.grid(True)
plt.show()


# In[39]:


# Calculer les résidus pour chaque modèle
residuals_gb = y_test - y_pred_gb
residuals_rf = y_test - y_pred_rf

# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
X_test_with_date = X_test.copy()
X_test_with_date['Date_comptable'] = palim.loc[X_test.index, 'Date_comptable']

# Tracer les résidus en fonction de la date comptable pour chaque modèle
plt.figure(figsize=(10, 6))
plt.scatter(X_test_with_date['Date_comptable'], residuals_gb, label="Gradient Boosting", alpha=0.7)
plt.scatter(X_test_with_date['Date_comptable'], residuals_rf, label="Random Forest", alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Date comptable")
plt.ylabel("Résidus")
plt.title("Analyse des résidus en fonction de la date comptable")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[40]:


# Créer un nouveau DataFrame contenant toutes les informations
results_df = pd.DataFrame({
    'Valeurs_Réelles': y_test,
    'Date_comptable': X_test_with_date['Date_comptable'],
    'Valeurs_Prédites_RF': y_pred_rf,
    'Valeurs_Prédites_GB': y_pred_gb,
    'Résidus_RF': residuals_rf,
    'Résidus_GB': residuals_gb,
    'RMSE_RF': rmse_rf,
    'RMSE_GB': rmse_gb,
    'R²_RF': r2_rf,
    'R²_GB': r2_gb,
    'MAE_RF': mae_rf,
    'MAE_GB': mae_gb,
    'Code_Article': X_test['Code_Article'],
    'Code_Site': X_test['Code_Site']
})

# Afficher le DataFrame résultant
print(results_df)


# In[41]:


results_df.to_csv('S10_RF_GB.csv')


# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix

# Préparation des données
features = ['Code_Article', 'Code_Site', 'Date_comptable', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT']
X = palim[features]
y = palim['Qté_facturée']

# Convertir la colonne "Date_comptable" en caractéristiques numériques
X['Date_comptable'] = pd.to_datetime(X['Date_comptable'])
X['Année'] = X['Date_comptable'].dt.year
X['Mois'] = X['Date_comptable'].dt.month
X['Jour'] = X['Date_comptable'].dt.day
X = X.drop('Date_comptable', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle GradientBoostingRegressor
model_gb = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
model_gb.fit(X_train, y_train)

# Créer et entraîner le modèle RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model_rf.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test avec les deux modèles
y_pred_gb = model_gb.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Calculer les métriques d'évaluation pour chaque modèle
rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

r2_gb = r2_score(y_test, y_pred_gb)
r2_rf = r2_score(y_test, y_pred_rf)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Afficher les résultats
print("Résultats du modèle Gradient Boosting :")
print("RMSE :", rmse_gb)
print("R² :", r2_gb)
print("MAE :", mae_gb)

print("\nRésultats du modèle Random Forest :")
print("RMSE :", rmse_rf)
print("R² :", r2_rf)
print("MAE :", mae_rf)


# In[43]:


from sklearn.metrics import confusion_matrix

# Faire des prédictions sur l'ensemble de test pour les deux modèles
y_pred_gb = model_gb.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Convertir les prédictions en valeurs binaires (0 si la quantité est nulle, 1 sinon)
y_pred_bin_gb = (y_pred_gb > 0).astype(int)
y_pred_bin_rf = (y_pred_rf > 0).astype(int)

# Convertir les vraies étiquettes en valeurs binaires
y_test_bin = (y_test > 0).astype(int)

# Calculer la matrice de confusion pour les deux modèles
cm_gb = confusion_matrix(y_test_bin, y_pred_bin_gb)
cm_rf = confusion_matrix(y_test_bin, y_pred_bin_rf)

print("Matrice de confusion pour Gradient Boosting:")
print(cm_gb)

print("\nMatrice de confusion pour Random Forest:")
print(cm_rf)


# In[44]:


# Calculate True Positives, True Negatives, False Positives, False Negatives
tp_gb = 4024
tn_gb = 0
fp_gb = 0
fn_gb = 173

# Calculate Precision, Recall, F1 Score, and Accuracy
precision_gb = tp_gb / (tp_gb + fp_gb)
recall_gb = tp_gb / (tp_gb + fn_gb)
f1_gb = 2 * (precision_gb * recall_gb) / (precision_gb + recall_gb)
accuracy_gb = (tp_gb + tn_gb) / (tp_gb + tn_gb + fp_gb + fn_gb)

print("F1 Score for Gradient Boosting:", f1_gb)
print("Accuracy for Gradient Boosting:", accuracy_gb)


# In[45]:


# Calculate True Positives, True Negatives, False Positives, False Negatives
tp_rf = 4197
tn_rf = 0
fp_rf = 0
fn_rf = 0

# Calculate Precision, Recall, F1 Score, and Accuracy
precision_rf = tp_rf / (tp_rf + fp_rf)
recall_rf = tp_rf / (tp_rf + fn_rf)
f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
accuracy_rf = (tp_rf + tn_rf) / (tp_rf + tn_rf + fp_rf + fn_rf)

print("F1 Score for Random Forest:", f1_rf)
print("Accuracy for Random Forest:", accuracy_rf)


# In[ ]:





# In[ ]:





# In[ ]:




