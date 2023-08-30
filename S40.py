#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importation du table S40 


# In[2]:


import pandas as pd
vente_S40=pd.read_csv("S40_data.csv")


# In[3]:


# affichage du table 


# In[4]:


vente_S40


# In[5]:


# selectionner les colonnes necessaires :


# In[6]:


columns=['Code_Site','Date_comptable','Code_Article','Code_Fam_Stat_Article_1','Code_Fam_Stat_Article_2','Code_Fam_Stat_Article_3',
'Code_Categorie_Article',
'Affaire',
'Mt_Ligne_HT',
'Qté_facturée']


# In[7]:


vente_S40 = vente_S40.loc[:,columns]


# In[8]:


# nouvelle table


# In[9]:


vente_S40


# In[10]:


# changement des types de quelques colonnes :


# In[11]:


vente_S40['Code_Fam_Stat_Article_1'] = pd.factorize(vente_S40['Code_Fam_Stat_Article_1'])[0]
vente_S40['Date_comptable'] = pd.to_datetime(vente_S40['Date_comptable'])
vente_S40['Code_Fam_Stat_Article_2'] = pd.factorize(vente_S40['Code_Fam_Stat_Article_2'])[0]
vente_S40['Code_Fam_Stat_Article_3'] = pd.factorize(vente_S40['Code_Fam_Stat_Article_3'])[0]


# In[12]:


# filter les données par catégorie des articles :


# In[13]:


# Groupement des données par catégorie d'article
groupes = vente_S40.groupby('Code_Categorie_Article')

# Itération sur les groupes et enregistrement dans des fichiers CSV
for categorie, groupe in groupes:
    nom_fichier = f"categorie_S40{categorie}.csv"  # Nom du fichier CSV
    groupe.to_csv(nom_fichier, index=False)


# In[14]:


# on travaille sur la catégorie la plus demandée:


# In[15]:


# Catégorie Palim : Produit Alimentaire 


# In[16]:


Palim=pd.read_csv("categorie_S40PALIM.csv")


# In[17]:


Palim


# In[18]:


# changement date comptable to datetime :


# In[19]:


Palim['Date_comptable'] = pd.to_datetime(Palim['Date_comptable'])


# In[20]:


# Changement les colonnes code_site , code_article et affaire au valeurs numériques :


# In[21]:


# Code_site 


# In[22]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Code_Site'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Code_Site'] = Palim['Code_Site'].map(mapping)

# Vérifier les modifications
print(Palim['Code_Site'])


# In[23]:


# code_article


# In[24]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Article"
valeurs_uniques = Palim['Code_Article'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Code_Article'] = Palim['Code_Article'].map(mapping)

# Vérifier les modifications
print(Palim['Code_Article'])


# In[25]:


# supprimer la colonne catégorie des articles puisque on a un seul catégorie :


# In[26]:


colonn='Code_Categorie_Article'
Palim=Palim.drop(colonn,axis=1)


# In[27]:


# affaire 


# In[28]:


# Obtenir toutes les valeurs uniques de la colonne "Affaire"
valeurs_uniques = Palim['Affaire'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Affaire'] = Palim['Affaire'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[29]:


# filtrage du table on date croissant 


# In[30]:


Palim_1 = Palim.sort_values('Date_comptable')


# In[31]:


# enregistrement de nouvelle table


# In[32]:


Palim_1.to_csv('Palim_S40_Date.csv')


# In[33]:


palim=pd.read_csv('Palim_S40_Date.csv')


# In[34]:


palim


# In[35]:


col='Unnamed: 0'


# In[36]:


palim=palim.drop(col,axis=1)


# In[37]:


palim['Date_comptable'] = pd.to_datetime(palim['Date_comptable'])
palim.info()


# In[38]:


# machine learning :


# # Random Forest Regressor

# In[39]:


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


# In[40]:


# Calcule R²


# In[41]:


from sklearn.metrics import r2_score


# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Afficher le coefficient de détermination (R²)
print("Coefficient de détermination (R²) :", r2)


# In[42]:


# Optimiser les hyper-paramètres:


# In[43]:


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


# In[44]:


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
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
print('Quantité prédite:', y_pred)


# In[45]:


from sklearn.metrics import r2_score


# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Afficher le coefficient de détermination (R²)
print("Coefficient de détermination (R²) :", r2)


# In[ ]:





# In[46]:


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


# In[47]:


import pandas as pd

# Créer un DataFrame pour comparer les valeurs réelles et prédites
resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test, 'Valeurs_Prédites': y_pred})

print(resultats_comparaison)


# In[48]:


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

# Créer un graphique de ligne pour afficher les valeurs réelles et prédites en fonction du temps
plt.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Réelles'], label='Valeurs Réelles')
plt.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Prédites'], label='Valeurs Prédites')

# Ajouter une légende et un titre au graphique
plt.legend()
plt.xlabel("Date comptable")
plt.ylabel("Quantité")
plt.title("Comparaison des valeurs réelles et prédites en fonction du temps")

# Afficher le graphique
plt.show()


# In[49]:


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


# In[50]:


import pandas as pd
import plotly.express as px

# Supposons que resultats_comparaison est un DataFrame contenant les vraies valeurs et les valeurs prédites pour chaque produit et la date comptable
# Créer un DataFrame pour comparer les valeurs réelles et prédites pour chaque produit
resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test, 'Valeurs_Prédites': y_pred, 'Code_Article': X_test['Code_Article']})

# Trouver les articles les plus vendus (par exemple, les 5 articles les plus vendus)
top_articles = resultats_comparaison.groupby('Code_Article')['Valeurs_Réelles'].sum().nlargest(5).index

# Créer les graphiques pour chaque article le plus vendu
for article in top_articles:
    # Filtrer les données pour l'article sélectionné
    donnees_article = resultats_comparaison[resultats_comparaison['Code_Article'] == article]

    # Créer le graphique de dispersion pour l'article
    fig_article = px.scatter(donnees_article, x='Valeurs_Réelles', y='Valeurs_Prédites', hover_data=['Code_Article'],
                             title=f'Prédictions pour l\'article {article}')

    # Afficher le graphique
    fig_article.show()


# # gardien boosting regressor

# In[51]:


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


# # hyper-paramètres

# In[52]:


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


# In[53]:


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
model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
model.fit(X_train1, y_train1)

# Faire des prédictions sur l'ensemble de test
y_pred1 = model.predict(X_test1)

# Calculer le RMSE
rmse = mean_squared_error(y_test1, y_pred1, squared=False)
print('RMSE:', rmse)


# In[54]:


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


# In[55]:


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
gb_model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Créer et entraîner le modèle RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
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


# In[56]:


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


# In[57]:


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


# In[58]:


# Tracer les prédictions réelles par rapport aux valeurs réelles pour les deux modèles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, label="Gradient Boosting", alpha=0.7)
plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.7)
plt.xlabel("Valeurs Réelles")
plt.ylabel("Prédictions")
plt.title("Comparaison des prédictions réelles et prédites")
plt.legend()
plt.grid(True)
plt.show()


# In[59]:


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


# In[60]:


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


# In[61]:


# Comparaison des performances par sous-ensemble de données (par exemple, par code article)

unique_code_articles = X_test['Code_Article'].unique()

for code_article in unique_code_articles:
    X_code_article = X_test[X_test['Code_Article'] == code_article]
    y_code_article = y_test[X_test['Code_Article'] == code_article]
    
    y_pred_gb_code_article = gb_model.predict(X_code_article)
    y_pred_rf_code_article = rf_model.predict(X_code_article)
    
    rmse_gb_code_article = mean_squared_error(y_code_article, y_pred_gb_code_article, squared=False)
    rmse_rf_code_article = mean_squared_error(y_code_article, y_pred_rf_code_article, squared=False)
    
    print(f"Code Article {code_article}:")
    print("RMSE Gradient Boosting :", rmse_gb_code_article)
    print("RMSE Random Forest :", rmse_rf_code_article)
    print()


# In[62]:


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


# In[63]:


results_df.to_csv('S40_RF_GB.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


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


# In[65]:


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


# Gradient Boosting:

# True Positives (TP) = 2137
# True Negatives (TN) = 0
# False Positives (FP) = 0
# False Negatives (FN) = 71

# In[66]:


# Calculate True Positives, True Negatives, False Positives, False Negatives
tp_gb = 2137
tn_gb = 0
fp_gb = 0
fn_gb = 71

# Calculate Precision, Recall, F1 Score, and Accuracy
precision_gb = tp_gb / (tp_gb + fp_gb)
recall_gb = tp_gb / (tp_gb + fn_gb)
f1_gb = 2 * (precision_gb * recall_gb) / (precision_gb + recall_gb)
accuracy_gb = (tp_gb + tn_gb) / (tp_gb + tn_gb + fp_gb + fn_gb)

print("F1 Score for Gradient Boosting:", f1_gb)
print("Accuracy for Gradient Boosting:", accuracy_gb)


# Random Forest

# True Positives (TP) = 2208: This means the model correctly predicted 2208 samples as positive.
# True Negatives (TN) = 0: Since there is only one class, there are no true negatives.
# False Positives (FP) = 0: The model did not make any false positive predictions.
# False Negatives (FN) = 0: The model did not make any false negative predictions.

# In[67]:


# Calculate True Positives, True Negatives, False Positives, False Negatives
tp_rf = 2208
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





# In[68]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

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

# Créer le modèle GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)

# Perform cross-validation with 5 folds
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Print the R² scores for each fold
print("R² scores for each fold:", scores)

# Calculate the mean R² score
mean_r2 = scores.mean()
print("Mean R² score:", mean_r2)


# In[69]:


from sklearn.ensemble import RandomForestRegressor

# Créer le modèle RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

# Perform cross-validation with 5 folds
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Print the R² scores for each fold
print("R² scores for each fold:", scores)

# Calculate the mean R² score
mean_r2 = scores.mean()
print("Mean R² score:", mean_r2)


# In[70]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

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

# Créer les modèles GradientBoostingRegressor et RandomForestRegressor
gb_model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

# Fonction pour évaluer les modèles avec différentes métriques
def evaluate_model(model, X, y):
    rmse = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
    mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
    r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return rmse, mae, r2, f1, accuracy

# Évaluer les modèles
rmse_gb, mae_gb, r2_gb, f1_gb, accuracy_gb = evaluate_model(gb_model, X, y)
rmse_rf, mae_rf, r2_rf, f1_rf, accuracy_rf = evaluate_model(rf_model, X, y)

# Afficher les résultats
print("Gradient Boosting:")
print("RMSE:", rmse_gb)
print("MAE:", mae_gb)
print("R²:", r2_gb)
print("F1-score:", f1_gb)
print("Accuracy:", accuracy_gb)

print("\nRandom Forest:")
print("RMSE:", rmse_rf)
print("MAE:", mae_rf)
print("R²:", r2_rf)
print("F1-score:", f1_rf)
print("Accuracy:", accuracy_rf)


# In[ ]:





# In[71]:


from sklearn.model_selection import train_test_split

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

# Diviser les données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


# Diviser l'ensemble d'entraînement en ensembles d'entraînement réel (80%) et de validation (20%)
X_train_real, X_val, y_train_real, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[73]:


from sklearn.ensemble import GradientBoostingRegressor

# Créer le modèle GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement réel
model.fit(X_train_real, y_train_real)

# Évaluer les performances sur l'ensemble de validation
y_pred_val = model.predict(X_val)

# Calculer les métriques d'évaluation sur l'ensemble de validation
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

# Afficher les résultats
print("Résultats sur l'ensemble de validation:")
print("RMSE:", rmse_val)
print("MAE:", mae_val)
print("R²:", r2_val)


# In[74]:


# Évaluer les performances sur l'ensemble de test
y_pred_test = model.predict(X_test)

# Calculer les métriques d'évaluation sur l'ensemble de test
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Afficher les résultats
print("Résultats sur l'ensemble de test:")
print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("R²:", r2_test)


# In[75]:


from sklearn.model_selection import cross_val_score

# Créer le modèle GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)

# Effectuer la validation croisée avec 5 plis
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculer les métriques d'évaluation moyennes pour chaque pli
cv_rmse_scores = (-cv_scores)**0.5
cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

# Afficher les résultats
print("R² scores for each fold:", cv_r2_scores)
print("Mean R² score:", cv_r2_scores.mean())


# In[76]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Diviser les données en ensembles d'entraînement, de validation et de test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Créer et entraîner le modèle Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de validation
y_pred_val = model.predict(X_val)

# Évaluer les performances sur l'ensemble de validation
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
r2_val = r2_score(y_val, y_pred_val)

print("Résultats sur l'ensemble de validation:")
print("RMSE:", rmse_val)
print("R²:", r2_val)

# Faire des prédictions sur l'ensemble de test
y_pred_test = model.predict(X_test)

# Évaluer les performances sur l'ensemble de test
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)

print("\nRésultats sur l'ensemble de test:")
print("RMSE:", rmse_test)
print("R²:", r2_test)


# In[77]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# Prédire les quantités facturées pour 1 mois et 3 mois
nouvelles_données_1_mois = X_test.iloc[-1].copy()
nouvelles_données_1_mois['Mois'] += 1
nouvelles_predictions_1_mois_gb = model_gb.predict(nouvelles_données_1_mois.values.reshape(1, -1))
nouvelles_predictions_1_mois_rf = model_rf.predict(nouvelles_données_1_mois.values.reshape(1, -1))

nouvelles_données_3_mois = X_test.iloc[-1].copy()
nouvelles_données_3_mois['Mois'] += 3
nouvelles_predictions_3_mois_gb = model_gb.predict(nouvelles_données_3_mois.values.reshape(1, -1))
nouvelles_predictions_3_mois_rf = model_rf.predict(nouvelles_données_3_mois.values.reshape(1, -1))

# Afficher les résultats
print("Résultats du modèle Gradient Boosting :")
print("RMSE :", rmse_gb)
print("R² :", r2_gb)
print("MAE :", mae_gb)

print("\nRésultats du modèle Random Forest :")
print("RMSE :", rmse_rf)
print("R² :", r2_rf)
print("MAE :", mae_rf)

print("\nPrédictions pour 1 mois (Gradient Boosting) :", nouvelles_predictions_1_mois_gb[0])
print("Prédictions pour 1 mois (Random Forest) :", nouvelles_predictions_1_mois_rf[0])

print("\nPrédictions pour 3 mois (Gradient Boosting) :", nouvelles_predictions_3_mois_gb[0])
print("Prédictions pour 3 mois (Random Forest) :", nouvelles_predictions_3_mois_rf[0])


# In[99]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

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

# Utiliser la dernière date de l'ensemble de test pour faire une prédiction pour un mois à l'avenir
derniere_date_test = palim['Date_comptable'].max()
date_future_1_mois = derniere_date_test + pd.DateOffset(months=1)
date_future_3_mois = derniere_date_test + pd.DateOffset(months=3)

# Préparer les données pour la prédiction
donnees_prediction = X_test.iloc[-1].copy()  # Utiliser la dernière ligne de l'ensemble de test comme données pour la prédiction
donnees_prediction['Mois'] = date_future_1_mois.month  # Modifier le mois pour la prédiction d'un mois à l'avenir

# Faire la prédiction pour un mois à l'avenir avec les deux modèles
prediction_1_mois_gb = model_gb.predict(donnees_prediction.values.reshape(1, -1))
prediction_1_mois_rf = model_rf.predict(donnees_prediction.values.reshape(1, -1))

# Afficher les prédictions
print("\nPrédiction pour un mois à l'avenir avec Gradient Boosting :", prediction_1_mois_gb[0])
print("Prédiction pour un mois à l'avenir avec Random Forest :", prediction_1_mois_rf[0])


# In[102]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Utiliser la dernière date de l'ensemble de test pour faire une prédiction pour 1 mois à l'avenir
derniere_date_test = palim['Date_comptable'].max()
date_future_1_mois = derniere_date_test + pd.DateOffset(months=1)

# Préparer les données pour la prédiction
donnees_prediction = X_test.iloc[-1].copy()  # Utiliser la dernière ligne de l'ensemble de test comme données pour la prédiction
donnees_prediction['Mois'] = date_future_1_mois.month  # Modifier le mois pour la prédiction d'un mois à l'avenir

# Faire la prédiction pour 1 mois à l'avenir avec les deux modèles
prediction_1_mois_gb = model_gb.predict(donnees_prediction.values.reshape(1, -1))
prediction_1_mois_rf = model_rf.predict(donnees_prediction.values.reshape(1, -1))

# Créer un DataFrame pour comparer les valeurs réelles et les prédictions
resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test, 'Valeurs_Prédites_GB': model_gb.predict(X_test), 'Valeurs_Prédites_RF': model_rf.predict(X_test)})

# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
resultats_comparaison['Date_comptable'] = palim.loc[X_test.index, 'Date_comptable']

# Trier les données par date pour le tracé
resultats_comparaison.sort_values(by='Date_comptable', inplace=True)

# Créer une figure et des axes pour le graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Tracer les valeurs réelles en fonction du temps
ax.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Réelles'], label='Valeurs Réelles', marker='o')

# Tracer les prédictions en fonction du temps
ax.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Prédites_GB'], label='Prédictions GB', marker='x')
ax.plot(resultats_comparaison['Date_comptable'], resultats_comparaison['Valeurs_Prédites_RF'], label='Prédictions RF', marker='s')

# Ajouter une légende, un titre et des labels d'axe
ax.legend()
ax.set_xlabel("Date comptable")
ax.set_ylabel("Quantité")
ax.set_title("Comparaison des valeurs réelles et des prédictions en fonction du temps")

# Formatter les étiquettes de l'axe des x en dates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Ajouter une grille pour faciliter la lecture du graphique
ax.grid(True)

# Faire pivoter les étiquettes de l'axe des x pour une meilleure lisibilité
plt.xticks(rotation=45)

# Afficher le graphique
plt.tight_layout()
plt.show()


# In[104]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Utiliser la dernière date de l'ensemble de test pour faire une prédiction pour 1 mois à l'avenir
derniere_date_test = palim['Date_comptable'].max()
date_future_1_mois = derniere_date_test + pd.DateOffset(months=1)

# Préparer les données pour la prédiction
donnees_prediction = X_test.iloc[-1].copy()  # Utiliser la dernière ligne de l'ensemble de test comme données pour la prédiction
donnees_prediction['Mois'] = date_future_1_mois.month  # Modifier le mois pour la prédiction d'un mois à l'avenir

# Faire la prédiction pour 1 mois à l'avenir avec les deux modèles
prediction_1_mois_gb = model_gb.predict(donnees_prediction.values.reshape(1, -1))
prediction_1_mois_rf = model_rf.predict(donnees_prediction.values.reshape(1, -1))

# Créer une liste de dates pour les prédictions futures (30 jours à partir de la dernière date de test)
dates_futures = pd.date_range(start=derniere_date_test + pd.DateOffset(days=1), periods=30, freq='D')

# Répéter les valeurs de prédiction pour chaque jour de la période de prédiction
prediction_1_mois_gb = [prediction_1_mois_gb[0]] * len(dates_futures)
prediction_1_mois_rf = [prediction_1_mois_rf[0]] * len(dates_futures)

# Créer une DataFrame pour stocker les prédictions futures
df_predictions = pd.DataFrame({'Dates_futures': dates_futures, 'Prédiction_GB': prediction_1_mois_gb, 'Prédiction_RF': prediction_1_mois_rf})

# Afficher les prédictions futures en fonction de la date comptable
fig, ax = plt.subplots(figsize=(10, 6))

# Tracer les prédictions futures GB en fonction du temps
ax.plot(df_predictions['Dates_futures'], df_predictions['Prédiction_GB'], label='Prédiction GB', marker='x')

# Tracer les prédictions futures RF en fonction du temps
ax.plot(df_predictions['Dates_futures'], df_predictions['Prédiction_RF'], label='Prédiction RF', marker='s')

# Ajouter une légende, un titre et des labels d'axe
ax.legend()
ax.set_xlabel("Date comptable")
ax.set_ylabel("Quantité")
ax.set_title("Prédictions futures d'un mois à l'avenir en fonction du temps")

# Formatter les étiquettes de l'axe des x en dates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Ajouter une grille pour faciliter la lecture du graphique
ax.grid(True)

# Faire pivoter les étiquettes de l'axe des x pour une meilleure lisibilité
plt.xticks(rotation=45)

# Afficher le graphique
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[78]:



resultats_comparaison = pd.DataFrame({'Valeurs_Réelles': y_test, 'Valeurs_Prédites': y_pred, 'Code_Article': X_test['Code_Article'],'Code_Site':X_test['Code_Site']})


# In[79]:


resultats_comparaison


# In[80]:


# Réajouter la colonne "Date_comptable" à partir du DataFrame original "palim"
resultats_comparaison['Date_comptable'] = palim.loc[X_test.index, 'Date_comptable']

# Assurez-vous que "Date_comptable" est au format de date
resultats_comparaison['Date_comptable'] = pd.to_datetime(resultats_comparaison['Date_comptable'])

# Trier les données par date pour le tracé
resultats_comparaison.sort_values(by='Date_comptable', inplace=True)


# In[81]:


resultats_comparaison


# In[82]:


resultats_comparaison.to_csv('S40_predicte.csv')


# In[83]:


resultats_comparaison.to_excel('S40_predicte.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




