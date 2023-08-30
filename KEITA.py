#!/usr/bin/env python
# coding: utf-8

# # GROUPE KEITA S.A
# 

# Le Groupe KEITA a un rôle de coordination et de supervision de ses filiales avec une équipe constitué de jeunes cadres qui ont la volonté de faire du Groupe KEITA une référence sous régionale. Ce défis de devenir une référence sous régionale est une vision de son Président Fondateur Monsieur Modibo KEITA, un Self Made Man aujourd’hui à la tête de l’un des plus grands Groupe Privé au Mali.
# 

# 
# # Les Faits Importants
# Un des plus grands employeurs du Mali soit près de 1000 emplois permanents et des centaines d’emplois indirects, Leader dans le **commerce agro-alimentaire** au Mali, Leader dans **l’industrie agro-alimentaire** au Mali, Leader dans **l’agriculture mécanisée** au Mali, Possède une flotte de 430 camions et citernes, Première société des pays hinterland au Port d’Abidjan et de Conakry, Premier opérateur économique privé à se doter du système ERP intégré dans l’ensemble de ses filiales aussi du logiciel de Gestion de Trésorerie SAGE XRT (n°1 mondial), L’un des premiers bénéficiaires d’un financement privé par le pool bancaire BAD,BOAD et Banque Atlantique, Première société africaine à doter d’un complexe d’unités combinées unique en son genre au monde, selon l’équipementier BUHLER, N°1 mondial en usines de Minoteries.

# # Filiales du Groupe KEITA S.A

# ## Grand Distributeur Céréalier du Mali (GDCM-SA)
# 
# Filiale du Groupe KEITA est une société de droit malien créée en 1994. Elle a pour activité l’importation et l’exportation de produits céréaliers et alimentaires. Elle assure leurs distributions sous forme de vente en gros et demis gros sur le marché national.
# GDCM-SA est de nos jours, la première société du Mali dans la vente et dans la distribution de produits céréaliers grâce à son important réseau de distribution reparti sur l’ensemble du territoire national avec des magasins à Bamako et dans toutes les régions du Mali…
# 
# 

# ![GDCM.jpeg](attachment:GDCM.jpeg)

# ## M3-SA
# Le Moulin Moderne du Mali (M3 – S.A) situé à Ségou, est la filiale industrie-alimentaire du Groupe KEÏTA crée en 2008.
# Certifiée Iso 9001 de 2015 depuis 201, la société M3-SA est composée de 13 unités de production sur le même site dont :
# 
# L’unité Minoterie
# 
# L’unité AB(Aliment Bétail)
# 
# L’unité STB (Sucre Tomate Biscuit)
# 
# La Rizerie
# …

# ![rizerie.jpeg](attachment:rizerie.jpeg)

# ## Complexe Agro Pastoral et Industriel (CAI-SA)
# Situe à 60 km de la ville de Ségou dans la zone de l’office du Niger près du village de Diado, le Complexe Agro Pastoral et Industriel crée en 2009 est la filiale agro-industrielle du Groupe KEITA qui s’étend sur une superficie de 20 000 Hectares.
# Cette Filiale est dotée d’équipement ultra moderne de Marques : JOHN DEERE, GRUMME, CLASS, CATARPILLARD…
# 
# 

# ![cai.jpeg](attachment:cai.jpeg)

# ## ChôlaTrading Transport CT2-SA
# Filiale du Groupe KEITA est la branche spécialisée dans le transport de marchandise solide et liquide ainsi que la commercialisation d’équipement de transport et de matériels agricoles.
# Doté d’un parc auto performant avec des camions neufs et bien entretenus, CT2-SA assure le transport des filiales sœurs mais également celles des clients particuliers depuis les ports ouest africains vers tout le Mali. Tous ses camions sont dotés de système de géolocalisation capables de connaître leurs positions et d’autres informations…

# ![Citerne.jpeg](attachment:Citerne.jpeg)

# In[ ]:





# # Connection de la base et jupyter : 

# Pyodbc est une bibliothèque Python qui permet aux programmes Python d'interagir avec des bases de données relationnelles à l'aide de l'interface ODBC (Open Database Connectivity). ODBC est une API standard pour accéder aux bases de données, indépendante du système d'exploitation et du SGBD utilisé. Pyodbc utilise des pilotes ODBC pour se connecter aux bases de données, ce qui permet une compatibilité avec une grande variété de SGBD, tels que Microsoft SQL Server, Oracle, MySQL, PostgreSQL, etc.
# 
# Pour utiliser Pyodbc avec SQL Server Management Studio (SSMS), il faut d'abord installer les pilotes ODBC pour SQL Server. Ensuite, il faut installer la bibliothèque Pyodbc à l'aide d'un gestionnaire de paquets tel que pip. Une fois que la bibliothèque est installée, il est possible de se connecter à la base de données SQL Server depuis Python en utilisant les paramètres de connexion appropriés.

# In[1]:


import pyodbc


# In[2]:


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=LAPTOP-I2LRJMG3\\BASEDONNÉE;'
                      'Database=DWH_KEITA;'
                      'Trusted_Connection=yes;')


# # Importation des Tables : 

# # Base Société :

# On Commence par les Sociétés pour mieux comprendre les différentes types , nature et autres facteurs de groupe KEITA 

# In[3]:


# création de cursor pour faire la connection : 


# In[4]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM Dim_Société')
for row in cursor:
    print(row)


# In[5]:


# on utilise la bib PANDAS pour manipuler la base et faire les transformations 


# In[6]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM Dim_Société', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\Dim_Société.csv', index=False)


# In[7]:


# lire la base en CSV 
societe=pd.read_csv("Dim_Société.csv")


# In[8]:


# Affichage de la base :
societe


# La base de données semble contenir des informations sur différentes sociétés, telles que leur nom, leur raison sociale, leur modèle comptable, leur pays, leur numéro de SIREN, etc. Elle contient également plusieurs colonnes relatives à la facturation, telles que la facture simplifiée, le type de facturation, l'élément de facturation, etc. Enfin, la base de données comporte une colonne "ROWID" qui peut être utilisée comme identifiant unique pour chaque ligne de la base.
# 
# 
# on trouve qu'on a 7 sociétés : 
# 
# **GRAND DISTRIBUTEUR CEREALIER AUMALI**
# 
# **Groupes de Site Mali**
# 
# **Groupe de Sites Hors Mali**
# 
# **MOULIN MODERNE DU MALI**
# 
# **CHÔLA TRADING TRANSPORT**
# 
# **COMPLEXE AGRO PASTORAL ET INDUSTRIE**
# 
# **GROUPE KEITA SA**
# 
# 
# 
# 
# GRAND DISTRIBUTEUR CEREALIER AUMALI : Il s'agit probablement d'une entreprise opérant dans le secteur de la distribution de céréales, située au Mali. 
# 
# Groupes de Site Mali : Il est difficile de donner une signification précise sans plus de contexte. Il pourrait s'agir d'une désignation pour un ensemble de sites géographiques appartenant à une même entreprise ou organisation, situés au Mali.
# 
# Groupe de Sites Hors Mali : Il s'agit probablement d'une désignation pour un ensemble de sites géographiques appartenant à une même entreprise ou organisation, mais situés en dehors du Mali.
# 
# MOULIN MODERNE DU MALI : Il s'agit d'une entreprise malienne opérant dans le secteur de la transformation de céréales et de la production de farine.
# 
# CHÔLA TRADING TRANSPORT : Il s'agit probablement d'une entreprise opérant dans le secteur du commerce et du transport, mais sans plus de contexte, il est difficile de donner une signification plus précise.
# 
# COMPLEXE AGRO PASTORAL ET INDUSTRIE : Il s'agit probablement d'une entreprise opérant dans le secteur agro-pastoral et industriel, qui pourrait être impliquée dans la production et la transformation de produits agricoles.
# 
# GROUPE KEITA SA : Il s'agit probablement d'une entreprise appartenant à la famille Keita, qui est un nom de famille courant en Afrique de l'Ouest, et qui opère dans divers secteurs d'activité. La désignation "SA" indique qu'il s'agit d'une société anonyme, une forme courante d'entreprise en droit français.
# 
# **GROUPE KEITA : 3 GRAND FAMILLES : COMMERCE AGRO-ALIMENTAIRE / INDUSTRIE AGRO-ALIMENTAIRE ET AGRO- MECANISEE**
# 
# la prévision de la vente doivent etre par famille , donc il faut grouper les familles ,  les ventes , les articles et tt facteurs qui influent chaque famille 

# ![Groupe%20Keita.png](attachment:Groupe%20Keita.png)

# # Base Site : 

# Aprés avoir etudier les différentes sociétés du groupe Keita , il faut faire une etude exploratoire sur les sociétés à l'aide des sites 

# In[9]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM Dim_Site')
for row in cursor:
    print(row)


# In[10]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM Dim_Site', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\Dim_Site.csv', index=False)


# In[11]:


site_societe = pd.read_csv('Dim_Site.csv')


# In[12]:


site_societe


# # semaine 2 : analyse exploratoire sur les sites des sociétés 

# On trouve qu'on a pour chaque sociétés plusieurs sites au Mali et Hors Mali : 
# 

# Le groupe en question est une entité importante dans le domaine agricole, qui regroupe trois grandes familles : le commerce agricole, l'industrie agricole et les agriculteurs mécanisés. Chacune de ces familles a son propre rôle à jouer dans la chaîne de production alimentaire.
# 
# 
# 

# La famille du commerce agro-alimentaire est divisée en deux sous-familles : GDC et CT2. GDC compte 11 sociétés locales et 4 sociétés aux COTE D'IVOIRE , SENEGAL , GHANA ET GUINEE, tandis que CT2 a 4 sociétés locales et 3 sociétés COTE D'IVOIRE , GUINEE ET SENEGAL . Ces sociétés sont impliquées dans la vente et la distribution de produits agricoles, tels que les céréales, les légumes et les fruits, ainsi que dans l'importation et l'exportation de ces produits. Elles ont un rôle crucial dans l'approvisionnement des consommateurs et des industries alimentaires en matières premières.
# 

# ![GRAND%20DISTRIBUTEUR%20CEREALIER%20AUMALI.png](attachment:GRAND%20DISTRIBUTEUR%20CEREALIER%20AUMALI.png)

# ![GRAND%20DISTRIBUTEUR%20CEREALIER%20AUMALI%202.png](attachment:GRAND%20DISTRIBUTEUR%20CEREALIER%20AUMALI%202.png)

# ![carte-villes-mali.jpg](attachment:carte-villes-mali.jpg)

# ![GRAND%20DISTRIBUTEUR%20CEREALIER%20.%20HORS%20MALI.png](attachment:GRAND%20DISTRIBUTEUR%20CEREALIER%20.%20HORS%20MALI.png)

# ![CH%C3%94LA%20TRADING%20TRANSPORT.png](attachment:CH%C3%94LA%20TRADING%20TRANSPORT.png)

# La famille de l'industrie agricole comprend la sous-famille M3, qui compte 6 sociétés locales et 3 sociétés aux COTE D'IVOIRE , GUINEE ET SENEGAL. Ces entreprises sont impliquées dans la transformation des produits agricoles en produits alimentaires finis, tels que les farines, les huiles et les conserves. Elles jouent un rôle important dans la création de valeur ajoutée à partir des matières premières et dans la production de produits de qualité pour les consommateurs.
# 
# 

# ![MOULIN%20MODERNE%20DU%20MALI.png](attachment:MOULIN%20MODERNE%20DU%20MALI.png)

# La famille des agriculteurs mécanisés est représentée par la sous-famille CAI, qui compte 2 sociétés locales et 1 société au  COTE D'IVOIRE. Ces entreprises fournissent des services de mécanisation agricole aux agriculteurs locaux, tels que la location de machines agricoles, la fourniture de semences et d'engrais, ainsi que des conseils techniques pour optimiser la production agricole. Ils contribuent ainsi à l'amélioration de la productivité et de la rentabilité des exploitations agricoles.
# 
# 

# ![COMPLEXE%20AGRO%20PASTORAL%20ET%20INDUSTRIE.png](attachment:COMPLEXE%20AGRO%20PASTORAL%20ET%20INDUSTRIE.png)

# En somme, le groupe est un acteur important dans le secteur agricole, qui regroupe différentes familles et sous-familles pour assurer une production et une distribution alimentaire efficaces et durables. Les sociétés locales et internationales impliquées dans ce groupe travaillent ensemble pour répondre aux besoins des consommateurs et des industries alimentaires, tout en contribuant au développement économique et social de la région.
# 

# In[13]:


# manipulation de la base : 


# In[14]:


site_societe_columns=site_societe.columns
site_societe_columns


# In[15]:


site_societe.info()


# In[16]:


site_societe.describe()


# In[17]:


# nous avons extraire une nouvelle base avec les données neccessaires seulement : 


# In[18]:


site__societe = site_societe.loc[:, ['Id_dimSite','UPDTICK_0','Site','Nom','Intitulé_court','Pays','Production','Vente','Achat','Dépôt','Finance','Site financier']]


# In[19]:


site__societe 


# In[20]:


# nous avons filtrer la base pour extraire les site par societe : GDC / CT2 / CAI et M3


# # GDC

# In[21]:


# Spécifier les id_dimsite à filtrer dans la base :
id_GDC = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# Filtrer les données en fonction de la liste d'id_dimsite
site_GDC = site__societe[site__societe['Id_dimSite'].isin(id_GDC)]

# Afficher le DataFrame filtré
print(site_GDC)


# In[22]:


# enregister la base en csv :
site_GDC=site_GDC.to_csv('site_GDC.csv')


# In[23]:


GDC=pd.read_csv('site_GDC.csv')


# In[24]:


GDC


# In[25]:


# correction :
# site ABIDJAN / DAKAR / GHANA / CONAKRY 
# changement des pays : 
# Liste des ID de dimension spécifiés pour le filtrage
Id_dimSite = [2,3,4,12]

# Nouvelles valeurs pour la colonne "pays"
new_pays_values = ['CD', 'SG', 'GN', 'GU']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
for id_dimsite, new_pays_value in zip(Id_dimSite, new_pays_values):
    GDC.loc[GDC['Id_dimSite'] == id_dimsite, 'Pays'] = new_pays_value

GDC


# On trouve qu'on a deux sites de productions S1010 et S1002 , et une seul site financier qu'est le S1010 , par contre tous les sites ont les services Achat , Vente et Dépot 

# In[26]:


import matplotlib.pyplot as plt

# Compter le nombre de sites pour chaque activité
production_count = GDC["Production"].value_counts()
achat_count = GDC["Achat"].value_counts()
vente_count = GDC["Vente"].value_counts()
depot_count = GDC["Dépôt"].value_counts()
finance_count = GDC["Finance"].value_counts()

# Créer un diagramme en barres pour chaque activité
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 12))

production_count.plot(kind="bar", ax=ax[0])
ax[0].set_title("Production")

achat_count.plot(kind="bar", ax=ax[1])
ax[1].set_title("Achat")

vente_count.plot(kind="bar", ax=ax[2])
ax[2].set_title("Vente")

depot_count.plot(kind="bar", ax=ax[3])
ax[3].set_title("Dépôt")

finance_count.plot(kind="bar", ax=ax[4])
ax[4].set_title("Finance")

plt.tight_layout()
plt.show()


# In[27]:


# Créer un diagramme circulaire pour chaque activité
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 12))

production_count.plot.pie(ax=ax[0])
ax[0].set_title("Production")

achat_count.plot.pie(ax=ax[1])
ax[1].set_title("Achat")

vente_count.plot.pie(ax=ax[2])
ax[2].set_title("Vente")

depot_count.plot.pie(ax=ax[3])
ax[3].set_title("Dépôt")

finance_count.plot.pie(ax=ax[4])
ax[4].set_title("Finance")

plt.tight_layout()
plt.show()


# In[28]:


# Compter le nombre de sites par type d'activité
counts = GDC[['Production', 'Vente', 'Achat', 'Dépôt', 'Finance']].apply(pd.Series.value_counts)

# Visualiser les résultats sous forme de diagramme en barres
counts.plot(kind='bar', figsize=(10, 6))
plt.title('Nombre de sites par type d\'activité')
plt.xlabel('Type d\'activité')
plt.ylabel('Nombre de sites')
plt.show()


# # M3

# In[29]:


# Spécifier les id_dimsite à filtrer dans la base :
id_M3 = [16,17,18,19,20,21,22,23,24]

# Filtrer les données en fonction de la liste d'id_dimsite
site_M3 = site__societe[site__societe['Id_dimSite'].isin(id_M3)]

# Afficher le DataFrame filtré
print(site_M3)

# enregister la base en csv :
site_M3=site_M3.to_csv('site_M3.csv')

M3=pd.read_csv('site_M3.csv')

M3


# In[30]:


# correction :
# site ABIDJAN / DAKAR / GUINEE
# changement des pays : 
# Liste des ID de dimension spécifiés pour le filtrage
Id_dimSite = [22,23,24]

# Nouvelles valeurs pour la colonne "pays"
new_pays_values = ['CD', 'GU', 'SN']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
for id_dimsite, new_pays_value in zip(Id_dimSite, new_pays_values):
    M3.loc[M3['Id_dimSite'] == id_dimsite, 'Pays'] = new_pays_value

M3


# On a six sites de productions qui sont situées dans le pays Mali , les sites etrangres sont des sites de points de ventes / achat et stocks 
# On remarque que ces sites du Mali  chacune est une site financier 
# 

# In[31]:


# Compter le nombre de sites par type d'activité
counts = M3[['Production', 'Vente', 'Achat', 'Dépôt', 'Finance']].apply(pd.Series.value_counts)

# Visualiser les résultats sous forme de diagramme en barres
counts.plot(kind='bar', figsize=(10, 6))
plt.title('Nombre de sites par type d\'activité')
plt.xlabel('Type d\'activité')
plt.ylabel('Nombre de sites')
plt.show()


# # CT2

# In[32]:


# Spécifier les id_dimsite à filtrer dans la base :
id_CT2 = [25,28,30,32,33,34,35,36]

# Filtrer les données en fonction de la liste d'id_dimsite
site_CT2 = site__societe[site__societe['Id_dimSite'].isin(id_CT2)]

# Afficher le DataFrame filtré
print(site_CT2)

# enregister la base en csv :
site_CT2=site_CT2.to_csv('site_CT2.csv')

CT2=pd.read_csv('site_CT2.csv')

CT2


# In[33]:


# correction :
# site ABIDJAN / GUINEE
# changement des pays : 
# Liste des ID de dimension spécifiés pour le filtrage
Id_dimSite = [34,35]

# Nouvelles valeurs pour la colonne "pays"
new_pays_values = ['CD', 'GU']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
for id_dimsite, new_pays_value in zip(Id_dimSite, new_pays_values):
    CT2.loc[CT2['Id_dimSite'] == id_dimsite, 'Pays'] = new_pays_value

CT2


# Tous les sites sont des sites de : production , achat , vente et depot 
# et pour le service financier just le siege CT2 : S3001

# In[34]:


# Compter le nombre de sites par type d'activité
counts = CT2[['Production', 'Vente', 'Achat', 'Dépôt', 'Finance']].apply(pd.Series.value_counts)

# Visualiser les résultats sous forme de diagramme en barres
counts.plot(kind='bar', figsize=(10, 6))
plt.title('Nombre de sites par type d\'activité')
plt.xlabel('Type d\'activité')
plt.ylabel('Nombre de sites')
plt.show()


# # CAI

# In[35]:


# Spécifier les id_dimsite à filtrer dans la base :
id_CAI = [26,27,31]

# Filtrer les données en fonction de la liste d'id_dimsite
site_CAI = site__societe[site__societe['Id_dimSite'].isin(id_CAI)]

# Afficher le DataFrame filtré
print(site_CAI)

# enregister la base en csv :
site_CAI=site_CAI.to_csv('site_CAI.csv')

CAI=pd.read_csv('site_CAI.csv')

CAI


# In[36]:


# correction :
# site ABIDJAN / DAKAR / GUINEE
# changement des pays : 
# Liste des ID de dimension spécifiés pour le filtrage
Id_dimSite = [31]

# Nouvelles valeurs pour la colonne "pays"
new_pays_values = ['CD']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
for id_dimsite, new_pays_value in zip(Id_dimSite, new_pays_values):
    CAI.loc[CAI['Id_dimSite'] == id_dimsite, 'Pays'] = new_pays_value

CAI


# cette groupe CAI ces trois sites sont tous de : production , vente , achat , depot et finance 

# In[37]:


# Compter le nombre de sites par type d'activité
counts = CAI[['Production', 'Vente', 'Achat', 'Dépôt', 'Finance']].apply(pd.Series.value_counts)

# Visualiser les résultats sous forme de diagramme en barres
counts.plot(kind='bar', figsize=(10, 6))
plt.title('Nombre de sites par type d\'activité')
plt.xlabel('Type d\'activité')
plt.ylabel('Nombre de sites')
plt.show()


# # SIEGE KEITA

# In[38]:


# Spécifier les id_dimsite à filtrer dans la base :
id_KEITA = [29]

# Filtrer les données en fonction de la liste d'id_dimsite
site_KEITA = site__societe[site__societe['Id_dimSite'].isin(id_KEITA)]

# Afficher le DataFrame filtré
print(site_KEITA)

# enregister la base en csv :
site_KEITA=site_KEITA.to_csv('site_KEITA.csv')

KEITA=pd.read_csv('site_KEITA.csv')

KEITA


# 
# # Connexion à la base Categories des articles pour analyser les différentes catégories  

# In[39]:


# création de cursor pour faire l'execution de la table : dim_categorie_article 
# afficher les lignes 


# In[40]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM dim_Categorie_Article')
for row in cursor:
    print(row)


# In[41]:


# Transformation de la base en base CSV pour la traiter avec les différentes bibliotheques de python : 
# pour cette etape on a besoin de PANDAS pour lire la base SQL et la transformer en CSV :


# In[42]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM dim_Categorie_Article', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\dim_Categorie_Article.csv', index=False)


# In[43]:


categorie_article=pd.read_csv('dim_Categorie_Article.csv')


# In[44]:


categorie_article


# *** id_dim_categorie_article *** : identifiant numérique unique pour chaque catégorie d'article.
# 
# *** Catégorie *** : code de la catégorie d'article. Les codes sont des chaînes de caractères de 4 caractères maximum.
# 
# *** Texte *** : description textuelle de la catégorie d'article.

# # Articles

# In[45]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM Dim_Article', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\dim_article.csv', index=False)


# In[46]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM Dim_Article')
for row in cursor:
    print(row)


# In[47]:


article=pd.read_csv("dim_article.csv")


# In[48]:


article.head()


# # S00 CATEGORIES ARTICLES 

# In[49]:


# Filtre par societe :
# societe siege S00
filtre_article_s00 = article.loc[article['Société'] == 'S00']

# Afficher les résultats
filtre_article_s00


# In[50]:


# calculer le nombre de catégories dans la colonne "catégorie"
nb_categories_s00 = filtre_article_s00['Catégorie'].nunique()
categorie=filtre_article_s00['Catégorie'].unique()
print(f"Le nombre de catégories est : {nb_categories_s00}")
print(f"les differents catégories de la société S00 : {categorie}")


# # S10 CATEGORIES ARTICLES 

# In[51]:


# Filtre par societe :
# societe siege S10
filtre_article_s10 = article.loc[article['Société'] == 'S10']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_categories_s10 = filtre_article_s10['Catégorie'].nunique()
categorie=filtre_article_s10['Catégorie'].unique()
print(f"Le nombre de catégories est : {nb_categories_s10}")
print(f"les differents catégories de la société S10 : {categorie}")
# Afficher les résultats
filtre_article_s10


# # S20 CATEGORIES ARTICLES 

# In[52]:


# Filtre par societe :
# societe siege S20
filtre_article_s20 = article.loc[article['Société'] == 'S20']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_categories_s20 = filtre_article_s20['Catégorie'].nunique()
categorie=filtre_article_s20['Catégorie'].unique()
print(f"Le nombre de catégories est : {nb_categories_s20}")
print(f"les differents catégories de la société S20 : {categorie}")
# Afficher les résultats
filtre_article_s20


# # S30 CATEGORIES ARTICLES 

# In[53]:


# Filtre par societe :
# societe siege S30
filtre_article_s30 = article.loc[article['Société'] == 'S30']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_categories_s30 = filtre_article_s30['Catégorie'].nunique()
categorie=filtre_article_s30['Catégorie'].unique()
print(f"Le nombre de catégories est : {nb_categories_s30}")
print(f"les differents catégories de la société S30 : {categorie}")
# Afficher les résultats
filtre_article_s30


# # S40 CATEGORIES ARTICLES 

# In[54]:


# Filtre par societe :
# societe siege S40
filtre_article_s40 = article.loc[article['Société'] == 'S40']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_categories_s40 = filtre_article_s40['Catégorie'].nunique()
categorie=filtre_article_s40['Catégorie'].unique()
print(f"Le nombre de catégories est : {nb_categories_s40}")
print(f"les differents catégories de la société S40 : {categorie}")
# Afficher les résultats
filtre_article_s40


# # S_VIDE CATEGORIES ARTICLES 

# In[55]:


# Filtre par societe :
# vide
filtre_article_vide = article.loc[article['Société'] == ' ']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_categories_vide = filtre_article_vide['Catégorie'].nunique()
categorie=filtre_article_vide['Catégorie'].unique()
print(f"Le nombre de catégories est : {nb_categories_vide}")
print(f"les differents catégories : {categorie}")
# Afficher les résultats
filtre_article_vide


# # NOMBRE DES CATEGORIES PAR SOCIETES

# In[56]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(['S00', 'S10', 'S20', 'S30', 'S40','Vide'])
y = np.array([nb_categories_s00,nb_categories_s10,nb_categories_s20,nb_categories_s30,nb_categories_s40,nb_categories_vide])

# Création du diagramme à barres
plt.bar(x, y)

# Ajout de labels et de titre
plt.xlabel('Société')
plt.ylabel('Nombre des catégories')
plt.title('Nombre des catégories par société')

# Affichage du diagramme
plt.show()


# Aprés avoir comprendre les différentes catégories des articles  dans chaque site , il faut vérifier les nombres d'articles par site 

# # NOMBRE DES ARTICLES PAR SOCIETES 

# ## NOMBRES DES ARTICLES :

# In[57]:


# calculer le nombre de catégories dans la colonne "catégorie"
nb_articles = article['Article'].nunique()
liste_article=article['Article'].unique()
print(f"Le nombre des articles est : {nb_articles}")
print(f"les differents articles : {liste_article}")


# ## VIDE 

# In[58]:


# Filtre par societe :
# vide
filtre_article_vide = article.loc[article['Société'] == ' ']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_article_vide = filtre_article_vide['Article'].nunique()
liste_article_S_VIDE=filtre_article_vide['Article'].unique()
print(f"Le nombre des articles des societe vide sans sites  est : {nb_article_vide}")
print(f"les differents articles  : {liste_article_S_VIDE}")
# Afficher les résultats
filtre_article_vide


# ## S10

# In[59]:


# Filtre par societe :
# S10
filtre_article_S10 = article.loc[article['Société'] == 'S10']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_article_S10 = filtre_article_S10['Article'].nunique()
liste_article_S10=filtre_article_S10['Article'].unique()
print(f"Le nombre des articles du société S10 est : {nb_article_S10}")
print(f"les differents articles : {liste_article_S10}")
# Afficher les résultats
filtre_article_S10


# ## S20

# In[60]:


# Filtre par societe :
# S20
filtre_article_S20 = article.loc[article['Société'] == 'S20']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_article_S20 = filtre_article_S20['Article'].nunique()
liste_article_S20=filtre_article_S20['Article'].unique()
print(f"Le nombre des articles du société S20 est : {nb_article_S20}")
print(f"les differents articles : {liste_article_S20}")
# Afficher les résultats
filtre_article_S20


# ## S30

# In[61]:


# Filtre par societe :
# S30
filtre_article_S30 = article.loc[article['Société'] == 'S30']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_article_S30= filtre_article_S30['Article'].nunique()
liste_article_S30=filtre_article_S30['Article'].unique()
print(f"Le nombre des articles du société S30 est : {nb_article_S30}")
print(f"les differents articles  : {liste_article_S30}")
# Afficher les résultats
filtre_article_S30


# ## S4O

# In[62]:


# Filtre par societe :
# S40
filtre_article_S40 = article.loc[article['Société'] == 'S40']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_article_S40 = filtre_article_S40['Article'].nunique()
liste_article_S40=filtre_article_S40['Article'].unique()
print(f"Le nombre des articles est : {nb_article_S40}")
print(f"les differents catégories : {liste_article_S40}")
# Afficher les résultats
filtre_article_S40


# ##  SIEGE KEITA : S00

# In[63]:


# Filtre par societe :
# S00
filtre_article_S00 = article.loc[article['Société'] == 'S00']

# calculer le nombre de catégories dans la colonne "catégorie"
nb_article_S00 = filtre_article_S00['Article'].nunique()
liste_article_S00=filtre_article_S00['Article'].unique()
print(f"Le nombre des articles est : {nb_article_S00}")
print(f"les differents catégories : {liste_article_S00}")
# Afficher les résultats
filtre_article_S00


# In[64]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(['S00', 'S10', 'S20', 'S30', 'S40','Vide'])
y = np.array([nb_article_S00,nb_article_S10,nb_article_S20,nb_article_S30,nb_article_S40,nb_article_vide])

# Création du diagramme à barres
plt.bar(x, y)

# Ajout de labels et de titre
plt.xlabel('Société')
plt.ylabel("Nombre d'articles ")
plt.title("Nombre d'articles par société ")

# Affichage du diagramme
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


#Semaine 3


# In[66]:


# Convertir les listes en ensembles
ensemble_S10 = set(liste_article_S10)
ensemble_S20 = set(liste_article_S20)
ensemble_S30 = set(liste_article_S30)
ensemble_S40 = set(liste_article_S40)
ensemble_S_vide = set(liste_article_S_VIDE)
ensemble_S00 = set(liste_article_S00)


# Vérifier s'il y a des articles en commun
articles_communs_S10S20 = ensemble_S10 & ensemble_S20 
articles_communs_S10S30 = ensemble_S10 & ensemble_S30 
articles_communs_S10S40 = ensemble_S10 & ensemble_S40 
articles_communs_S10S00 = ensemble_S10 & ensemble_S00 
articles_communs_S10SVIDE = ensemble_S10 & ensemble_S_vide



# Afficher les résultats
if articles_communs_S10S20:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S10S20)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
    
# Afficher les résultats
if articles_communs_S10S30:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S10S30)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S10S40:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S10S40)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S10S00:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S10S00)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S10SVIDE:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S10SVIDE)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# In[67]:


# Convertir les listes en ensembles
ensemble_S10 = set(liste_article_S10)
ensemble_S20 = set(liste_article_S20)
ensemble_S30 = set(liste_article_S30)
ensemble_S40 = set(liste_article_S40)
ensemble_S_vide = set(liste_article_S_VIDE)
ensemble_S00 = set(liste_article_S00)


# Vérifier s'il y a des articles en commun
articles_communs_S20S30 = ensemble_S20 & ensemble_S30 
articles_communs_S20S40 = ensemble_S20 & ensemble_S40 
articles_communs_S20S00 = ensemble_S20 & ensemble_S00 
articles_communs_S20SVIDE = ensemble_S20 & ensemble_S_vide


# Afficher les résultats
if articles_communs_S20S30:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S20S30)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S20S40:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S20S40)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S20S00:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S20S00)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S20SVIDE:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S20SVIDE)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# In[68]:


# Convertir les listes en ensembles
ensemble_S10 = set(liste_article_S10)
ensemble_S20 = set(liste_article_S20)
ensemble_S30 = set(liste_article_S30)
ensemble_S40 = set(liste_article_S40)
ensemble_S_vide = set(liste_article_S_VIDE)
ensemble_S00 = set(liste_article_S00)


# Vérifier s'il y a des articles en commun
articles_communs_S30S40 = ensemble_S30 & ensemble_S40 
articles_communs_S30S00 = ensemble_S30 & ensemble_S00 
articles_communs_S30SVIDE = ensemble_S30 & ensemble_S_vide

# Afficher les résultats
if articles_communs_S30S40:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S30S40)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S30S00:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S30S00)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S30SVIDE:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S30SVIDE)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# In[69]:


# Convertir les listes en ensembles
ensemble_S10 = set(liste_article_S10)
ensemble_S20 = set(liste_article_S20)
ensemble_S30 = set(liste_article_S30)
ensemble_S40 = set(liste_article_S40)
ensemble_S_vide = set(liste_article_S_VIDE)
ensemble_S00 = set(liste_article_S00)


# Vérifier s'il y a des articles en commun
articles_communs_S40S00 = ensemble_S40 & ensemble_S00 
articles_communs_S40SVIDE = ensemble_S40 & ensemble_S_vide

# Afficher les résultats
if articles_communs_S40S00:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S40S00)
else:
    print("Les deux listes n'ont pas d'articles en commun.")
# Afficher les résultats
if articles_communs_S40SVIDE:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S40SVIDE)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# In[70]:


# Convertir les listes en ensembles
ensemble_S10 = set(liste_article_S10)
ensemble_S20 = set(liste_article_S20)
ensemble_S30 = set(liste_article_S30)
ensemble_S40 = set(liste_article_S40)
ensemble_S_vide = set(liste_article_S_VIDE)
ensemble_S00 = set(liste_article_S00)


# Vérifier s'il y a des articles en commun
articles_communs_S00SVIDE = ensemble_S00 & ensemble_S_vide
# Afficher les résultats
if articles_communs_S00SVIDE:
    print("Les deux listes ont des articles en commun:")
    print(articles_communs_S00SVIDE)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# pause 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Base Categorie clients :

# In[71]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM Dim_Catégorie_Client')
for row in cursor:
    print(row)


# In[72]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM Dim_Catégorie_Client', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\Dim_Catégorie_Client.csv', index=False)


# In[73]:


categorie_client=pd.read_csv('Dim_Catégorie_Client.csv')


# In[74]:


categorie_client


# **Clients** : les clients sont des personnes ou des entreprises qui achètent les produits ou les services d'une entreprise pour leur usage personnel ou professionnel. Ils peuvent être des clients directs ou indirects, selon qu'ils achètent directement auprès de l'entreprise ou via un revendeur, par exemple.
# 
# **Clients inter-sites** : les clients inter-sites sont des entités qui achètent des produits ou des services d'une entreprise pour les utiliser dans différentes filiales ou sites de leur propre organisation. Par exemple, une entreprise de fabrication de produits alimentaires peut vendre des ingrédients à une chaîne de restaurants ayant des succursales dans différents endroits.
# 
# **Clients société du groupe** : les clients société du groupe sont des filiales ou des entités d'une même société mère qui achètent des produits ou des services les unes des autres. Par exemple, une entreprise qui possède plusieurs filiales dans différents pays peut vendre des produits ou des services à une autre filiale qui en a besoin.

#  Les clients achètent des produits ou des services pour leur propre usage, tandis que les clients inter-sites et les clients société du groupe achètent des produits ou des services pour les utiliser dans différentes filiales ou entités de leur propre organisation.

# # Base Client :

# In[75]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM Dim_Client')
for row in cursor:
    print(row)


# In[76]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM Dim_Client', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\Dim_Client.csv', index=False)


# In[77]:


client=pd.read_csv('Dim_Client.csv')


# In[78]:


client


# on a 3 categories de clients , il faut vverifier et corriger les erreurs s'ils exisent 

# # Client : 

# In[79]:


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Catégorie'] == 'CL']

# Afficher les résultats
filtre_categorie_client


# nombres des clients de type client est : 961 

# # Clients inter-sites : CI

# In[80]:


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Catégorie'] == 'CI']

# Afficher les résultats
filtre_categorie_client


# On a 5 societe/client : S1001/S1002/S2080/S1006/S3007 , mais il faute verifier les autres site 
#     deux hyp à verifier : si les sites existent => correction de catégorie du client de CL à CI 
# 

# ## Site S10 : GDC , Grand Distributeur Cereale du MALI

# In[81]:


client_site_S10 =['S1003','S1004','S1005','S1009','S1007','S1008','S1010','S1011','S1020','S1030','S1040','S1050']
client_s10=client['Client']
ensemble_S10 = set(client_site_S10)
ensemble_client=set(client_s10)


# Vérifier s'il y a des clients en commun
client_commun_S10 = ensemble_S10 & ensemble_client
# Afficher les résultats
if client_commun_S10:
    print("Les deux listes ont des articles en commun:")
    print(client_commun_S10)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# In[82]:


# correction pour la societe S10 : Grand distributeur cereale du mali : GDC


# In[83]:


client['Client'].info()


# In[84]:


client_site_S10 =['S1003','S1004','S1005','S1009','S1007','S1008','S1010','S1011','S1020','S1030','S1040','S1050']


# Nouvelles valeurs pour la colonne "pays"
new_categorie_client = ['CI','CI','CI','CI','CI','CI','CI','CI','CI','CI','CI','CI']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
client.loc[client['Client'].isin(client_site_S10), 'Catégorie'] = new_categorie_client
client


# In[85]:


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Catégorie'] == 'CI']

# Afficher les résultats
filtre_categorie_client


# ## Site S20 : M3 

# In[86]:


client_site_S20 =['S2070','S2060','S2050','S2040','S2030','S2020','S2010','S2001']
client_s20=client['Client']
ensemble_S20 = set(client_site_S20)
ensemble_client=set(client_s20)


# Vérifier s'il y a des clients en commun
client_commun_S20 = ensemble_S20 & ensemble_client
# Afficher les résultats
if client_commun_S20:
    print("Les deux listes ont des articles en commun:")
    print(client_commun_S20)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# In[87]:


client_site_S20 =['S2070','S2060','S2050','S2040','S2030','S2020','S2010','S2001']


# Nouvelles valeurs pour la colonne "pays"
new_categorie_client = ['CI','CI','CI','CI','CI','CI','CI','CI']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
client.loc[client['Client'].isin(client_site_S20), 'Catégorie'] = new_categorie_client
client


# In[88]:


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Catégorie'] == 'CI']

# Afficher les résultats
filtre_categorie_client


# ## Chola Trading , CT2 S30

# In[89]:


client_site_S30 =['S3008','S3006','S3005','S3002','S3004','S3001','S3003']
client_s30=client['Client']
ensemble_S30 = set(client_site_S30)
ensemble_client=set(client_s30)


# Vérifier s'il y a des clients en commun
client_commun_S30 = ensemble_S30 & ensemble_client
# Afficher les résultats
if client_commun_S30:
    print("Les deux listes ont des articles en commun:")
    print(client_commun_S30)
else:
    print("Les deux listes n'ont pas d'articles en commun.")


# S3008 n'est pas un client inter-site

# In[90]:


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Client'] == 'S3008']

# Afficher les résultats
filtre_categorie_client


# S3008 n'est pas un client 

# In[91]:


client_site_S30 =['S3006','S3005','S3002','S3004','S3001','S3003']


# Nouvelles valeurs pour la colonne "pays"
new_categorie_client = ['CI','CI','CI','CI','CI','CI']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
client.loc[client['Client'].isin(client_site_S30), 'Catégorie'] = new_categorie_client
client

# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Catégorie'] == 'CI']

# Afficher les résultats
filtre_categorie_client


# ## CAI , 

# In[92]:


client_site_S40 =['S4020','S4002','S4001']
client_s40=client['Client']
ensemble_S40 = set(client_site_S40)
ensemble_client=set(client_s40)


# Vérifier s'il y a des clients en commun
client_commun_S40 = ensemble_S40 & ensemble_client
# Afficher les résultats
if client_commun_S40:
    print("Les deux listes ont des articles en commun:")
    print(client_commun_S40)
else:
    print("Les deux listes n'ont pas d'articles en commun.")

    


# S4020 n'est pas un client inter-site

# In[93]:


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Client'] == 'S4020']

# Afficher les résultats
filtre_categorie_client


# S4020 n'est pas un client 

# In[94]:


client_site_S40 =['S4002','S4001']


# Nouvelles valeurs pour la colonne "pays"
new_categorie_client = ['CI','CI']

# Boucle pour changer les valeurs dans la colonne "pays" en filtrant par l'ID de dimension spécifié
client.loc[client['Client'].isin(client_site_S40), 'Catégorie'] = new_categorie_client


# Filtre par categorie du client :
# CL
filtre_categorie_client = client.loc[client['Catégorie'] == 'CI']

# Afficher les résultats
filtre_categorie_client


# # Siege Groupe Keita , S00

# In[95]:


# Filtre par categorie du client :
# GROUPE KEITA 
filtre_categorie_client = client.loc[client['Client'] == 'S0101']

# Afficher les résultats
filtre_categorie_client


# le siege groupe Keita est un clinet société du groupe , mais dans la base client est noté de type client 

# # Clients société du groupe

# In[96]:


# Filtre par categorie du client :
# C
filtre_categorie_client = client.loc[client['Catégorie'] == 'CSG']

# Afficher les résultats
filtre_categorie_client


# le client CL29258 est un client de type client et non pas de type client société du groupe 

# pause

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[97]:


# Semaine 4 


# # base Representant Commercials 

# In[98]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM Dim_Representant_Commercial')
for row in cursor:
    print(row)


# In[99]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM Dim_Representant_Commercial ', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\Dim_Representant_Commercial.csv', index=False)


# In[100]:


R_Commercial=pd.read_csv('Dim_Representant_Commercial.csv')


# In[101]:


R_Commercial


# In[102]:


R_Commercial.isnull()


# In[103]:


R_Commercial['Code_Societe']


# In[104]:


R_Commercial['Code_Site_vente']


# In[105]:


R_Commercial.corr()


# les colonnes : Taux_commission1_1 jusqu'à Taux_commission2_99 sont tous vide , 
#     la colonne base_commission est aussi vide 
# les colonne site et société pour indique à chaque representant les sites ou bien les sociétés sont vides 
# 

# In[106]:


Colonne_commercial=["id_dim_Representant_Commercial","Code_Representant","Nom_Representant","Date_Creation"]
Representant_Commercial=R_Commercial[Colonne_commercial]


# In[107]:


Representant_Commercial


# On a 20 representants commercials : 
#     le premier avec Code_representant : inconnus est le plus important puisque dans la table factures ventes : 
#             plusque de 40500 factures avec cet representant commercial inconnu ( 87924 totales factures )
#     ainsi que le 16eme representant est defini par CT2 qu'est la refrence d'une site , avec seulement 2 factures dans la table 
#     Factures Ventes 
#     

# # TABLE CA PAR REPRESENTANT

# In[108]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM FACT_CA_PAR_REPRESENTANT')
for row in cursor:
    print(row)


# In[109]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM FACT_CA_PAR_REPRESENTANT ', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\FACT_CA_PAR_REPRESENTANT.csv', index=False)


# In[110]:


CA_Representant=pd.read_csv('FACT_CA_PAR_REPRESENTANT.csv')


# In[111]:


CA_Representant


# dans cette base en trouve que les code_sociétés et les codes sites sont definis 
# on peut donc grouper les representant commercials par leurs sites , sociétés et memes par groupe des clients 

# In[112]:


# Grouper les données par représentant, société, site et client, et compter les occurrences
resultats = CA_Representant.groupby('Code_Representant_1').agg({'Code_Societé': 'nunique', 'Code_Site': 'nunique', 'Code_Client': 'nunique'})

# Renommer les colonnes
resultats = resultats.rename(columns={'Code_Societé': 'nombre_societes', 'Code_Site': 'nombre_sites', 'Code_Client': 'nombre_clients'})

# Afficher les résultats
resultats


# On observe que le représentant "CT2" a travaillé avec 1 société, 1 site et a eu affaire à 1 client. Cela indique qu'il a une portée relativement limitée dans ses interactions.
# 
# Certains représentants, tels que "REP003" et "REP014", ont travaillé avec un nombre plus élevé de sociétés (2 et 4 respectivement) et de sites (10 et 16 respectivement), ce qui suggère une plus grande étendue géographique et une implication dans un plus grand nombre de lieux de travail.
# 
# En termes de clients, "REP003" a eu affaire à 124 clients différents, tandis que "REP006" n'a interagi qu'avec 8 clients. Cela met en évidence la variation significative dans le nombre de clients traités par chaque représentant.

# 
# 
# 
# 
# Le premier représentant dans le tableau ne possède pas d'identifiant (Code_Representant_1). Cela peut poser une problématique car l'absence d'identification spécifique peut rendre difficile le suivi et l'analyse des performances de ce représentant.
# 
# 
# 
# 
# 
# Problématique :
# 
# Comment évaluer et mesurer les performances du représentant sans identifiant de manière précise et fiable ?
# Comment suivre et comparer les résultats de ce représentant par rapport aux autres représentants ?
# 
# 
# 
# 
# Solutions possibles :
# 
# Attribution d'un identifiant unique : Il serait bénéfique d'attribuer un identifiant spécifique au représentant qui n'en a pas. Cela permettrait de le distinguer des autres représentants et faciliterait la collecte de données sur ses performances individuelles.
# 
# Suivi manuel des performances : Dans le cas où il n'est pas possible d'attribuer un identifiant, il pourrait être nécessaire de suivre manuellement les performances du représentant sans ID. Cela pourrait impliquer la création d'un système de suivi distinct, où les informations sur les transactions, les ventes et autres indicateurs clés seraient collectées de manière spécifique pour ce représentant.
# 
# Analyse qualitative : En l'absence de données quantitatives précises, il peut être utile de se concentrer sur une analyse qualitative du représentant. Cela pourrait impliquer l'évaluation de ses compétences en communication, sa relation avec les clients, son engagement dans le travail d'équipe, etc. Des entretiens et des observations directes pourraient être réalisés pour obtenir des informations plus détaillées.
# 
# Il est important de trouver une solution adaptée à la situation spécifique de ce représentant sans ID, afin de pouvoir évaluer ses performances de manière juste et équitable, et de prendre des décisions éclairées pour son développement professionnel.
# 
# 
# 
# 
# 
# 
# 

# In[113]:


resultats = CA_Representant.groupby(['Code_Representant_1','Code_Societé','Code_Site','Code_Client']).agg({'CA_HT':'sum'})
resultats


# In[114]:


# Obtenir la liste des représentants uniques
representants_uniques = CA_Representant['Code_Representant_1'].unique()

# Parcourir chaque représentant et afficher les données regroupées correspondantes
for representant in representants_uniques:
    representant_data = CA_Representant[CA_Representant['Code_Representant_1'] == representant]
    resultats = representant_data.groupby(['Code_Representant_1','Code_Societé','Code_Site','Code_Client']).agg({'CA_HT':'sum'})
    print("Représentant:", representant)
    print(resultats)
    print()  # Ajouter une ligne vide pour séparer les résultats de chaque représentant


# In[115]:



# Calculer le chiffre d'affaires total par représentant
ca_par_representant = CA_Representant.groupby('Code_Representant_1')['CA_HT'].sum()

# Trier les représentants par chiffre d'affaires décroissant
representants_tries = ca_par_representant.sort_values(ascending=False)

representants_tries_formates = representants_tries.map('{:,.2f}'.format)


# Afficher les représentants avec le chiffre d'affaires le plus élevé
print(representants_tries_formates)


# In[116]:


representants_tries_formates = representants_tries.map('{:,.2f}'.format) + '  (' + (representants_tries / representants_tries.sum() * 100).map('{:.2f}'.format) + '%)'

representants_tries_formates


# In[117]:


# Calculer le nombre de sites par représentant
sites_par_representant = CA_Representant.groupby('Code_Representant_1')['Code_Client'].nunique()

# Calculer le chiffre d'affaires total par représentant
ca_par_representant = CA_Representant.groupby('Code_Representant_1')['CA_HT'].sum()

# Créer un graphique à barres pour le nombre de sites par représentant
plt.figure(figsize=(10, 6))
plt.bar(sites_par_representant.index, sites_par_representant.values)
plt.xlabel('Représentant')
plt.ylabel('Nombre des clients ')
plt.title('Nombre des clients par représentant')
plt.xticks(rotation=45)
plt.show()

# Créer un graphique à barres pour le chiffre d'affaires par représentant
plt.figure(figsize=(10, 6))
plt.bar(ca_par_representant.index, ca_par_representant.values)
plt.xlabel('Représentant')
plt.ylabel('Chiffre d\'affaires')
plt.title('Chiffre d\'affaires par représentant')
plt.xticks(rotation=45)
plt.show()


# In[118]:


CA_Representant.corr()


# Les variables "Id_dimSociété" et "Id_dimSite" ont une forte corrélation positive de 0,969, ce qui indique qu'elles sont étroitement liées et évoluent généralement de manière similaire.
# 
# 
# La variable "id_dim_Representant_Commercial" a une corrélation faible avec les autres variables, avec des valeurs proches de zéro, ce qui indique qu'elle a peu d'influence sur les autres variables.
# 
# 
# Les variables "Qty_Facturée_US", "CA_HT" et "CA_TTC" ont une corrélation positive avec la variable "Année", bien que la corrélation soit faible, indiquant une légère tendance à augmenter ensemble.
# 
# 
# Les variables "CA_HT" et "CA_TTC" ont une forte corrélation positive de 1,000, ce qui est attendu car elles représentent le chiffre d'affaires calculé de différentes manières (HT et TTC).
# 
# 
# La variable "Année" a une corrélation positive de 0,126 avec la variable "Id_dimSociété", indiquant une légère relation entre les deux.
# 
# 
# La variable "Année" a une corrélation négative de -0,249 avec la variable "Mois", ce qui indique une relation inverse entre ces deux variables.
# 
# 
# La variable "Id_dimClient" a une corrélation positive de 0,101 avec la variable "Id_dimSociété", indiquant une certaine relation entre les clients et les sociétés.
# 

# In[119]:


# à suivre 


# In[ ]:





# # CA par famille d'article : 

# In[120]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM DTM_CA_PAR_Famille_Article')
for row in cursor:
    print(row)


# In[121]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM DTM_CA_PAR_Famille_Article', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\DTM_CA_PAR_Famille_Article.csv', index=False)


# In[122]:


CA_Famille_Article=pd.read_csv('DTM_CA_PAR_Famille_Article.csv')


# In[123]:


CA_Famille_Article


# In[124]:


CA_Famille_Article.isnull().any


# la base contient les information concernant la CA par famille d'article : 
# - CA qu'est caractériser par : Société / site / client / catégories de client / representant commercial / Qté / montant ...
# il faut etudier le chiffre d'affaire et analyser tous ces facteurs pour mieux comprendre les relations entre eux , 

# In[125]:


CA_Famille_Article.corr()


# In[126]:


# Liste pour stocker les noms des colonnes valides (sans valeurs NaN)
colonnes_valides = []

# Parcours des colonnes de la base de données
for colonne in CA_Famille_Article.columns:
    # Vérification si la colonne ne contient pas de valeurs NaN
    if not pd.isna(CA_Famille_Article[colonne]).any():
        # Ajout de la colonne à la liste des colonnes valides
        colonnes_valides.append(colonne)

# Filtrage de la base de données en utilisant les colonnes valides
CA_Famille_Article_filtré = CA_Famille_Article[colonnes_valides]

# Affichage de la base de données filtrée
print(CA_Famille_Article_filtré)


# In[127]:


colonne_supprime=['id_Famille_Statistique_Client3','id_Famille_Statistique_Article4','id_dim_Famille_Statistique_Article5','Montant_Brut']
CA_Famille_Article.drop(colonne_supprime, axis=1, inplace=True)


# In[128]:


import seaborn as sns

# Sélection des colonnes numériques
colonnes_numeriques = CA_Famille_Article.select_dtypes(include=['float64', 'int64'])

# Calcul de la corrélation en utilisant la méthode de Spearman
correlation = colonnes_numeriques.corr(method='spearman')


correlation


# Les variables "Id_dimSociété" et "Id_dimSite" sont fortement corrélées (0.907140), ce qui suggère une relation étroite entre ces deux variables.
# 
# 
# Les variables "id_dim_Catégorie_Client" et "Id_dimClient" ont une corrélation faible et positive (0.040095), indiquant une relation légèrement positive entre ces deux variables.
# 
# 
# Les variables liées aux familles statistiques des clients et des articles (comme "id_dim_Famille_Statistique_Client1", "id_Famille_Statistique_Client2", "id_dim_famille_stat_article1", etc.) montrent des corrélations positives entre elles, ce qui suggère une certaine cohérence dans la classification des clients et des articles.
# 
# 
# Les variables liées aux montants et aux quantités facturées (comme "Montant_HT", "Montant_TTC", "Qty_Facturée", "Qty_Facturée_US") sont fortement corrélées entre elles, ce qui est attendu puisqu'elles représentent différentes mesures financières liées aux transactions.
# 
# 
# La variable "id_dim_Representant_Commercial" (représentant le représentant commercial) présente une forte corrélation négative avec plusieurs autres variables, notamment les variables liées aux familles statistiques des clients et des articles. Cela pourrait indiquer une certaine diversité ou segmentation parmi les représentants commerciaux.

# In[129]:


# etude de variance 


# In[130]:


# Sélectionner les variables pertinentes pour l'analyse de variance
selected_variables = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

# Diviser les données en groupes en fonction d'une variable catégorielle (par exemple, id_dim_Catégorie_Client)
grouped_par_representant = CA_Famille_Article.groupby('Code_Representant_1')

# Calculer la variance pour chaque groupe
variances = grouped_par_representant[selected_variables].var()

# Afficher les résultats

variances


# les résultats que nous avons fournis pour les variances ne semblent pas très clairs car ils contiennent des valeurs très élevées et des valeurs NaN (non disponibles). Les valeurs élevées peuvent indiquer une grande dispersion des données, tandis que les valeurs NaN indiquent l'absence de données ou des calculs impossibles.
# 
# pour clarifier et améliorer l'analyse des variances dans votre base de données :
# 
# Assurez-vous que les valeurs dans les colonnes 'Montant_HT', 'Montant_TTC', 'Qty_Facturée' et 'Qty_Facturée_US' sont correctement formatées en tant que nombres. Vous pouvez utiliser la fonction astype de Pandas pour convertir les colonnes en nombres si nécessaire.
# 
# 

# In[131]:


# Convertir les colonnes en nombres
CA_Famille_Article['Montant_HT'] = pd.to_numeric(CA_Famille_Article['Montant_HT'], errors='coerce')
CA_Famille_Article['Montant_TTC'] = pd.to_numeric(CA_Famille_Article['Montant_TTC'], errors='coerce')
CA_Famille_Article['Qty_Facturée'] = pd.to_numeric(CA_Famille_Article['Qty_Facturée'], errors='coerce')
CA_Famille_Article['Qty_Facturée_US'] = pd.to_numeric(CA_Famille_Article['Qty_Facturée_US'], errors='coerce')


# Vérifiez les valeurs NaN dans vos données. Identifiez la raison pour laquelle ces valeurs sont manquantes et décidez de la façon dont vous souhaitez les gérer. Vous pouvez les supprimer, les remplacer par des valeurs par défaut ou les ignorer selon le contexte de votre analyse.
# 
# 

# In[132]:


# Supprimer les lignes contenant des valeurs manquantes
CA_Famille_Article = CA_Famille_Article.dropna()

# Remplacer les valeurs manquantes par une valeur par défaut
CA_Famille_Article = CA_Famille_Article.fillna(0)  # Remplacer NaN par 0, par exemple


# Échelonnez les valeurs des colonnes si elles diffèrent considérablement en magnitude. Par exemple, si les valeurs dans 'Montant_HT' sont beaucoup plus grandes que celles dans 'Qty_Facturée', cela peut fausser l'analyse de variance. Vous pouvez normaliser les données en les mettant à l'échelle sur une plage commune ou en utilisant des techniques telles que la normalisation z-score.
# 
# 

# In[133]:


from sklearn.preprocessing import MinMaxScaler

# Instancier un objet scaler
scaler = MinMaxScaler()

# Mettre à l'échelle les colonnes spécifiques
columns_to_scale = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']
CA_Famille_Article[columns_to_scale] = scaler.fit_transform(CA_Famille_Article[columns_to_scale])


# Utilisez des techniques de visualisation telles que des graphiques ou des diagrammes pour mieux comprendre les distributions des variables et identifier les valeurs aberrantes ou les schémas intéressants.
# 
# 

# In[134]:


import matplotlib.pyplot as plt

# Tracer un histogramme pour chaque colonne
CA_Famille_Article.hist(figsize=(30, 15))
plt.show()


# # Montant , QTE et Representant commercials

# Effectuez une analyse de variance plus détaillée en utilisant des outils statistiques appropriés, tels que les tests de signification, les comparaisons de groupes ou les modèles de régression. Cela peut vous aider à obtenir des informations plus approfondies sur les différences entre les groupes et à identifier les variables qui ont le plus d'impact sur la variance.

# In[135]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('Code_Representant_1')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# Dans le contexte de l'analyse de variance (ANOVA) que nous avons réalisée, la variance mesure la dispersion des données d'une variable autour de leur moyenne. Plus précisément, elle quantifie à quel point les valeurs individuelles d'une variable diffèrent de la moyenne de cette variable.
# 
# Dans notre cas, l'analyse de variance nous permet de déterminer s'il y a des différences significatives dans les valeurs des variables ("Montant_HT", "Montant_TTC", "Qty_Facturée", "Qty_Facturée_US") entre les différents groupes de représentants.
# 
# Lorsque nous obtenons une valeur de variance élevée, cela signifie que les données à l'intérieur de chaque groupe de représentants se dispersent considérablement par rapport à leur moyenne respective. En d'autres termes, les valeurs individuelles des variables peuvent varier de manière significative d'un représentant à l'autre au sein du même groupe.
# 
# La valeur de p (p-value) est utilisée pour déterminer si cette variation de la variance entre les groupes est statistiquement significative. Une valeur de p faible (généralement inférieure à 0,05) indique une différence significative dans les variances entre les groupes, ce qui suggère que les groupes diffèrent véritablement les uns des autres.
# 
# Dans notre cas, les valeurs de p très proches de zéro indiquent qu'il y a des différences significatives dans les variances des variables entre les groupes de représentants. Cela suggère que les groupes de représentants diffèrent réellement les uns des autres en termes de montants HT et TTC, ainsi que de quantités facturées et quantités facturées en USD.
# 
# En conclusion, l'analyse de variance montre que les groupes de représentants ont des variations significatives dans les valeurs des variables étudiées. Cela peut indiquer des différences de performance, de comportement ou d'autres facteurs entre les différents représentants.
# 
# 
# 
# 
# 

# In[136]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

# Grouper les données par représentant
grouped_data = CA_Famille_Article.groupby('Code_Representant_1')

for column in columns_to_analyze:
    print(f"Variable: {column}")
    data_groups = [group_data[column] for group_name, group_data in grouped_data if len(group_data) >= 2]
    if len(data_groups) >= 2:
        statistics, p_value = f_oneway(*data_groups)
        for group_name, group_data in grouped_data:
            if len(group_data) >= 2:
                print(f"Representant: {group_name}")
                print(f"Statistic: {statistics}")
                print(f"p-value: {p_value}\n")
    else:
        print("Insufficient data for analysis.\n")


# # les Montants HT et TTC , les QTE avec les catégories des clients 

# In[137]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('id_dim_Catégorie_Client')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# Dans cette analyse, nous examinons les variables "Montant_HT", "Montant_TTC", "Qty_Facturée" et "Qty_Facturée_US".
# 
# Pour les variables "Montant_HT" et "Montant_TTC", nous observons des statistiques de test élevées (14.13738311904145 et 14.13718782014182 respectivement) et des valeurs de p très proches de zéro (7.273196330945176e-07 et 7.274616216908051e-07 respectivement). Cela suggère qu'il y a des différences significatives dans les variances de ces variables entre les différentes catégories des clients .
# 
# En revanche, pour les variables "Qty_Facturée" et "Qty_Facturée_US", nous observons des statistiques de test très faibles (0.009388074488561389 et 0.010243587265825893 respectivement) et des valeurs de p élevées (0.9906558574232349 et 0.9898087013967928 respectivement). Cela indique qu'il n'y a pas de différences significatives dans les variances de ces variables entre les groupes de données.
# 
# En résumé, ces résultats suggèrent que les groupes de données diffèrent significativement en termes de variances pour les variables "Montant_HT" et "Montant_TTC", mais pas pour les variables "Qty_Facturée" et "Qty_Facturée_US". Cela peut indiquer des disparités importantes dans les montants facturés entre les groupes de données, tandis que les quantités facturées ne montrent pas de différences significatives.
# 
# 
# 
# 
# 

# # CLIENT ET CA 

# In[138]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('Id_dimClient')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# Les résultats des tests de variance montrent que les valeurs de p sont très proches de zéro pour toutes les variables analysées. Cela indique qu'il y a des différences significatives dans les variances des variables entre les clients.
# 
# 

# # MONTANT , QTE ET CATEGORIE ARTICLE

# In[139]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('id_dim_categorie_article')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# D'après les résultats obtenus, les tests de variance montrent des valeurs de p très proches de zéro pour toutes les variables analysées. Cela indique qu'il y a des différences significatives dans les variances des variables entre les catégories d'articles.
# 
# Cela suggère que les catégories d'articles diffèrent réellement les unes des autres en termes de montants HT et TTC, ainsi que de quantités facturées et quantités facturées en USD.
# 
# En conclusion, l'analyse de variance indique des variations significatives dans les valeurs des variables entre les catégories d'articles. Cela peut indiquer des différences dans les caractéristiques, la demande ou d'autres facteurs entre les différentes catégories d'articles.
# 
# 

# # MONTANT , QTE ET ARTICLE

# In[140]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('id_dim_Article')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# D'après les résultats obtenus, les tests de variance montrent des valeurs de p très proches de zéro pour toutes les variables analysées. Cela indique qu'il y a des différences significatives dans les variances des variables entre les articles et les autres factures.
# 
# 

# # SITE ET CA 

# In[141]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('Id_dimSite')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# Pour les variables "Montant_HT" et "Montant_TTC", les valeurs de p sont très proches de zéro, ce qui suggère qu'il y a des différences significatives dans les variances entre les sites. Cela indique que les sites diffèrent réellement les uns des autres en termes de montants HT et TTC.
# 
# En revanche, pour les variables "Qty_Facturée" et "Qty_Facturée_US", les valeurs de p sont relativement élevées (0.744 et 0.670 respectivement). Cela suggère qu'il n'y a pas de différences significatives dans les variances entre les sites pour ces variables. En d'autres termes, les sites ont des quantités facturées similaires, que ce soit en quantité facturée ou en quantité facturée en USD.
# 
# En conclusion, les tests de variance indiquent des différences significatives dans les montants HT et TTC entre les sites, mais pas dans les quantités facturées. Cela peut suggérer des variations dans les montants des transactions effectuées par les sites, tandis que les quantités facturées restent globalement similaires.

# # SOCIETES

# In[142]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('Id_dimSociété')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# Pour les variables "Montant_HT" et "Montant_TTC", les valeurs de p sont très proches de zéro, ce qui suggère qu'il y a des différences significatives dans les variances entre les sociétés. Cela indique que les sociétés diffèrent réellement les unes des autres en termes de montants HT et TTC.
# 
# Pour la variable "Qty_Facturée", la valeur de p est de 0.011, ce qui est inférieur au seuil de significativité communément utilisé de 0.05. Cela suggère qu'il existe une différence significative dans les variances entre les sociétés en ce qui concerne les quantités facturées.
# 
# De même, pour la variable "Qty_Facturée_US", la valeur de p est de 0.007, ce qui est inférieur au seuil de significativité de 0.05. Cela indique qu'il existe une différence significative dans les variances entre les sociétés en ce qui concerne les quantités facturées en USD.
# 
# En conclusion, les tests de variance montrent des différences significatives dans les montants HT et TTC ainsi que dans les quantités facturées et les quantités facturées en USD entre les sociétés. Cela suggère que les sociétés diffèrent réellement les unes des autres en termes de leurs transactions financières et de leurs quantités facturées.

# # DATAKEY

# In[143]:


from scipy.stats import f_oneway

# Effectuer un test de variance pour chaque colonne
columns_to_analyze = ['Montant_HT', 'Montant_TTC', 'Qty_Facturée', 'Qty_Facturée_US']

for column in columns_to_analyze:
    groups = CA_Famille_Article.groupby('DateKey')[column]
    statistics, p_value = f_oneway(*[group[1] for group in groups])
    print(f"Variable: {column}")
    print(f"Statistic: {statistics}")
    print(f"p-value: {p_value}\n")


# Les variables "Montant_HT" et "Montant_TTC" ont des valeurs de p très proches de zéro, ce qui suggère qu'il y a des différences significatives dans les variances entre les dates. Cela indique que les dates diffèrent réellement les unes des autres en termes de montants HT et TTC.
# 
# Pour la variable "Qty_Facturée", la valeur de p est de 2.11e-52, ce qui est inférieur au seuil de significativité de 0.05. Cela suggère qu'il existe une différence significative dans les variances entre les dates en ce qui concerne les quantités facturées.
# 
# De même, pour la variable "Qty_Facturée_US", la valeur de p est également très proche de zéro, ce qui indique une différence significative dans les variances entre les dates en ce qui concerne les quantités facturées en USD.
# 
# En conclusion, les tests de variance montrent des différences significatives dans les montants HT et TTC ainsi que dans les quantités facturées et les quantités facturées en USD entre les dates. Cela suggère que les dates diffèrent réellement les unes des autres en termes de transactions financières et de quantités facturées.

# In[ ]:





# Après avoir étudié la variance des variables dans votre ensemble de données, vous pouvez prendre plusieurs mesures pour analyser plus en détail les différences entre les groupes. Voici quelques suggestions :
# 
# Analyser les différences moyennes : Outre la variance, vous pouvez calculer les moyennes des variables pour chaque groupe et les comparer. Cela vous donnera des informations sur les différences moyennes entre les groupes de représentants, les catégories d'articles, les clients, les sites, etc.
# 
# Réaliser des tests statistiques supplémentaires : Vous pouvez effectuer des tests statistiques supplémentaires pour analyser les différences significatives entre les groupes. Par exemple, vous pouvez utiliser un test t-student pour comparer les moyennes entre deux groupes, ou une analyse de variance à deux facteurs pour étudier les interactions entre plusieurs variables.
# 
# Visualiser les données : Utilisez des graphiques et des visualisations pour représenter les différences entre les groupes de manière plus claire. Par exemple, vous pouvez créer des diagrammes en boîte et des graphiques en barres pour comparer les valeurs moyennes, ou des graphiques en dispersion pour montrer la distribution des données dans chaque groupe.
# 
# Effectuer des analyses supplémentaires : En fonction de votre objectif d'analyse, vous pouvez également effectuer d'autres analyses spécifiques. Par exemple, si vous souhaitez étudier les facteurs qui influencent les montants HT, vous pouvez réaliser une régression linéaire en utilisant d'autres variables explicatives telles que les catégories d'articles, les clients, etc.
# 
# Interpréter les résultats : Analysez attentivement les résultats obtenus à partir des différentes étapes d'analyse. Identifiez les tendances, les modèles et les différences significatives entre les groupes. Utilisez ces informations pour formuler des conclusions et des recommandations basées sur vos objectifs d'analyse.

# In[ ]:





# In[ ]:





# In[ ]:





# In[144]:


# semaine 5


# # CATEGORIES ARTICLES LES PLUS VENDUS 

# In[145]:


# Regrouper les données par article et calculer les quantités vendues
articles_quantites = CA_Famille_Article.groupby('Code_Categorie_Article')['Qty_Facturée'].sum()

# Trier les articles par quantités vendues de manière décroissante
top_articles = articles_quantites

# Afficher les articles les plus vendus
print(top_articles)


# # TOP 10

# In[146]:


# Regrouper les données par article et calculer les quantités vendues
articles_quantites = CA_Famille_Article.groupby('Code_Categorie_Article')['Qty_Facturée'].sum()

# Trier les articles par quantités vendues de manière décroissante
top_10_articles = articles_quantites.nlargest(10)

# Afficher les articles les plus vendus
print(top_10_articles)


# # TRANSACTIONS DES VENTES POUR LA CATEGORIE PALIM EN FONCTION DE TEMPS :  

# In[147]:


import matplotlib.pyplot as plt

# Sélectionner l'article souhaité
Categorie_article_code = 'PALIM'
Categorie_article_data = CA_Famille_Article[CA_Famille_Article['Code_Categorie_Article'] == Categorie_article_code]

# Convertir la colonne 'datekey' en format de date
Categorie_article_data['Date_comptable'] = pd.to_datetime(Categorie_article_data['Date_comptable'])

# Trier les données par date
Categorie_article_data = Categorie_article_data.sort_values('Date_comptable')


# Créer le graphique linéaire
plt.figure(figsize=(15,6))
plt.plot(Categorie_article_data['Date_comptable'], Categorie_article_data['Qty_Facturée'])
plt.title(f"Quantité facturée de l'article {Categorie_article_code}")
plt.xlabel('Date')
plt.ylabel('Quantité facturée')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[148]:


import matplotlib.pyplot as plt

# Sélectionner les catégories d'articles les plus vendues
top_categories = ['PALIM', 'SRVV', 'PRAL', 'PRAV', 'EMBAV', 'DIVR', 'MAPR', 'ALBV', 'PRCT2', 'CNCTV']
category_data = CA_Famille_Article[CA_Famille_Article['Code_Categorie_Article'].isin(top_categories)]

# Convertir la colonne 'Date_comptable' en format de date
category_data['Date_comptable'] = pd.to_datetime(category_data['Date_comptable'])

# Grouper les données par catégorie d'article et date comptable, puis calculer la somme des quantités facturées
grouped_data = category_data.groupby(['Code_Categorie_Article', 'Date_comptable'])['Qty_Facturée'].sum().reset_index()

# Créer le graphique linéaire
plt.figure(figsize=(10, 6))
for category in top_categories:
    category_subset = grouped_data[grouped_data['Code_Categorie_Article'] == category]
    plt.plot(category_subset['Date_comptable'], category_subset['Qty_Facturée'], label=category)

plt.title("Quantités facturées des catégories d'articles les plus vendues")
plt.xlabel('Date comptable')
plt.ylabel('Quantité facturée')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()


# In[149]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sélectionner les catégories d'articles les plus vendues
top_categories = ['PALIM', 'SRVV', 'PRAL', 'PRAV', 'EMBAV', 'DIVR', 'MAPR', 'ALBV', 'PRCT2', 'CNCTV']
category_data = CA_Famille_Article[CA_Famille_Article['Code_Categorie_Article'].isin(top_categories)]

# Convertir la colonne 'Date_comptable' en format de date
category_data['Date_comptable'] = pd.to_datetime(category_data['Date_comptable'])

# Grouper les données par catégorie d'article et date comptable, puis calculer la somme des quantités facturées
grouped_data = category_data.groupby(['Code_Categorie_Article', 'Date_comptable'])['Qty_Facturée'].sum().reset_index()

# Créer un graphique linéaire avancé avec seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(x='Date_comptable', y='Qty_Facturée', hue='Code_Categorie_Article', data=grouped_data)

plt.title("Quantités facturées des catégories d'articles les plus vendues")
plt.xlabel('Date comptable')
plt.ylabel('Quantité facturée')
plt.xticks(rotation=45)
plt.legend(title='Catégorie d\'article')
plt.grid(True)
plt.show()


# In[150]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sélectionner les catégories d'articles les plus vendues
top_categories = ['PALIM', 'SRVV', 'PRAL', 'PRAV', 'EMBAV', 'DIVR', 'MAPR', 'ALBV', 'PRCT2', 'CNCTV']
category_data = CA_Famille_Article[CA_Famille_Article['Code_Categorie_Article'].isin(top_categories)]

# Convertir la colonne 'Date_comptable' en format de date
category_data['Date_comptable'] = pd.to_datetime(category_data['Date_comptable'])

# Grouper les données par catégorie d'article et date comptable, puis calculer la somme des quantités facturées
grouped_data = category_data.groupby(['Code_Categorie_Article', 'Date_comptable'])['Qty_Facturée'].sum().reset_index()

# Créer une figure avec des sous-tracés
fig = make_subplots(rows=1, cols=1)

# Créer une trace pour chaque catégorie d'article
for category in top_categories:
    category_group = grouped_data[grouped_data['Code_Categorie_Article'] == category]
    fig.add_trace(go.Scatter(x=category_group['Date_comptable'], y=category_group['Qty_Facturée'], name=category), row=1, col=1)

# Configurer les boutons de sélection
buttons = []
for category in top_categories:
    visible = [False] * len(top_categories)
    visible[top_categories.index(category)] = True
    buttons.append(dict(label=category, method='update', args=[{'visible': visible}, {'title': category}]))

# Configurer les options de mise en page
fig.update_layout(updatemenus=[{'buttons': buttons, 'direction': 'down', 'showactive': True, 'x': 0.5, 'y': 1.2}])
fig.update_layout(title="Quantités facturées des catégories d'articles les plus vendues", xaxis_title='Date comptable', yaxis_title='Quantité facturée')

# Afficher le graphique interactif
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # ARTICLES LES PLUS VENDUS 

# In[151]:


# Regrouper les données par article et calculer les quantités vendues
articles_quantites = CA_Famille_Article.groupby('Code_Article')['Qty_Facturée'].sum()

# Trier les articles par quantités vendues de manière décroissante
top_10_articles = articles_quantites.nlargest(10)

# Afficher les articles les plus vendus
print(top_10_articles)


# In[152]:


import matplotlib.pyplot as plt

# Sélectionner l'article souhaité
article_code = 'PAL0030'
article_data = CA_Famille_Article[CA_Famille_Article['Code_Article'] == article_code]

# Convertir la colonne 'datekey' en format de date
article_data['Date_comptable'] = pd.to_datetime(article_data['Date_comptable'])

# Trier les données par date
article_data = article_data.sort_values('Date_comptable')

# Créer le graphique linéaire
plt.figure(figsize=(15,6))
plt.plot(article_data['Date_comptable'], article_data['Qty_Facturée'])
plt.title(f"Quantité facturée de l'article {article_code}")
plt.xlabel('Date')
plt.ylabel('Quantité facturée')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[153]:



import matplotlib.pyplot as plt

# Sélectionner l'article souhaité
article_code = 'SVV0034'
article_data = CA_Famille_Article[CA_Famille_Article['Code_Article'] == article_code]

# Convertir la colonne 'datekey' en format de date
article_data['Date_comptable'] = pd.to_datetime(article_data['Date_comptable'])

# Trier les données par date
article_data = article_data.sort_values('Date_comptable')

# Créer le graphique linéaire
plt.figure(figsize=(15,6))
plt.plot(article_data['Date_comptable'], article_data['Qty_Facturée'])
plt.title(f"Quantité facturée de l'article {article_code}")
plt.xlabel('Date')
plt.ylabel('Quantité facturée')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# # VENTE PAR SOCIETE PAR SITE ET PAR CATEGORIE 

# Pour étudier la vente des catégories d'articles par rapport aux sociétés et aux sites, ainsi que vérifier les catégories en commun entre les sites et les sociétés, vous pouvez suivre les étapes suivantes :
# 
# Commencez par filtrer vos données en fonction des sociétés et des sites que vous souhaitez analyser. Utilisez les colonnes "Code_Societé" et "Code_Site" pour filtrer les données correspondantes.
# 
# Ensuite, effectuez une analyse groupée en regroupant les données par catégorie d'articles, société et site. Calculez la somme des ventes ou des quantités vendues pour chaque combinaison catégorie d'articles, société et site. Cela vous donnera les chiffres de vente ou les quantités vendues pour chaque catégorie d'articles dans chaque société et site.
# 
# Utilisez des graphiques adaptés pour visualiser les données. Par exemple, vous pouvez utiliser un graphique à barres empilées pour représenter les ventes ou les quantités vendues par catégorie d'articles, en différenciant les couleurs pour chaque société et site. Cela vous permettra de voir les ventes totales pour chaque catégorie d'articles, ainsi que la contribution de chaque société et site à ces ventes.
# 
# Analysez les graphiques et identifiez les catégories d'articles qui se vendent le mieux dans chaque société et site. Vous pouvez également comparer les ventes entre les sociétés et les sites pour déterminer s'il y a des différences significatives.
# 
# Pour vérifier les catégories en commun entre les sites et les sociétés, vous pouvez utiliser des graphiques supplémentaires. Par exemple, vous pouvez créer un diagramme de Venn qui montre les catégories d'articles communes entre différents sites ou sociétés. Cela vous permettra de visualiser les chevauchements et les distinctions entre les catégories d'articles vendues dans chaque site ou société.

# In[154]:


import pandas as pd
import matplotlib.pyplot as plt

# Extraire la liste des sociétés et des sites à partir des colonnes correspondantes
societes = CA_Famille_Article['Code_Societé'].unique()
sites = CA_Famille_Article['Code_Site'].unique()

# Effectuer une analyse groupée pour calculer les ventes par catégorie, société et site
grouped_data = CA_Famille_Article.groupby(['Code_Categorie_Article', 'Code_Societé', 'Code_Site'])['Qty_Facturée'].sum().reset_index()

# Créer des graphiques pour visualiser les ventes par catégorie, société et site
fig, axes = plt.subplots(len(societes), len(sites), figsize=(12, 8))

for i, societe in enumerate(societes):
    for j, site in enumerate(sites):
        ax = axes[i][j]
        data = grouped_data[(grouped_data['Code_Societé'] == societe) & (grouped_data['Code_Site'] == site)]
        ax.bar(data['Code_Categorie_Article'], data['Qty_Facturée'])
        ax.set_xlabel('Catégorie')
        ax.set_ylabel('Quantité vendue')
        ax.set_title(f'Société: {societe}, Site: {site}')

# Vérifier les catégories en commun entre les sites et les sociétés
categories_communes = set(grouped_data['Code_Categorie_Article'])

for i, societe in enumerate(societes):
    for j, site in enumerate(sites):
        ax = axes[i][j]
        data = grouped_data[(grouped_data['Code_Societé'] == societe) & (grouped_data['Code_Site'] == site)]
        categories_site_societe = set(data['Code_Categorie_Article'])
        categories_non_communes = categories_site_societe.difference(categories_communes)
        ax.text(0.05, 0.9, f'Catégories non communes: {", ".join(categories_non_communes)}', transform=ax.transAxes)

# Afficher les graphiques
plt.tight_layout()
plt.show()


# In[155]:


import pandas as pd
import plotly.express as px



# Effectuer une analyse groupée pour calculer les ventes par catégorie, société et site
grouped_data = CA_Famille_Article.groupby(['Code_Categorie_Article', 'Code_Societé', 'Code_Site'])['Qty_Facturée'].sum().reset_index()

# Créer un graphique interactif avec Plotly
fig = px.bar(grouped_data, x='Code_Categorie_Article', y='Qty_Facturée', color='Code_Societé', facet_col='Code_Site',
             labels={'Code_Categorie_Article': 'Catégorie', 'Qty_Facturée': 'Quantité vendue'},
             title='Ventes par catégorie, société et site')

# Activer le mode de survol pour afficher les valeurs exactes
fig.update_traces(hovertemplate='Catégorie: %{x}<br>Société: %{color}<br>Site: %{facet_col}<br>Quantité vendue: %{y}')

# Afficher le graphique interactif
fig.show()


# In[156]:


import pandas as pd
import plotly.graph_objects as go



# Effectuer une analyse groupée pour calculer les ventes par catégorie, société et site
grouped_data_societe = CA_Famille_Article.groupby('Code_Societé')['Qty_Facturée'].sum().reset_index()
grouped_data_site = CA_Famille_Article.groupby('Code_Site')['Qty_Facturée'].sum().reset_index()
grouped_data_categorie = CA_Famille_Article.groupby('Code_Categorie_Article')['Qty_Facturée'].sum().reset_index()

# Créer des graphiques séparés pour chaque dimension
fig_societe = go.Figure(data=[go.Bar(x=grouped_data_societe['Code_Societé'], y=grouped_data_societe['Qty_Facturée'])])
fig_societe.update_layout(xaxis_title='Société', yaxis_title='Quantité vendue', title='Ventes par société')

fig_site = go.Figure(data=[go.Bar(x=grouped_data_site['Code_Site'], y=grouped_data_site['Qty_Facturée'])])
fig_site.update_layout(xaxis_title='Site', yaxis_title='Quantité vendue', title='Ventes par site')

fig_categorie = go.Figure(data=[go.Bar(x=grouped_data_categorie['Code_Categorie_Article'], y=grouped_data_categorie['Qty_Facturée'])])
fig_categorie.update_layout(xaxis_title='Catégorie', yaxis_title='Quantité vendue', title='Ventes par catégorie')

# Afficher les graphiques séparément
fig_societe.show()
fig_site.show()
fig_categorie.show()


# In[157]:


import pandas as pd
import plotly.graph_objects as go


# Effectuer une analyse groupée pour calculer les ventes par catégorie de client et catégorie d'article
grouped_data_client = CA_Famille_Article.groupby('Code_Categorie_Client')['Qty_Facturée'].sum().reset_index()
grouped_data_article = CA_Famille_Article.groupby('Code_Categorie_Article')['Qty_Facturée'].sum().reset_index()

# Créer un graphique en barres pour les ventes par catégorie de client
fig_client = go.Figure(data=[go.Bar(x=grouped_data_client['Code_Categorie_Client'], y=grouped_data_client['Qty_Facturée'])])
fig_client.update_layout(xaxis_title='Catégorie de client', yaxis_title='Quantité vendue', title='Ventes par catégorie de client')

# Créer un graphique en barres pour les ventes par catégorie d'article
fig_article = go.Figure(data=[go.Bar(x=grouped_data_article['Code_Categorie_Article'], y=grouped_data_article['Qty_Facturée'])])
fig_article.update_layout(xaxis_title='Catégorie d\'article', yaxis_title='Quantité vendue', title='Ventes par catégorie d\'article')

# Afficher les graphiques
fig_client.show()
fig_article.show()


# In[158]:


import pandas as pd
import plotly.graph_objects as go


# Effectuer une analyse groupée pour calculer les quantités totales achetées par client
grouped_data_client = CA_Famille_Article.groupby('Code_Client')['Qty_Facturée'].sum().reset_index()

# Trier les clients par quantité décroissante
top_clients = grouped_data_client.sort_values('Qty_Facturée', ascending=False).head(10)

# Filtrer les données pour inclure uniquement les articles achetés par les clients les plus importants
filtered_data = CA_Famille_Article[CA_Famille_Article['Code_Client'].isin(top_clients['Code_Client'])]

# Effectuer une analyse groupée pour obtenir les articles achetés par chaque client
grouped_data_article = filtered_data.groupby(['Code_Client', 'Code_Article'])['Qty_Facturée'].sum().reset_index()

# Créer un graphique en barres empilées pour les articles achetés par chaque client
fig = go.Figure()

for client in top_clients['Code_Client']:
    data = grouped_data_article[grouped_data_article['Code_Client'] == client]
    fig.add_trace(go.Bar(x=data['Code_Article'], y=data['Qty_Facturée'], name=client))

fig.update_layout(xaxis_title='Article', yaxis_title='Quantité vendue', title='Articles achetés par les clients les plus importants')
fig.show()


# In[159]:


import pandas as pd
import plotly.graph_objects as go


# Créer un graphique en ligne pour chaque client
fig = go.Figure()

# Effectuer une boucle sur chaque client
for client in CA_Famille_Article['Code_Client'].unique():
    data = CA_Famille_Article[CA_Famille_Article['Code_Client'] == client]
    fig.add_trace(go.Scatter(x=data['Date_comptable'], y=data['Qty_Facturée'], mode='lines', name=client))

# Personnaliser le layout du graphique
fig.update_layout(xaxis_title='Date comptable', yaxis_title='Quantité vendue', title='Transactions des articles achetés par client')

# Afficher le graphique
fig.show()


# In[160]:


import pandas as pd
import plotly.graph_objects as go

# Sélectionner le client spécifique
client = 'CL00162'
data = CA_Famille_Article[CA_Famille_Article['Code_Client'] == client]

# Obtenir la liste des articles achetés par le client
articles = data['Code_Article'].unique()

# Créer un graphique linéaire pour chaque article acheté
fig = go.Figure()
for article in articles:
    article_data = data[data['Code_Article'] == article]
    fig.add_trace(go.Scatter(x=article_data['Date_comptable'], y=article_data['Qty_Facturée'], mode='lines', name=article))

# Personnaliser le layout du graphique
fig.update_layout(xaxis_title='Date comptable', yaxis_title='Quantité vendue', title=f'Transactions des articles achetés par {client}')

# Afficher le graphique
fig.show()


# In[161]:


import pandas as pd
import plotly.graph_objects as go

# Sélectionner le client spécifique
client = 'CL02524'
data = CA_Famille_Article[CA_Famille_Article['Code_Client'] == client]

# Obtenir la liste des articles achetés par le client
articles = data['Code_Article'].unique()

# Créer un graphique linéaire pour chaque article acheté
fig = go.Figure()
for article in articles:
    article_data = data[data['Code_Article'] == article]
    fig.add_trace(go.Scatter(x=article_data['Date_comptable'], y=article_data['Qty_Facturée'], mode='lines', name=article))

# Personnaliser le layout du graphique
fig.update_layout(xaxis_title='Date comptable', yaxis_title='Quantité vendue', title=f'Transactions des articles achetés par {client}')

# Afficher le graphique
fig.show()


# In[162]:


# à suivre 


# In[ ]:





# In[ ]:





# # MOVEMENT SAISONNIER

# Pour étudier le mouvement saisonnier d'un article, vous pouvez suivre les étapes suivantes :
# 
# Collecte des données : Rassemblez les données historiques sur les ventes de l'article sur une période significative, de préférence sur plusieurs années. Assurez-vous d'inclure les informations de date pour chaque transaction.
# 
# Visualisation des données : Tracez un graphique de la quantité vendue de l'article en fonction du temps (mois, trimestre, etc.). Cela vous permettra de visualiser les fluctuations saisonnières éventuelles.
# 
# Décomposition saisonnière : Utilisez des techniques de décomposition saisonnière pour isoler les composantes saisonnières des données. L'une des méthodes couramment utilisées est la décomposition additive ou multiplicative de la série chronologique.
# 
# Analyse des tendances : Analysez les composantes saisonnières décomposées pour identifier les modèles saisonniers. Recherchez des schémas récurrents au cours de chaque année, tels que des pics de ventes à certaines périodes de l'année.
# 
# Prévisions saisonnières : Utilisez les modèles saisonniers identifiés pour effectuer des prévisions saisonnières. Cela vous permettra d'estimer les ventes futures de l'article en tenant compte des tendances saisonnières.
# 
# Analyse des facteurs saisonniers : Identifiez les facteurs qui influencent les variations saisonnières de l'article. Cela peut inclure des événements saisonniers spécifiques, des vacances, des promotions saisonnières, etc.
# 
# Interprétation des résultats : Analysez les résultats obtenus et tirez des conclusions sur le mouvement saisonnier de l'article. Identifiez les périodes de l'année où la demande est la plus élevée ou la plus faible, et évaluez l'impact des facteurs saisonniers sur les ventes.
# 
# L'étude du mouvement saisonnier d'un article vous permettra de mieux comprendre les variations saisonnières de la demande et de prendre des décisions éclairées en termes de gestion des stocks, de stratégies de marketing saisonnier et de planification des ressources.
# 
# 
# 
# 
# 

# In[163]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Filtrer les données pour l'article le plus vendu
article_plus_vendu = CA_Famille_Article[CA_Famille_Article['Code_Article'] == 'PAL0030']

# Vérifier le nombre d'observations
num_observations = article_plus_vendu.shape[0]
if num_observations < 24:
    print("Nombre d'observations insuffisant pour la décomposition saisonnière.")
else:
    # Convertir la colonne de dates en format de date
    article_plus_vendu['Date_comptable'] = pd.to_datetime(article_plus_vendu['Date_comptable'])

    # Définir la colonne de dates comme index du DataFrame
    article_plus_vendu.set_index('Date_comptable', inplace=True)

    # Effectuer la décomposition saisonnière
    decomposition = seasonal_decompose(article_plus_vendu['Qty_Facturée'], model='additive', period=12)

    # Tracer les composantes saisonnières décomposées
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observé')
    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Tendance')
    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Saisonnier')
    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Résiduel')

    # Afficher le graphique
    plt.tight_layout()
    plt.show()


# In[164]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Filtrer les données pour l'article le plus vendu
article_plus_vendu = CA_Famille_Article[CA_Famille_Article['Code_Article'] == 'PAL0030']

# Vérifier le nombre d'observations
num_observations = article_plus_vendu.shape[0]
if num_observations < 24:
    print("Nombre d'observations insuffisant pour la décomposition saisonnière.")
else:
    # Convertir la colonne de dates en format de date
    article_plus_vendu['Date_comptable'] = pd.to_datetime(article_plus_vendu['Date_comptable'])

    # Définir la colonne de dates comme index du DataFrame
    article_plus_vendu.set_index('Date_comptable', inplace=True)

    # Effectuer la décomposition saisonnière
    decomposition = seasonal_decompose(article_plus_vendu['Qty_Facturée'], model='additive', period=12)

    # Tracer les composantes saisonnières décomposées
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    
    # Style des lignes et couleurs
    ax1.plot(decomposition.observed, linestyle='-', linewidth=2, color='blue')
    ax2.plot(decomposition.trend, linestyle='--', linewidth=2, color='green')
    ax3.plot(decomposition.seasonal, linestyle=':', linewidth=2, color='red')
    ax4.plot(decomposition.resid, linestyle='-.', linewidth=2, color='orange')
    
    # Limites des axes y
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)
    
    # Grille
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    
    # Titres des sous-graphiques
    ax1.set_ylabel('Observé')
    ax2.set_ylabel('Tendance')
    ax3.set_ylabel('Saisonnier')
    ax4.set_ylabel('Résiduel')
    
    # Titre global du graphique
    plt.suptitle('Décomposition saisonnière de PAL0030')
    
    # Ajustement de la disposition des sous-graphiques
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Afficher le graphique
    plt.show()


# L'interprétation des résultats de la décomposition saisonnière peut fournir des informations utiles sur les tendances saisonnières d'un article. Voici comment interpréter les composantes du graphique :
# 
# Observé : Cette composante représente la série de données d'origine, c'est-à-dire les quantités facturées de l'article au fil du temps. Cette composante peut être utilisée pour visualiser la tendance générale de vente de l'article.
# 
# Tendance : La composante de tendance représente la variation à long terme des ventes de l'article. Elle met en évidence les changements de niveau moyen au fil du temps. Une tendance ascendante indique une augmentation des ventes, tandis qu'une tendance descendante indique une diminution des ventes. Une tendance horizontale suggère une stabilité dans les ventes.
# 
# Saisonnier : La composante saisonnière montre les variations périodiques régulières qui se répètent à intervalles fixes. Elle met en évidence les schémas saisonniers tels que des pics ou des creux récurrents dans les ventes de l'article. Par exemple, si vous analysez les ventes d'un article saisonnier comme des vêtements d'hiver, vous pouvez observer une augmentation marquée des ventes pendant les mois d'hiver et une baisse pendant les mois d'été.
# 
# Résiduel : La composante résiduelle représente les variations aléatoires non expliquées par la tendance et la composante saisonnière. Elle peut contenir du bruit, des erreurs de mesure ou d'autres facteurs imprévus. Si cette composante montre des motifs ou des structures significatifs, cela peut indiquer une information supplémentaire qui n'est pas capturée par les autres composantes.
# 
# L'interprétation des résultats dépendra du contexte spécifique de votre analyse. Vous pouvez utiliser ces informations pour prendre des décisions commerciales, comme ajuster les niveaux de stock, planifier des promotions saisonnières ou identifier les périodes de forte demande pour optimiser la gestion des stocks.
# 
# Il est important de noter que l'interprétation des résultats doit être basée sur une analyse approfondie et une compréhension du domaine commercial. La décomposition saisonnière fournit une visualisation utile des tendances, mais il est toujours recommandé de considérer d'autres facteurs et de valider les résultats avec d'autres données et analyses pour obtenir une image complète de la situation.
# 
# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# Pour segmenter les clients en fonction de leur comportement d'achat, vous pouvez utiliser différentes variables disponibles dans votre base de données. Voici quelques exemples de critères que vous pouvez utiliser :
# 
# Fréquence d'achat : Analysez la fréquence à laquelle chaque client effectue des achats. Regroupez les clients en segments tels que "acheteurs réguliers" (qui effectuent des achats fréquents) et "acheteurs occasionnels" (qui effectuent des achats moins fréquents).
# 
# Montant des achats : Étudiez le montant total des achats effectués par chaque client sur une période donnée. Créer des segments tels que "grands dépensiers" (qui effectuent des achats importants) et "petits dépensiers" (qui effectuent des achats de faible valeur).
# 
# Catégories de produits achetés : Analysez les catégories de produits ou les types d'articles achetés par chaque client. Vous pouvez créer des segments tels que « clients axés sur la mode », « clients axés sur l'électronique », « clients axés sur l'alimentation », etc.
# 
# Historique d'achat : Étudiez l'historique d'achat de chaque client pour identifier les tendances et les comportements récurrents. Regroupez les clients en segments tels que "clients fidèles" (qui ont une longue histoire d'achat) et "nouveaux clients" (qui ont récemment commencé à acheter).
# 
# Une fois que vous avez défini ces critères de segmentation, vous pouvez utiliser des techniques d'analyse de données ou des outils statistiques pour effectuer l'analyse et créer les segments de clients. Ces segments vous permettent de mieux comprendre les différents comportements d'achat et de personnalisation de vos actions marketing et commerciales en fonction des besoins et des préférences spécifiques de chaque segment.
# 
# Il est important de noter que la segmentation des clients basée sur le comportement d'achat peut être un processus itératif. Vous devrez régulièrement évaluer et ajuster vos segments en fonction de l'évolution des données et des objectifs de votre entreprise.
# 
# 
# 
# 
# 

# In[165]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sélectionner les colonnes pertinentes pour l'analyse de segmentation
data = CA_Famille_Article[['Code_Client', 'Montant_HT', 'Qty_Facturée']]

# Standardiser les données pour les mettre à l'échelle
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Montant_HT', 'Qty_Facturée']])

# Appliquer l'algorithme de clustering (par exemple, K-means) pour segmenter les clients
kmeans = KMeans(n_clusters=5)  # Spécifier le nombre de clusters souhaité
kmeans.fit(scaled_data)

# Ajouter les étiquettes de cluster aux données
data['Cluster'] = kmeans.labels_

# Afficher les statistiques des clusters
cluster_stats = data.groupby('Cluster').agg({
    'Montant_HT': ['mean', 'min', 'max'],
    'Qty_Facturée': ['mean', 'min', 'max'],
    'Code_Client': 'count'
})
print(cluster_stats)

# Visualiser les clusters
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Montant_Achat')
plt.ylabel('Qty_Achat')
plt.title('Segmentation des clients')
plt.show()


# In[166]:


import pandas as pd


# Calcul de la fréquence d'achat par client
frequency = CA_Famille_Article.groupby('Code_Client')['Date_comptable'].count()

# Calcul de la moyenne et de l'écart-type de la fréquence d'achat
mean_frequency = frequency.mean()
std_frequency = frequency.std()

# Définition du seuil en fonction de la moyenne et de l'écart-type
threshold = mean_frequency + 2 * std_frequency

# Segmentation des clients en acheteurs réguliers et occasionnels
segmentation = pd.Series(['Régulier' if f >= threshold else 'Occasionnel' for f in frequency], index=frequency.index)

# Ajout de la colonne de segmentation au DataFrame des clients
CA_Famille_Article['Segment'] = segmentation

# Affichage du résultat
print(CA_Famille_Article)


# In[167]:


CA_Famille_Article


# # SEMENTATION CLIENT EN FONCTION DE LEUR COMPORTEMENT D ACHAT 

# Pour segmenter les clients en fonction de leur comportement d'achat, vous pouvez utiliser différentes variables disponibles dans votre base de données. Voici quelques exemples de critères que vous pouvez utiliser :
# 
# Fréquence d'achat : Analysez la fréquence à laquelle chaque client effectue des achats. Regroupez les clients en segments tels que "acheteurs réguliers" (qui effectuent des achats fréquents) et "acheteurs occasionnels" (qui effectuent des achats moins fréquents).
# 
# Montant des achats : Étudiez le montant total des achats effectués par chaque client sur une période donnée. Créer des segments tels que "grands dépensiers" (qui effectuent des achats importants) et "petits dépensiers" (qui effectuent des achats de faible valeur).
# 
# Catégories de produits achetés : Analysez les catégories de produits ou les types d'articles achetés par chaque client. Vous pouvez créer des segments tels que « clients axés sur la mode », « clients axés sur l'électronique », « clients axés sur l'alimentation », etc.
# 
# Historique d'achat : Étudiez l'historique d'achat de chaque client pour identifier les tendances et les comportements récurrents. Regroupez les clients en segments tels que "clients fidèles" (qui ont une longue histoire d'achat) et "nouveaux clients" (qui ont récemment commencé à acheter).
# 
# Une fois que vous avez défini ces critères de segmentation, vous pouvez utiliser des techniques d'analyse de données ou des outils statistiques pour effectuer l'analyse et créer les segments de clients. Ces segments vous permettent de mieux comprendre les différents comportements d'achat et de personnalisation de vos actions marketing et commerciales en fonction des besoins et des préférences spécifiques de chaque segment.
# 
# Il est important de noter que la segmentation des clients basée sur le comportement d'achat peut être un processus itératif. Vous devrez régulièrement évaluer et ajuster vos segments en fonction de l'évolution des données et des objectifs de votre entreprise.

# In[168]:


cursor = conn.cursor()
cursor.execute('SELECT * FROM FACT_FACTURES_VENTES')
for row in cursor:
    print(row)


# In[169]:


import pandas as pd

# Exécuter la requête SQL et récupérer les données dans un DataFrame
df = pd.read_sql('SELECT * FROM FACT_FACTURES_VENTES', conn)

# Écrire les données dans un fichier CSV
df.to_csv(r'C:\Users\mehdi\Desktop\HYDATIS\keita pfe\FACT_FACTURES_VENTES.csv', index=False)


# In[170]:


facture_vente=pd.read_csv('FACT_FACTURES_VENTES.csv')


# In[171]:


facture_vente


# # Correction des catégories des clients et séparer chaque catégorie seul 

# # Client inter site 

# In[172]:


client_inter_site = ['S1003', 'S1004', 'S1005', 'S1009', 'S1007', 'S1008', 'S1010', 'S1011', 'S1020', 'S1030', 'S1040', 'S1050', 'S2070', 'S2060', 'S2050', 'S2040', 'S2030', 'S2020', 'S2010', 'S2001', 'S3008', 'S3006', 'S3005', 'S3002', 'S3004', 'S3001', 'S3003', 'S4020', 'S4002', 'S4001']

# Nouvelle valeur pour la colonne "Code_Categorie_Client"
new_categorie_client = 'CI'

# Modifier les valeurs dans la colonne "Code_Categorie_Client" en filtrant par l'ID de dimension spécifié
for client in client_inter_site:
    facture_vente.loc[facture_vente['Code_Client'] == client, 'Code_Categorie_Client'] = new_categorie_client

# Afficher le DataFrame facture_vente mis à jour
facture_vente


# # client de groupe siege 

# In[173]:


client_inter_site = ['S0101']

# Nouvelle valeur pour la colonne "Code_Categorie_Client"
new_categorie_client = 'CSG'

# Modifier les valeurs dans la colonne "Code_Categorie_Client" en filtrant par l'ID de dimension spécifié
for client in client_inter_site:
    facture_vente.loc[facture_vente['Code_Client'] == client, 'Code_Categorie_Client'] = new_categorie_client

# Afficher le DataFrame facture_vente mis à jour
facture_vente


# # modéfication de client CL29258 de type CSG au CL 

# In[174]:


client_inter_site = ['CL29258']

# Nouvelle valeur pour la colonne "Code_Categorie_Client"
new_categorie_client = 'CL'

# Modifier les valeurs dans la colonne "Code_Categorie_Client" en filtrant par l'ID de dimension spécifié
for client in client_inter_site:
    facture_vente.loc[facture_vente['Code_Client'] == client, 'Code_Categorie_Client'] = new_categorie_client

# Afficher le DataFrame facture_vente mis à jour
facture_vente


# In[175]:


# Filtrer la base en conservant uniquement les lignes avec "CL" dans la colonne "Code_Categorie_Client"
facture_vente_client = facture_vente.loc[facture_vente['Code_Categorie_Client'] == 'CL']

# Sauvegarder le DataFrame filtré dans un fichier CSV
facture_vente_client.to_csv('facture_vente_type_client.csv', index=False)


# In[176]:


vente_client=pd.read_csv('facture_vente_type_client.csv')


# In[177]:


vente_client


# # Definir les seuils de segmentation 

# Pour justifier les seuils de manière plus objective, vous pouvez utiliser des approches statistiques ou basées sur les données. Voici quelques méthodes couramment utilisées pour définir les seuils de segmentation :
# 
# Analyse statistique : Vous pouvez effectuer une analyse statistique des données d'achat pour identifier la distribution des valeurs. En utilisant des mesures telles que la moyenne, l'écart-type ou des quartiles, vous pouvez définir les seuils de manière objective. Par exemple, vous pouvez considérer que les clients ayant un nombre d'achats supérieur à la moyenne plus un certain nombre d'écart-types sont des acheteurs fréquents.
# 
# Méthode des quantiles : Vous pouvez diviser les données d'achat en quantiles pour créer des segments équilibrés. Par exemple, vous pouvez diviser les clients en quartiles en fonction du nombre d'achats ou du montant total des achats. Cela permet de créer des segments de taille égale, tels que les 25% des clients ayant le plus grand nombre d'achats.
# 
# Analyse des valeurs aberrantes : Vous pouvez identifier les valeurs aberrantes dans les données d'achat et les exclure de l'analyse. Cela peut aider à éviter que des clients avec des comportements d'achat atypiques ne faussent les seuils de segmentation.
# 
# Approche basée sur les objectifs : Vous pouvez définir les seuils en fonction des objectifs spécifiques de votre entreprise. Par exemple, si votre objectif est de cibler les clients à fort potentiel de revenus, vous pouvez définir des seuils plus élevés pour les acheteurs fréquents et les montants d'achat.
# 
# Il est important de noter que le choix des seuils dépend de la nature de votre activité, de vos objectifs de segmentation et de votre connaissance du domaine. Vous pouvez également utiliser des techniques d'apprentissage automatique telles que le clustering ou la classification pour identifier automatiquement les seuils optimaux à partir des données.
# 
# Une fois que vous avez défini les seuils, vous pouvez les utiliser dans le code pour segmenter les clients en fonction de la fréquence d'achat ou du montant des achats.

# In[178]:


# Calculer la moyenne et l'écart type pour le nombre d'achats
moyenne_achats = vente_client['Numéro_facture'].value_counts().mean()
ecart_type_achats = vente_client['Numéro_facture'].value_counts().std()
print('moyenne achats est ', moyenne_achats)
print('ecart_type achat est ', ecart_type_achats)
# Définir les seuils de segmentation pour le nombre d'achats
seuil_achats_inf = moyenne_achats - ecart_type_achats
seuil_achats_sup = moyenne_achats + ecart_type_achats
print('seuil achat inf est ', seuil_achats_inf)
print('seuil achat sup est ', seuil_achats_sup)
# Calculer la moyenne et l'écart type pour le montant total des achats
moyenne_montant = vente_client['Mt_Ligne_HT'].mean()
ecart_type_montant = vente_client['Mt_Ligne_HT'].std()
print('moyenne_montant',moyenne_montant)
print('ecart_type montant', ecart_type_montant)
# Définir les seuils de segmentation pour le montant total des achats
seuil_montant_inf = moyenne_montant - ecart_type_montant
seuil_montant_sup = moyenne_montant + ecart_type_montant
print('seuil montant inf' , seuil_montant_inf)
print('seuil montant sup ', seuil_montant_sup)



# La moyenne des achats est de 68.79, ce qui représente une estimation du nombre moyen d'achats par client. Cela peut servir de référence pour évaluer la fréquence d'achat des clients.
# 
# L'écart type des achats est de 1954.86, ce qui mesure la dispersion des données autour de la moyenne. Un écart type élevé indique une plus grande variabilité dans les fréquences d'achat des clients.
# 
# Le seuil inférieur d'achat est de -1886.07, ce qui signifie que les acheteurs dont la fréquence d'achat est inférieure à ce seuil peuvent être considérés comme des acheteurs occasionnels.
# 
# Le seuil supérieur d'achat est de 2023.65, ce qui indique que les acheteurs dont la fréquence d'achat est supérieure à ce seuil peuvent être considérés comme des acheteurs réguliers.
# 
# La moyenne des montants d'achat est de 8,688,134.21, ce qui représente une estimation du montant moyen des achats par client. Cela peut être utilisé pour évaluer le niveau de dépenses des clients.
# 
# L'écart type des montants d'achat est de 16,479,962.57, ce qui mesure la dispersion des données autour de la moyenne des montants d'achat. Un écart type élevé indique une plus grande variabilité dans les dépenses des clients.
# 
# Le seuil inférieur de montant d'achat est de -7,791,828.36, ce qui suggère que les acheteurs dont le montant d'achat est inférieur à ce seuil peuvent être considérés comme des acheteurs occasionnels.
# 
# Le seuil supérieur de montant d'achat est de 25,168,096.78, ce qui indique que les acheteurs dont le montant d'achat est supérieur à ce seuil peuvent être considérés comme des acheteurs réguliers.
# 
# 

# In[179]:


num_null=vente_client['Numéro_facture'].isnull().any()
num_null


# In[180]:


num_null=vente_client['Numéro_de_pièce'].isnull().any()
num_null 


# In[181]:


# Calculer la moyenne et l'écart type pour le nombre d'achats
moyenne_achats = vente_client['Numéro_de_pièce'].value_counts().mean()
ecart_type_achats = vente_client['Numéro_de_pièce'].value_counts().std()
print('moyenne achats est ', moyenne_achats)
print('ecart_type achat est ', ecart_type_achats)
# Définir les seuils de segmentation pour le nombre d'achats
seuil_achats_inf = moyenne_achats - ecart_type_achats
seuil_achats_sup = moyenne_achats + ecart_type_achats
print('seuil achat inf est ', seuil_achats_inf)
print('seuil achat sup est ', seuil_achats_sup)
# Calculer la moyenne et l'écart type pour le montant total des achats
moyenne_montant = vente_client['Mt_Ligne_HT'].mean()
ecart_type_montant = vente_client['Mt_Ligne_HT'].std()
print('moyenne_montant',moyenne_montant)
print('ecart_type montant', ecart_type_montant)
# Définir les seuils de segmentation pour le montant total des achats
seuil_montant_inf = moyenne_montant - ecart_type_montant
seuil_montant_sup = moyenne_montant + ecart_type_montant
print('seuil montant inf' , seuil_montant_inf)
print('seuil montant sup ', seuil_montant_sup)



# In[ ]:





# Si vous n'avez pas les informations sur le nombre d'achats des clients, vous devrez trouver une autre mesure pour segmenter vos clients en fonction de leur comportement d'achat. Voici quelques idées alternatives que vous pouvez considérer :
# 
# Montant total des achats : Vous pouvez utiliser le montant total des achats effectués par chaque client comme critère de segmentation. Par exemple, vous pouvez définir des seuils pour distinguer les clients à faible, moyen et haut montant d'achats.
# 
# Fréquence de commande : Si vous avez des informations sur les commandes passées par chaque client, vous pouvez calculer la fréquence de commande pour chaque client. Cela peut être mesuré en comptant le nombre de commandes effectuées sur une période donnée. Vous pouvez ensuite définir des seuils pour distinguer les clients à faible, moyen et haut niveau de fréquence de commande.
# 
# Dernière date d'achat : Vous pouvez segmenter les clients en fonction de la date de leur dernier achat. Par exemple, vous pouvez avoir des segments tels que "clients récents" pour ceux qui ont effectué un achat récemment, et "clients inactifs" pour ceux qui n'ont pas effectué d'achat depuis un certain temps.
# 
# Catégories de produits achetés : Si vous avez des informations sur les catégories de produits achetés par chaque client, vous pouvez segmenter les clients en fonction des catégories de produits les plus fréquemment achetées. Cela peut vous aider à identifier les segments de clients qui ont des préférences similaires en termes de produits.
# 
# L'approche de segmentation dépendra des données dont vous disposez et des objectifs spécifiques de votre analyse. Vous devrez choisir la mesure de segmentation la plus pertinente en fonction de votre cas d'utilisation.
# 
# 
# 
# 
# 

# # Segementation selon la fréquence d'achat

# Fréquence d'achat :
# 
# Acheteurs réguliers : les clients ayant une fréquence d'achat supérieure à la moyenne plus un certain nombre d'écart-types.
# Acheteurs occasionnels : les clients ayant une fréquence d'achat inférieure à la moyenne moins un certain nombre d'écart-types.

# In[182]:


import pandas as pd

# Supposons que vous avez un DataFrame appelé "table" contenant vos données
colonnes = vente_client.columns.tolist()

# Affichage des colonnes
print(colonnes)


# In[183]:


colonnes_segmentation = ['Code_Client', 'Numéro_facture', 'Mt_Ligne_HT', 'Date_comptable']
donnees_segmentation = vente_client[colonnes_segmentation]
donnees_segmentation


# # Segmentation selon nombre des transactions d'achats par client

# In[184]:


# Calcul du nombre d'achats par client
achats_par_client = donnees_segmentation.groupby('Code_Client')['Numéro_facture'].nunique()
achats_par_client


# In[185]:


achats_par_client.max()


# In[186]:


achats_par_client.describe()


# In[187]:


import matplotlib.pyplot as plt

# Plot histogram
plt.hist(achats_par_client, bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('Nombre d\'achats par client')
plt.ylabel('Fréquence')
plt.title('Distribution du nombre d\'achats par client')

# Show the plot
plt.show()
 


# In[188]:


import seaborn as sns
import matplotlib.pyplot as plt

# Créer une figure de taille personnalisée
plt.figure(figsize=(15, 6))

# Plot histogram using Seaborn
sns.histplot(achats_par_client, kde=True, color='skyblue')

# Add labels and title
plt.xlabel('Nombre d\'achats par client')
plt.ylabel('Fréquence')
plt.title('Distribution du nombre d\'achats par client')

# Customize the plot
sns.set(style='ticks', font_scale=1.2)
plt.grid(axis='y', linestyle='--')

# Show the plot
plt.show()


# In[189]:


moyenne=achats_par_client.mean()
moyenne


# In[190]:


ecart_type=achats_par_client.std()
ecart_type


# In[191]:


seuil_sup = moyenne + ecart_type
seuil_sup


# In[192]:


seuil_inf = moyenne - ecart_type
seuil_inf


# In[193]:


achats_par_client<seuil_sup


# In[194]:


# Calculer le nombre d'achats par client
achats_par_client = donnees_segmentation.groupby('Code_Client')['Numéro_facture'].transform('nunique')

# Ajouter la colonne "achats_par_client" à la table "donnees_segmentation"
donnees_segmentation['achats_par_client'] = achats_par_client

# Afficher la table "donnees_segmentation" mise à jour
print(donnees_segmentation)


# In[195]:


# Segmenter les clients en fonction du nombre d'achats par client et des seuils
donnees_segmentation['Segment_nbr_achat'] = 'Occasionnel'
donnees_segmentation.loc[donnees_segmentation['achats_par_client'] >= seuil_sup, 'Segment_nbr_achat'] = 'Régulier'

# Afficher la table "donnees_segmentation" avec la colonne "Segment" mise à jour
print(donnees_segmentation)


# In[196]:


import matplotlib.pyplot as plt

# Compter le nombre de clients dans chaque segment
segment_counts = donnees_segmentation['Segment_nbr_achat'].value_counts()

# Calculer le pourcentage de chaque segment
segment_percentages = segment_counts / len(donnees_segmentation) * 100

plt.figure(figsize=(10, 6))

# Créer le graphique à barres
plt.bar(segment_percentages.index, segment_percentages.values)

# Ajouter des étiquettes de pourcentage sur les barres
for i, count in enumerate(segment_counts):
    plt.text(i, segment_percentages[i], f"{count} ({segment_percentages[i]:.2f}%)",
             ha='center', va='bottom')

# Ajouter des titres et des étiquettes
plt.title("Segmentation des clients")
plt.xlabel("Segment")
plt.ylabel("Pourcentage de clients")

# Afficher le graphique
plt.show()


# # Segmentation selon la date comptable : 

# In[197]:


donnees_segmentation['Date_comptable'] = pd.to_datetime(donnees_segmentation['Date_comptable'])


# In[198]:


date_reference = donnees_segmentation['Date_comptable'].max()
date_reference


# In[199]:


donnees_segmentation['Jours_depuis_reference'] = (date_reference - donnees_segmentation['Date_comptable']).dt.days


# In[200]:


seuil_inf = -30  # Exemple de seuil inférieur (30 jours avant la date de référence)
seuil_sup = 30  # Exemple de seuil supérieur (30 jours après la date de référence)


# In[201]:


donnees_segmentation.loc[donnees_segmentation['Jours_depuis_reference'] < seuil_inf, 'Segment_nbr_jour'] = 'Ancien'
donnees_segmentation.loc[(donnees_segmentation['Jours_depuis_reference'] >= seuil_inf) & (donnees_segmentation['Jours_depuis_reference'] <= seuil_sup), 'Segment_nbr_jour'] = 'Récent'
donnees_segmentation.loc[donnees_segmentation['Jours_depuis_reference'] > seuil_sup, 'Segment_nbr_jour'] = 'Nouveau'


# In[202]:


import matplotlib.pyplot as plt
segment_counts = donnees_segmentation['Segment_nbr_jour'].value_counts()
segment_percentages = segment_counts / len(donnees_segmentation) * 100
print(segment_counts)
print(segment_percentages)

# Calculer le pourcentage de chaque segment
pourcentage_segment = donnees_segmentation['Segment_nbr_jour'].value_counts(normalize=True) * 100

# Afficher le graphique à barres
plt.figure(figsize=(8, 6))
plt.bar(pourcentage_segment.index, pourcentage_segment.values)
plt.xlabel('Segment')
plt.ylabel('Pourcentage')
plt.title('Répartition des segments des clients selon la date_comptable')
plt.show()


# In[ ]:





# In[ ]:





# # Segmentation des clients en fonction de montant total d'achats 

# In[203]:


# Calcul du montant total des achats par client
montant_total_achats = donnees_segmentation.groupby('Code_Client')['Mt_Ligne_HT'].sum()


# In[204]:


montant_total_achats.describe()


# In[205]:


import seaborn as sns
import matplotlib.pyplot as plt

# Créer une figure de taille personnalisée
plt.figure(figsize=(15, 6))

# Plot histogram using Seaborn
sns.histplot(montant_total_achats, kde=True, color='skyblue')

# Add labels and title
plt.xlabel('montant total achats par client')
plt.ylabel('Fréquence')
plt.title('Distribution du montant d\'achats par client')

# Customize the plot
sns.set(style='ticks', font_scale=1.2)
plt.grid(axis='y', linestyle='--')

# Show the plot
plt.show()


# In[206]:


moyenne_montant = montant_total_achats.mean()
moyenne_montant


# In[207]:


ecart_type_montant=montant_total_achats.std()
ecart_type_montant


# In[208]:


seuil_sup_montant=moyenne_montant+ecart_type_montant


# In[209]:


seuil_inf_montant=moyenne_montant-ecart_type_montant


# In[210]:


montant_total_achats


# In[211]:


# Créer un nouveau DataFrame avec la colonne "Code_Client" et "montant_total_achats"
montant_total_achats_df = pd.DataFrame({'Code_Client': montant_total_achats.index, 'montant_total_achats': montant_total_achats.values})

# Fusionner les deux DataFrames en utilisant la colonne "Code_Client" comme clé de fusion
donnees_segmentation = donnees_segmentation.merge(montant_total_achats_df, on='Code_Client', how='left')

# Afficher la table "donnees_segmentation" mise à jour
print(donnees_segmentation)


# In[212]:



# Ajouter une colonne "Segment" à la table "donnees_segmentation" pour la segmentation
donnees_segmentation['Segment_montant_total_achats'] = 'Inconnu'  # Valeur par défaut

# Segmenter les clients en fonction du montant total d'achat
donnees_segmentation.loc[donnees_segmentation['montant_total_achats'] >= seuil_sup_montant, 'Segment_montant_total_achats'] = 'Haut'
donnees_segmentation.loc[(donnees_segmentation['montant_total_achats'] >= seuil_inf_montant) & (donnees_segmentation['montant_total_achats'] < seuil_sup_montant), 'Segment_montant_total_achats'] = 'Moyen'
donnees_segmentation.loc[donnees_segmentation['montant_total_achats'] < seuil_inf_montant, 'Segment_montant_total_achats'] = 'Bas'

# Afficher les résultats de la segmentation
segments = donnees_segmentation['Segment_montant_total_achats'].value_counts()
pourcentages = segments / len(donnees_segmentation) * 100

print("Segmentation des clients :")
print(segments)
print("\nPourcentages :")
print(pourcentages)


# In[213]:


donnees_segmentation


# # PREDICTION DES VENTES 

# # PREDICTION DES VENTES : CATEGORIES ARTICLES / SOCIETE

# In[214]:


vente_client


# In[215]:


societes = vente_client["Code_Societé"].unique()  # Récupérer la liste des sociétés uniques

# Boucle pour filtrer et enregistrer chaque base de société
for societe in societes:
    vente_client_filtre = vente_client.query("Code_Societé == @societe")  # Filtrer la base en fonction de la société
    
    # Enregistrer la base filtrée dans un fichier CSV ou tout autre format souhaité
    nom_fichier = f"{societe}_data.csv"  # Nom du fichier basé sur la société
    vente_client_filtre.to_csv(nom_fichier, index=False)  # Enregistrer la base filtrée dans un fichier CSV
    
    # Vous pouvez également enregistrer la base filtrée dans une base de données si nécessaire
    # Assurez-vous d'adapter le code à votre système de gestion de base de données

    print(f"Données de la société {societe} enregistrées dans {nom_fichier}")


# In[216]:


vente_client_S20=pd.read_csv('S20_data.csv')


# In[217]:


vente_client_S20


# In[218]:


vente_client_S20 = vente_client_S20[['Date_comptable', 'Code_Societé', 'Code_Categorie_Article', 'Qté_facturée']]


# In[219]:


import matplotlib.pyplot as plt

# Créer un dictionnaire vide pour stocker les données de chaque catégorie d'article
donnees_categories = {}

# Parcourir toutes les catégories d'articles
for categorie in vente_client_S20['Code_Categorie_Article'].unique():
    # Sélectionner les données pour la catégorie actuelle
    donnees = vente_client_S20[vente_client_S20['Code_Categorie_Article'] == categorie]
    
    # Ajouter les données au dictionnaire
    donnees_categories[categorie] = donnees

# Créer un graphique linéaire pour chaque catégorie d'article
plt.figure(figsize=(10, 6))
for categorie, donnees in donnees_categories.items():
    plt.plot(donnees['Date_comptable'], donnees['Qté_facturée'], label=categorie)

# Ajouter des légendes et des titres
plt.legend()
plt.xlabel('Date Comptable')
plt.ylabel('Quantité')
plt.title('Ventes des Catégories d\'Articles au Fil du Temps')

# Afficher le graphique
plt.show()


# In[220]:


import pandas as pd

# Parcourir toutes les catégories d'articles
for categorie in vente_client_S20['Code_Categorie_Article'].unique():
    # Sélectionner les données pour la catégorie actuelle
    donnees = vente_client_S20[vente_client_S20['Code_Categorie_Article'] == categorie]
    
    # Enregistrer les données dans un fichier CSV pour la catégorie actuelle
    filename = categorie.replace(' ', '_') + '.csv'  # Nom du fichier CSV basé sur la catégorie
    donnees.to_csv(filename, index=False)


# In[221]:


print(f"Données de categories {donnees} enregistrées dans {filename}")


# In[222]:


prec_vente_client_S20=pd.read_csv('PREC.csv')


# In[223]:


prec_vente_client_S20


# In[224]:


palim_vente_client_S20=pd.read_csv('PALIM.csv')


# In[225]:


palim_vente_client_S20


# In[226]:



# Convertir la colonne 'Date_comptable' en format de date si nécessaire
palim_vente_client_S20['Date_comptable'] = pd.to_datetime(palim_vente_client_S20['Date_comptable'])

# Agréger les quantités en utilisant la somme pour chaque date
aggregated_data = palim_vente_client_S20.groupby('Date_comptable')['Qté_facturée'].sum()

# Afficher les données agrégées
print(aggregated_data)


# In[227]:


palim_S20=aggregated_data.to_csv('palim_s20.csv')


# In[228]:


palim_S20=pd.read_csv("palim_s20.csv")


# In[229]:


palim_S20


# In[230]:


# Afficher le graphique
plt.figure(figsize=(12, 6))
plt.plot(palim_S20['Date_comptable'], palim_S20['Qté_facturée'])
plt.title('Quantités facturées au fil du temps')
plt.xlabel('Date')
plt.ylabel('Quantité')
plt.xticks(rotation=45)
plt.show()


# In[231]:


# Paramètres de style pour Seaborn
sns.set(style='ticks', palette='Set2')

# Créer le graphique
plt.figure(figsize=(12, 6))
sns.lineplot(palim_S20['Date_comptable'], palim_S20['Qté_facturée'])
plt.title('Quantités facturées au fil du temps')
plt.xlabel('Date')
plt.ylabel('Quantité')

# Afficher la grille
plt.grid(True)

# Afficher le graphique
plt.show()


# In[232]:


palim_S20


# In[233]:


# Spécifier la date de séparation pour diviser les données
date_separation = '2022-01-01'

# Diviser les données en ensembles d'entraînement et de test
train_palim_S20 = palim_S20[palim_S20['Date_comptable'] < date_separation]
test_palim_S20 = palim_S20[palim_S20['Date_comptable'] >= date_separation]

# Enregistrer les ensembles d'entraînement et de test dans des fichiers CSV
train_palim_S20.to_csv('train_palim_S20.csv', index=False)
test_palim_S20.to_csv('test_palim_20.csv', index=False)


# In[234]:


train_palim_S20=pd.read_csv('train_palim_S20.csv')
train_palim_S20


# In[235]:


train_palim_S20.info()


# In[236]:


train_palim_S20.describe()


# In[237]:


test_palim_S20.describe()


# In[238]:


test_palim_S20.info()


# In[239]:


test_palim_S20


# In[240]:


test_palim_S20['Date_comptable'] = pd.to_datetime(test_palim_S20['Date_comptable'])


# In[241]:


test_palim_S20.info()


# In[242]:


train_palim_S20['Date_comptable'] = pd.to_datetime(train_palim_S20['Date_comptable'])


# In[243]:


train_palim_S20.info()


# # MODEL ARIMA

# In[244]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Création du modèle ARIMA
model = ARIMA(train_palim_S20['Qté_facturée'], order=(1, 0, 0))  # Exemple avec un ordre (p, d, q) = (1, 0, 0)

# Ajustement du modèle aux données d'entraînement
model_fit = model.fit()

# Prédictions sur les données de test
predictions = model_fit.predict(start=test_palim_S20.index[0], end=test_palim_S20.index[-1])

# Affichage des prédictions
plt.plot(test_palim_S20.index, test_palim_S20['Qté_facturée'], label='Données réelles')
plt.plot(predictions.index, predictions, label='Prédictions')
plt.xlabel('Date')
plt.ylabel('Quantité facturée')
plt.title('Prédictions ARIMA')
plt.legend()
plt.show()


# In[245]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculer les métriques d'évaluation
mse = mean_squared_error(test_palim_S20['Qté_facturée'], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_palim_S20['Qté_facturée'], predictions)
r2 = r2_score(test_palim_S20['Qté_facturée'], predictions)

# Afficher les métriques
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)


# In[246]:


vente=pd.read_csv('S20_data.csv')


# In[247]:


vente


# In[248]:


# Obtenir la liste des noms de colonnes
columns = vente.columns.tolist()

# Afficher toutes les colonnes
for column in columns:
    print(column)


# In[249]:


vente.corr()


# In[250]:


vente = vente.fillna(0)


# In[251]:


columns=['Id_dimSite','Date_comptable','id_dim_Article','Code_Fam_Stat_Article_1','Code_Fam_Stat_Article_2','Code_Fam_Stat_Article_3',
'id_dim_famille_stat_article1',
'id_dim_Famille_Statistique_Article2',
'id_dim_Famille_Statistique_Article3',
'id_dim_categorie_article',
'id_dim_Affaires',
'Mt_Ligne_HT',
'Qté_facturée']


# In[252]:


vente = vente.loc[:,columns]


# In[253]:


vente.corr()


# In[254]:


vente


# In[255]:


vente.info()


# In[256]:


vente['Date_comptable'] = pd.to_datetime(vente['Date_comptable'])


# In[257]:


vente['Code_Fam_Stat_Article_1'] = pd.factorize(vente['Code_Fam_Stat_Article_1'])[0]


# In[258]:


vente['Code_Fam_Stat_Article_2'] = pd.factorize(vente['Code_Fam_Stat_Article_2'])[0]


# In[259]:


vente['Code_Fam_Stat_Article_3'] = pd.factorize(vente['Code_Fam_Stat_Article_3'])[0]


# In[260]:


vente.info()


# In[261]:


vente.corr()


# In[262]:


# Groupement des données par catégorie d'article
groupes = vente.groupby('id_dim_categorie_article')

# Itération sur les groupes et enregistrement dans des fichiers CSV
for categorie, groupe in groupes:
    nom_fichier = f"categorie_{categorie}.csv"  # Nom du fichier CSV
    groupe.to_csv(nom_fichier, index=False)


# In[263]:


categorie_1=pd.read_csv("categorie_1.csv")


# In[264]:


categorie_1


# In[265]:


# Tri des données par date
categorie_1_sorted = categorie_1.sort_values('Date_comptable')

# Enregistrement des données triées dans un fichier CSV
categorie_1_sorted.to_csv('categorie_1_triees.csv', index=False)


# In[266]:


cat_1=pd.read_csv('categorie_1_triees.csv')
cat_1


# In[267]:


total_days = len(cat_1['Date_comptable'].unique())
total_days


# Modélisation des données non manquantes : Vous pouvez également choisir de ne modéliser que les données non manquantes et ignorer les jours manquants. Cela signifie que votre modèle ne fera pas de prédictions pour les jours manquants.
# 
# 

# In[268]:


# Supposons que vous ayez un DataFrame appelé df contenant vos données
# avec les colonnes Date_comptable, id_dim_Article et Qté_facturée

# Grouper les données par Date_comptable et id_dim_Article et calculer la somme des quantités
cat_1_grouped = cat_1.groupby(['Date_comptable', 'id_dim_Article'])['Qté_facturée'].sum().reset_index()

# Afficher le DataFrame regroupé
print(cat_1_grouped)


# In[269]:


group_columns = ['Date_comptable', 'id_dim_Article']

# Fusionner le DataFrame regroupé avec les autres colonnes du DataFrame d'origine
cat_1_grouped = pd.merge(cat_1_grouped, cat_1.drop('Qté_facturée', axis=1), on=group_columns, how='left')

# Afficher le DataFrame regroupé avec les autres colonnes
print(cat_1_grouped)


# In[270]:


cat_1


# In[271]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Supposons que vous ayez un DataFrame appelé df contenant vos données,
# avec les colonnes id_dim_Article, Code_Fam_Stat_Article_1, Code_Fam_Stat_Article_2, etc.

# Sélectionner les colonnes nécessaires pour la prédiction
features = ['id_dim_Article', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2','Code_Fam_Stat_Article_3',
           'id_dim_famille_stat_article1','id_dim_Famille_Statistique_Article2','id_dim_Famille_Statistique_Article3',
            'id_dim_Affaires','Mt_Ligne_HT']
target = 'Qté_facturée'

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(cat_1[features], cat_1[target], test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print('Erreur quadratique moyenne :', mse)


# In[272]:


moyenne_qte=cat_1['Qté_facturée'].mean()
moyenne_qte


# In[273]:


data_selected


# In[274]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Préparation des données
train_X = np.array(data_selected)  # Variables exogènes d'entraînement
train_y = np.array(data['Qté_facturée'])  # Quantités d'articles d'entraînement

# Création du modèle de réseau de neurones
model = Sequential()
model.add(Dense(32, input_dim=train_X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compilation du modèle
model.compile(loss='mean_squared_error', optimizer='adam')

# Entraînement du modèle
model.fit(train_X, train_y, epochs=100, batch_size=32, verbose=0)

# Prédictions sur l'ensemble de test
test_X = np.array(data_selected)  # Variables exogènes de test
predictions = model.predict(test_X)

# Affichage des prédictions
for i in range(len(test_X)):
    print(f"Prédiction pour l'exemple {i+1}: {predictions[i]}")


# In[ ]:


# Calcul de l'erreur quadratique moyenne (RMSE)
mse = ((predictions - test_X) ** 2).mean()
rmse = np.sqrt(mse)
print("RMSE:", rmse)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Préparation des données
features = ['id_dim_Article', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'id_dim_famille_stat_article1', 'id_dim_Famille_Statistique_Article2', 'id_dim_Famille_Statistique_Article3', 'id_dim_Affaires', 'Mt_Ligne_HT']
X = data[features]
y = data['Qté_facturée']

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


# In[ ]:


import matplotlib.pyplot as plt

# Tracer les prédictions et les valeurs réelles
plt.scatter(range(len(y_test)), y_test.values, color='blue', label='Quantité réelle')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Quantité prédite')

# Ajouter les légendes et les titres
plt.xlabel('Index')
plt.ylabel('Quantité')
plt.title('Prédictions vs Valeurs réelles')
plt.legend()

# Afficher le graphique
plt.show()


# In[ ]:


print('Quantité prédite:', y_pred)
print('Quantité réelle:', y_test)
print(y_test-y_pred)
print(((y_test-y_pred)/y_test)*100)


# In[ ]:


# correction : 


# # S20

# In[277]:


vente_S20=pd.read_csv("S20_data.csv")


# # changement les ID par des codes et noms 

# In[278]:


columns=['Code_Site','Date_comptable','Code_Article','Code_Fam_Stat_Article_1','Code_Fam_Stat_Article_2','Code_Fam_Stat_Article_3',
'Code_Categorie_Article',
'Affaire',
'Mt_Ligne_HT',
'Qté_facturée']


# In[279]:


vente_S20 = vente_S20.loc[:,columns]


# In[280]:


vente_S20


# In[281]:


vente_S20['Code_Fam_Stat_Article_1'] = pd.factorize(vente_S20['Code_Fam_Stat_Article_1'])[0]
vente_S20['Date_comptable'] = pd.to_datetime(vente_S20['Date_comptable'])
vente_S20['Code_Fam_Stat_Article_2'] = pd.factorize(vente_S20['Code_Fam_Stat_Article_2'])[0]
vente_S20['Code_Fam_Stat_Article_3'] = pd.factorize(vente_S20['Code_Fam_Stat_Article_3'])[0]


# In[282]:


# Groupement des données par catégorie d'article
groupes = vente_S20.groupby('Code_Categorie_Article')

# Itération sur les groupes et enregistrement dans des fichiers CSV
for categorie, groupe in groupes:
    nom_fichier = f"categorie_S20{categorie}.csv"  # Nom du fichier CSV
    groupe.to_csv(nom_fichier, index=False)


# # S20 PALIM

# In[341]:


Palim=pd.read_csv("categorie_S20PALIM.csv")


# In[342]:


Palim


# In[343]:


Palim.info()


# In[344]:


Palim['Date_comptable'] = pd.to_datetime(Palim['Date_comptable'])


# In[345]:


Palim.info()


# In[346]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Code_Site'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Code_Site'] = Palim['Code_Site'].map(mapping)

# Vérifier les modifications
print(Palim['Code_Site'])


# In[347]:


valeurs_uniques = Palim['Code_Site'].unique()
valeurs_uniques


# In[348]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Code_Article'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Code_Article'] = Palim['Code_Article'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[349]:


Palim


# In[350]:


colonn='Code_Categorie_Article'
Palim=Palim.drop(colonn,axis=1)


# In[351]:


Palim


# In[352]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = Palim['Affaire'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
Palim['Affaire'] = Palim['Affaire'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[295]:


Palim


# In[353]:


Palim_1 = Palim.sort_values('Date_comptable')


# In[354]:


Palim_1.to_csv('Palim_Date.csv')


# In[355]:


palim=pd.read_csv('Palim_Date.csv')


# In[356]:


palim


# In[300]:


col='Unnamed: 0'


# In[301]:


palim=palim.drop(col,axis=1)


# In[302]:


palim


# In[303]:


palim.info()


# In[357]:


palim['Date_comptable'] = pd.to_datetime(palim['Date_comptable'])
palim.info()


# In[305]:


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


# In[306]:


residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Valeurs réelles")
plt.ylabel("Résidus")
plt.title("Graphique de résidus")
plt.show()


# In[307]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison entre les valeurs réelles et les valeurs prédites")
plt.show()


# In[ ]:





# In[308]:


from sklearn.model_selection import cross_val_score

# Créer le modèle des forêts aléatoires
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Effectuer une validation croisée avec 5 plis (folds)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculer le score de RMSE moyen
rmse_scores = (-scores) ** 0.5
mean_rmse = rmse_scores.mean()

# Afficher les scores de RMSE pour chaque pli et le score moyen
print("Scores RMSE pour chaque pli:")
print(rmse_scores)
print("RMSE moyen:", mean_rmse)


# In[310]:


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


# In[311]:


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
model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
print('Quantité prédite:', y_pred)


# In[312]:


from sklearn.metrics import r2_score


# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Afficher le coefficient de détermination (R²)
print("Coefficient de détermination (R²) :", r2)


# In[359]:


import pandas as pd

# Convertir la colonne 'Date' en format de date
palim['Date_comptable'] = pd.to_datetime(palim['Date_comptable'])

# Extraire l'année, le mois, le jour et le jour de la semaine
palim['Annee'] = palim['Date_comptable'].dt.year
palim['Mois'] = palim['Date_comptable'].dt.month
palim['Jour'] = palim['Date_comptable'].dt.day
palim['JourSemaine'] = palim['Date_comptable'].dt.day_name()

# Afficher le DataFrame avec les nouvelles colonnes
print(palim[['Date_comptable', 'Annee', 'Mois', 'Jour', 'JourSemaine']])


# In[360]:


palim


# In[361]:


# Obtenir toutes les valeurs uniques de la colonne "Code_Site"
valeurs_uniques = palim['JourSemaine'].unique()

# Créer un dictionnaire de mapping pour associer chaque valeur unique à un numéro
mapping = {valeur: index+1 for index, valeur in enumerate(valeurs_uniques)}

# Remplacer les valeurs de la colonne par les numéros correspondants
palim['JourSemaine'] = palim['JourSemaine'].map(mapping)

# Vérifier les modifications
valeurs_uniques


# In[362]:


palim


# In[317]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Préparation des données
features = ['Code_Article', 'Code_Site', 'Code_Fam_Stat_Article_1', 'Code_Fam_Stat_Article_2', 'Code_Fam_Stat_Article_3', 'Affaire', 'Mt_Ligne_HT',
           'Annee','Mois','Jour','JourSemaine']
X = palim[features]
y = palim['Qté_facturée']


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle des forêts aléatoires
model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
print('Quantité prédite:', y_pred)


# In[318]:


from sklearn.metrics import r2_score


# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer le coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Afficher le coefficient de détermination (R²)
print("Coefficient de détermination (R²) :", r2)


# In[364]:


from sklearn.ensemble import RandomForestRegressor
articles_uniques = palim['Code_Article'].unique()

for article in articles_uniques:
    # Filtrer les données pour l'article spécifique
    data_article = palim[palim['Code_Article'] == article]
    
    # Préparation des données
    X = data_article[features]
    y = data_article['Qté_facturée']
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Créer et entraîner le modèle des forêts aléatoires
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Faire des prédictions pour l'article spécifique
    y_pred = model.predict(X_test)
    
    # Évaluer le modèle pour l'article spécifique
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Afficher les résultats pour l'article spécifique
    print(f"Article: {article}")
    print("RMSE:", rmse)
    print("R²:", r2)


# In[321]:


from sklearn.model_selection import cross_val_score

# Préparation des données
X = palim[features]
y = palim['Qté_facturée']

# Créer le modèle des forêts aléatoires
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Calculer le coefficient de détermination (R²) avec validation croisée k-fold
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Afficher les scores pour chaque pli
for fold, score in enumerate(cv_scores, 1):
    print(f"Fold {fold}: R² = {score}")

# Afficher le score moyen
print("Score moyen R²:", cv_scores.mean())


# In[322]:


tolerance = 100  # Tolérance ou marge d'erreur

# Convertir les valeurs prédites et les valeurs réelles en classes binaires
y_pred_binary = abs(y_pred - y_test) <= tolerance

# Calculer l'accuracy
accuracy = sum(y_pred_binary) / len(y_pred_binary)

# Afficher l'accuracy
print("Accuracy:", accuracy)


# In[333]:


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


# # GARDIEN BOOSTING REGRESSOR
