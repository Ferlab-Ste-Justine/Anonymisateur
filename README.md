# Anonymisateur
Projet pour anonymisation de texte avec pipeline NER.

# Pipeline de Reconnaissance d'Entités Nommées (NER) pour l'Anonymisation de Texte

## Vue d'ensemble
Ce projet met en œuvre un pipeline de reconnaissance d'entités nommées (NER) pour l'anonymisation de texte, en utilisant des bibliothèques Python telles que `transformers`, `pyspark` et `nltk`. Il inclut des fonctionnalités de prétraitement des données, d'application de modèles NER, de filtrage par Blacklist et de remplacement de noms.

L'objectif principal est d'anonymiser les informations sensibles(LES NOMS) dans des données textuelles en détectant et en remplaçant les entités nommées à l'aide d'un pipeline configurable.
Le défi est d'identifier le moins de vocabulaire médicale, à la syntaxe similaire. 

---

## Fonctionnalités
- **Pipeline NER** : Applique un modèle NER basé sur des transformeurs pour extraire les entités nommées.
- **Filtrage par liste noire** : Filtre des termes spécifiques basés sur une liste noire personnalisable.
- **Remplacement de noms** : Remplace les entités détectées par des noms alternatifs issus d'une base de données.
- **Traitement à grande échelle** : Utilise PySpark pour traiter efficacement de grands ensembles de données.

---

## Structure du Dossier

Voici une explication de la structure du dossier où se trouve ce projet :

### **1. Notebook principal**
Le fichier **`Anonymisation.ipynb`**  est le **fichier principal** du projet.  
C'est ici que vous pouvez exécuter le pipeline étape par étape, personnaliser les paramètres et tester les fonctionnalités.

### **2. Dossiers annexes**
Certains dossiers contiennent des fichiers nécessaires au fonctionnement du pipeline, mais **ne doivent pas être modifiés manuellement**. Voici leur contenu :

- **`Anonymisation/bert-large-NER/`** :
  - Contient les fichiers nécessaire au réseau neuronale pour le NER. Ce fichier est chargé automatiquement dans le pipeline. **NE pas modifiez**
  - **À Modifiez** le fichier **bert-large-NER/model_safetensors.txt** doit être remplacer par le vrai fichier(trop lourd pour github) **bert-large-NER/model.safetensors**, disponible ici: https://msss365-my.sharepoint.com/:u:/r/personal/alexandre_cantin_hsj_ssss_gouv_qc_ca/Documents/R%C3%A9seauNeuronale%20Anonymisation/bert-large-NER/model.safetensors?csf=1&web=1&e=sZO7xW

- **`Anonymisation/blacklist/`** :
  - Contient le fichier `blacklist.pkl`, Ce fichier est chargé automatiquement dans le pipeline. **Modifiez si nécessaire**

- **`Anonymisation/name_database/`** :
  - Contient les bases de données de noms utilisées pour le remplacement des entités détectées. Ce fichier est chargé automatiquement dans le pipeline
  - Ces fichiers servent au remplacement automatique des entités et ne nécessitent pas de modification directe.**Modifiez si nécessaire**

### **3. Dossier Résultant**
- **`output.csv`** :
  - le pipeline génère ce fichier à la sortie.
  - Vous pouvez les supprimer après exécution si vous ne souhaitez pas conserver les sorties.

---

## Installation
Assurez-vous d'avoir les dépendances suivantes installées :

- Python 3.7+
- Bibliothèques requises : `transformers`, `pyspark`, `nltk`, `tqdm`, `numpy`, `pandas`, `deep_translator`,

Installez les dépendances avec :
```bash
pip install transformers pyspark nltk tqdm numpy pandas deep-translator comet-ml
```

---

## Structure du projet
### Composants clés :
1. **Traduction** :
   - Traduit les textes français en anglais pour le **Pipeline NER**.
   
2. **Pipeline NER** :
   - `NER_text_split_v2` : Applique le NER sur un texte divisé en phrases.
   - `fix_output` : Ajuste les intervalles et les étiquettes d'entités pour garantir la cohérence.

3. **Blacklist** :
   - Charge une liste noire prédéfinie (`blacklist.pkl`) et filtre les étiquettes.

4. **Remplacement de noms** :
   - Remplace dynamiquement les noms des entités à l'aide d'une base de données configurable (`load_name_database`).

5. **Packaging du modèle** :
   - `model_packaged_v2` : Fonction principale qui intègre la traduction, la transformation de texte et le NER pour traiter le texte.

6. **Application du modèle** :
   - `apply_model` : Étend le processus d'anonymisation à un grand ensemble de données en utilisant Spark.
   
7. **Chargement des données** :
   - Chargez les données dans un DataFrame Spark pour le traitement.
   - Prend en charge les formats `.parquet` et `.csv`.

---


## Le Modèle
***apply_model(model_packaged_v2, df, ner_pipeline, blacklist, name_database, index = 'index', observation_value = 'observation_value', name = 'name')***

- **model_packaged_v2** (function) : La fonction d'application.
- **df** (pyspark df) : dataframe contenat les textes à anonymiser(voir spécificité plus bas).
- **ner_pipeline** (objet Hugging face)** : Le modèle Bert utilisé pour le NER
- **blacklist** (list de str) : Un identifiant unique pour chaque ligne.
- **name_database** (list de str)** : Liste de noms de remplacements.
- **`index`** (*str, optionnel*) : le nom de la colonne avec les index.
- **`observation_value`** (*str, optionnel*) : le nom de la colonne avec les textes à anonymiser
- **`name`** (*str, optionnel*) : le nom de la colonne avec les noms déjà connu présent dans le texte.

---


## Format des données d'entrée ***df***
Le DataFrame PySpark d'entrée doit contenir les colonnes suivantes:

- **index** (int) : Un identifiant unique pour chaque ligne.
- **observation_value** (str) : Le texte d'entrée à anonymiser.
- **name** (list de str, optionnel)** : Liste de noms déjà connu et associés à l'observation (facultatif).

***le nom des colonnes peut être passé comme argument à apply_model()***

### Exemple de DataFrame d'entrée
| index | observation_value                | name                |
|-------|----------------------------------|---------------------|
| 1     | "Alex Yu est allé au marché."    | ["Alex, Yu", 'Bob'] |
| 2     | "John Doe travaille à Paris."    | ["John", "Doe"]     |
| 3     | "Pas de nom associé ici."        | []                  |

Des colonnes supplémentaires peuvent être présentes, mais seules ces colonnes sont nécessaires pour le pipeline.

Attention: Le modèle fonctionne au meilleur de sa capacité lorsque les textes dans observation_value sont des phrases complètes !!!
---


## Utilisation
### 0. Charger les fonctions du Notebook(***model_packaged_v2***) et le model.safetensor: 
**Traduction**, **Pipeline NER**, **Blacklist**, **Remplacement de noms**, **Packaging du modèle**, **Application du modèle** et **Chargement des données**
Remplacer le fichier **bert-large-NER/model_safetensors.txt** par **bert-large-NER/model.safetensors** disponnible là: https://msss365-my.sharepoint.com/:u:/r/personal/alexandre_cantin_hsj_ssss_gouv_qc_ca/Documents/R%C3%A9seauNeuronale%20Anonymisation/bert-large-NER/model.safetensors?csf=1&web=1&e=sZO7xW

### 1. Charger les données
```python
spark = SparkSession.builder.appName("NERPipeline").getOrCreate()
df = ... # changer pour vos données
```

### 2. Préparer le pipelineNER, la blackliste et la BD de noms
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
loaded_model = AutoModelForTokenClassification.from_pretrained('bert-large-NER')   # Chargez le modèle Bert
loaded_tokenizer = AutoTokenizer.from_pretrained('bert-large-NER')   # Chargez le Tokenizer
ner_pipeline = pipeline("ner", model=loaded_model, tokenizer=loaded_tokenizer, aggregation_strategy="simple")   # Construire le ner_pipeline

with open("blacklist/blacklist.pkl", "rb") as file:  
    blacklist = pickle.load(file)  # Chargez la liste noire

name_database = load_name_database('prenomBD/prenom_M_et_F.csv')  # Chargez les remplacements de noms

```

### 3. Appliquer le modèle
```python
apply_model(model_packaged_v2, df, ner_pipeline, blacklist, name_database)
```

---

### Résultats
Le pipeline anonymise les entitées sensibles dans le texte. Il produit un fichier output.csv

---

### Remerciements
Ce projet utilise :
- **Transformers** par Hugging Face pour le NER (https://huggingface.co/dslim/bert-large-NER).
- **medical-wordlist** pour la construction de la blackliste (https://github.com/CodeSante/medical-wordlist)
- **donneesquebec** pour la BD de noms Québécois(https://www.donneesquebec.ca/recherche/dataset/banque-de-prenoms-garcons)(https://www.donneesquebec.ca/recherche/dataset/banque-de-prenoms-filles)
---
