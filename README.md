# msds696 Practicum - Per Curiam Authorship detection
### Practicum Overview
There are four types of United States Supreme Court Decisions: Opinion Of The Court, Concurrences, Dissents, and Per Curiams. The author of the first three types is indicated in the written decision. Per Curiam decisions are unauthored, so it is unknown who wrote them. The purpose of this practicum is to investigate whether or not, by using Natural Language Processing and Machine Learning, progress can be made in determining the authorship of these unsigned United States Supreme Court Per Curiam decisions. The practicum uses a number of classification models and cosine similarity to achieve this. The classification models include LinearSVC, RandomForest, GradientBoosting, MutlinomialNB, and LinearDiscriminantAnalysis. The following items are discussed:
1. project data and data source
2. project directory structure layout and description
3. creating the conda environment
4. the models
5. process to run the notebooks
6. final note

### Project data and data source
The data for the practicum was scraped from https://supreme.justica.com. The timeframe studied is 08/07/2010 to 01/20/2017, so the file list was initailly pruned to that timeframe. The final data set consists of 360 authored decisions, 40 for each of nine justices, used for training and testing, and 35 per curiam to determine authorship; the data is stored in the form of csv files.

### Project directory structure layout and description

```
├── authored.kv
├── authored.model
├── data
│   ├── authored.tgz
│   ├── cases.tgz
│   ├── csv
│   ├── curiam.tgz
│   ├── json.tgz
│   └── years.tgz
├── feature_classification.ipynb
├── ipynb-completed.tgz
├── README.md
├── scripts.tgz
├── text_classification.ipynb
└── utils
    ├── classification.py
    ├── __init__.py
    ├── __pycache__
    ├── regexes.py
    └── utils.py

```
The project layout consists of four top level files and four directories.  There are a total of 45 python script, jupyter notebook, and csv files. The top level notebooks are the only ones that need to be run to get the classification results. The files required to run them have not been compressed. All others have as they are not necessary to run the notebooks. They are included for curiosity and examination purposed only.

**Top level files:**
1. ***authored.kv*** - custom word2vec word embeddings generated from the authored decisions
2. ***authored.model*** - custom word2vec model
3. ***feature_classification.ipynb*** - classification using ***only*** text features
4. ***text_classification.ipynb*** - classification/cosine using ***only*** text content 

**Directories:**
1. ***data*** - location for intermediate and final data. Contains the following sub-directories:
   a. ***cases*** and ***years*** - contains raw html files
   b. ***authored*** and ***curiam*** - raw, extracted, and cleaned data in text format
   c. ***json*** - contains json files used for data extraction, cleaning and engineering
   d. ***csv*** - destination for fully cleaned data. there are 19 intermediate and final files. files required to run the models are at the csv top-level directory. the rest are in the final subdirectory
2. ***ipynb-completed*** - contains jupyter notebooks used for all steps from fetching the data to getting the data in its final form for modeling.
3. ***scripts*** - contains a single file, *fix_hyphens.py*, used to assist in manually cleaning hyphen issues.
4. ***utils*** - contains three files: *classification.py*, *regex.py*, and *utils.py*. These files consist of custom regex patterns/functions and utility functions that are imported into the notebooks.

### Creating the environment
The practicum uses python 3.9.7 and a default conda environment built with the following:
1. conda install beautifulsoup4 cupy gensim jupyter lexicalrichness matplotlib numpy pendulum pytorch regex requests scikit-learn scipy seaborn spacy textblob torchaudio torchvision wordcloud
2. conda install cudatoolkit=11.3 -c pytorch
3. pip install readability
4. **NOTE**: pytorch and the cudatoolkit are not essential as they are used for intermediate data processing and are not used in the final notebooks.

### The models
Models are run along two paths. Per curiam classification results are included and discussed in the notebooks.
##### engineered features only. 
1. This path ignores the text itself and uses only the engineered features. 
2. implemented models: *Random Forest* and *LinearSCV*.
3. The features include character count, token count, sentence count, mean sentence length, sentence length std, number of unique terms, % stopwords, # of function words, # of quotes, legal entity count, entity ratio, mtld lexical richness score, hdd lexical richness score, dale chall readability score, average word length, and the count for each of 167 function words.
4. csv files used: *authored_engineered_data.csv* and *curiam_engineered_data.csv*.
5. jupyter notebook: *feature_classification.ipynb* from top level repository directory. The notebook can be run as is witout any configuration or setup (after environment as described above is created).

##### text content only
1. this path ignores the engineered features and uses on the text content.
2. implemented models: *Random Forest*, *LinearCSV*, *Cosine Similarity*
3. csv files used: *similarity.csv*, *authored_engineered_data.csv*, *curiam_engineered_data.csv*, *authored_similarity.csv*, *curiam_similarity.csv*
4. jupyter notebook: *text_classification.ipynb* from top level repository directory. The notebook can be run as is without any configuration or setup (after environment as described above is created).

### Process to run the notebooks
The process for running the notebooks is quite simple:
1. clone the repository
2. cd to top level repository directory
3. create a conda environment using Python 3.9.7 and install the packages as described above
4. activate environment
5. launch jupyter notebook or jupyter lab
6. load and run *feature_classification.ipynb* and *text_classification.ipynb*


### *Final Note:*
All the files necessary to recreate the data from scratch are included. However, trying to do so is a wasted effort as cleaning the data included an immense amount of manual cleaning, even within the notebooks, as several cells of some of them were modified and rerun. Ultimately, it is impossible to recreate the authored and curiam data as it exists in its current form.