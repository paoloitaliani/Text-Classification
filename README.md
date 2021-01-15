# Text Multiclass Classification

## Introduction
The aim of this project is to build a model that is able to assign to a document, based on what it is about, one of the following 9 classes:  Agent, Place, Species, Work, Event, SportsSeason, UnitOfWork , TopicalConcept, Device. The data set used to train and test our models  contains 342,782 wikipedia articles and it can be downloaded [here](https://www.kaggle.com/danofer/dbpedia-classes?select=DBP_wiki_data.csv). All the models I'm going to use for the classification step require continous explanatory variables, but in this case the only variable that we have at disposal is the text of the document. In order to solve this problem we can represent text or words of each documet as a numerical vector and this technique is called word embedding.

## Data Cleaning

One import step when we deal with textual data is the data cleaning process, that basically it aims to delete all the elements of a document that are not useful for the analysis. Let's take a look how our documents look with an example:

```python
>>> df["text"][1]
'The 1917 Bali earthquake occurred at 06:50 local time on 21 January (23:11 on 20 January UTC). It had an estimated magnitude of 6.6 on the surface wave magnitude scale and had a maximum perceived intensity of IX (Violent) on the Mercalli intensity scale. It caused widespread damage across Bali, particularly in the south of the island. It triggered many landslides, which caused 80% of the 1500 casualties.'

```
As we can see our text is full of elements that are not going to improve our classification such as numbers and stopwords. Another good thing to do is to trasform all letters into lowercase and everything can be done by the following code.

```python
def clean_text(text):

    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',text)
    text = BAD_SYMBOLS_RE.sub('',text)

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')

```
Let's see how our documents have changed.

```python
>>> df["text"][1]
' bali earthquake occurred  local time  january   january utc estimated magnitude  surface wave magnitude scale maximum perceived intensity ix violent mercalli intensity scale caused widespread damage across bali particularly south island triggered many landslides caused   casualties'

```
Now it looks much better and we are done with text pre-processing.

## Text Classification with TF-IDF

As discussed in the introduction we have to change how the text of the documents is represented. I order to do that we can build numerical vectors of text based on the term frequency–inverse document frequency (TF-IDF). This statistic evaluates how relevant a word is to a document in a collection of documents (corpus). The numeric vectors that represent each document in our corpus are made of the TF-IDF statistics computed for each word and document. Below it is shown how it's done in python

```python
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df["text"]) #frequency of each word in the vocabulary for each document

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 1))
labels = df.category_id
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
```
Let's see how this vectorial representation looks for one of our documents
```python

```
Before moving on with the classification we have to change how the target variable (the classes of our interest) are coded, plus 
the definition of dictionaries that are going to be exploited later. Finally our data set is split into a train and test set.

```python
df['category_id'] = df['l1'].factorize()[0]
category_id_df = df[['l1', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'l1']].values)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf, labels, df.index, test_size=0.33, random_state=0)

```
I considered different models to perform the classification and a brief explanation of them is provided below.

#### Linear support-vector machine

SVMs are based on finding the "best" hyperplane that in a n-dimensional euclidean space is able to split two classes of a given dataset. By "best" I mean the hyperplane that is able to maximize the distance between the support vectors and the hyperplane itself. The support vectors are the closest points from both classes to the hyprplane. The following picture gives a more clear insight on how the hyperplane is defined in a very simple case, where we just have two explanatory variables and the hyperplane is a line.

<p align="center"><img src="images/image1.png" width=400></p>


SVMs don't support multi-class classification natively, but there are different approaches that solve this problem. The function LinearSVC from the scikit-learn package implements by default the One-vs-Rest approach that is based on splitting the multi-class dataset into multiple binary classification problems. An hyperplane is constructed for each class and each hyperplane separates the points of a the given class from the points of the remaing classes. The way this hyperplanes are defined is equivalent to the two classes case discussed above. The picture below shows a graphical representation of the hyperplanes in the case where we have only two explanatory variables

<p align="center"><img src="images/image2.png" width=400></p>

#### Logistic Regression

Logistic regression as SVMs doesn't support multi-class classification natively. The One-vs-Rest approach can be again implemented, in this case it is based on training a logistic regression classifier for each class, to compute the conditional probability of belonging to the corresponding class given our data. The observation is then assigned to the class that maximizes this probability.

#### Multinomial Naive Bayes 

This model as the previous one is based on finding the class that maximizes the conditional probability of belongig to it given our data. The difference originates from how this probability is computed that it is based on the bayes theorem and on the conditional indipendence between the feautures, that in our case are represented by the term frequencies–inverse document frequencies of words in each document. Below it is shown how the conditional probability of observing the k-th class, given the term frequencies–inverse document frequencies, is computed.


<p align="center"><img src="images/Screen Shot 2021-01-15 at 04.27.06.png" width=300></p>

#### Random Forest

Random forest is an ensamble approach based on grouping weak-learners (decision trees) providing a strong learner that is able to give more stable and accurate predictions. Basically we build a predefined number of trees using for each of them a random subsample of our dataset. At each node m features selected at random are used to perform the binary split that maximizes the homogeneity of the target variable, within the subsets created after the split. Each observation is classified according to the class that gets the most number of votes by the random trees. Below the picture shows the intuition behind this technique.

<p align="center"><img src="images/image4.png" width=400></p>
```python

```


```python

```

```python

```
