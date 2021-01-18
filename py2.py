import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Flatten,GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


df = pd.read_csv("/Users/Niolo/Documents/python/data/DBP_wiki_data.csv")

g=sns.catplot(x="l1", kind="count",palette="rocket", data=df)
g.set_xticklabels(rotation=90)
plt.gcf().subplots_adjust(bottom=0.30)
############ DATA cleaning

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):

    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',text)
    text = BAD_SYMBOLS_RE.sub('',text)
    text = text.replace("'s", '')

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')


df['category_id'] = df['l1'].factorize()[0]
category_id_df = df[['l1', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'l1']].values)


count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df["text"]) #frequency of each word in the vocabulary for each document

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 1))
labels = df.category_id
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)






X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_tfidf, labels, df.index, test_size=0.33, random_state=0)

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0, multi_class="ovr")
]

cv_df = pd.DataFrame(index=range(len(models)))
entries = []
for model in models:
  model.fit(X_train, y_train)
  model_name = model.__class__.__name__
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  entries.append((model_name, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])

model=LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=category_id_df.l1.values, yticklabels=category_id_df.l1.values,cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.gcf().subplots_adjust(bottom=0.30)
plt.show()


######################
Y = pd.get_dummies(df['l1']).values
X_train, X_test, y_train, y_test = train_test_split(df['text'], Y,test_size=0.33, random_state = 0)

MAX_NB_WORDS=10000 #number of words cnsidered in tokenizer.texts_to_sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_train)

X_traintk = tokenizer.texts_to_sequences(X_train)
X_testtk = tokenizer.texts_to_sequences(X_test)

mostfreq_idx = {k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:9999]}
vocab_size = len(tokenizer.word_index) + 1

print(X_train[0])
print(X_traintk[0])

maxlen = 100

X_traintk = pad_sequences(X_traintk, padding='post', maxlen=maxlen)
X_testtk = pad_sequences(X_testtk, padding='post', maxlen=maxlen)


print(X_traintk[0])

EMBEDDING_DIM=50
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=maxlen))
model.add(Flatten())
#model.add(SpatialDropout1D(0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 64

history = model.fit(X_traintk, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_testtk, y_test))

loss, accuracy = model.evaluate(X_traintk, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_testtk, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)
######## Pre-trained word embedding


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

EMBEDDING_DIM=50
embedding_matrix = create_embedding_matrix('/Users/Niolo/Documents/python/data/glove.6B.50d.txt',mostfreq_idx, embedding_dim=50)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM,weights=[embedding_matrix], input_length=maxlen,trainable=True))
model.add(Flatten())
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 64


history = model.fit(X_traintk, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_testtk, y_test))

loss, accuracy = model.evaluate(X_traintk, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_testtk, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))