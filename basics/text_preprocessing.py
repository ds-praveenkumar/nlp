from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import re

corpus = [
    "I am a good boy",
    "This is a bad day",
    "The cat sat on the mat",
    'you are a good boy',
    "my email id is dfdfld@gmail.com"
    'visit http:\www.google.com'
]
tokenized = []

############ Preprocessing ##########

## tokenization
for text in corpus:
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokenized.append(' '.join(tokens ) )
    print( tokens )    

# lemmetization
lemma = WordNetLemmatizer()
for sent in corpus:
    sentence = ' '.join( lemma.lemmatize(word, 'v') for word in sent.split() )
    print(sent ,sentence)

## vectorization
countvect = CountVectorizer(lowercase=True)
X = countvect.fit_transform(tokenized).toarray()
feat_df = pd.DataFrame( X, columns=countvect.get_feature_names_out())
print(feat_df)


tfidf = TfidfVectorizer( sublinear_tf=True)
features = tfidf.fit_transform(tokenized).toarray()
feat_df = pd.DataFrame( features, columns=tfidf.get_feature_names_out())
print( feat_df )


## clean email
for text in corpus:
    print(re.sub(r'\w+@\w+.com', '', text))
    print( re.sub( r'http\S+|www.\S+', '', text ) )
