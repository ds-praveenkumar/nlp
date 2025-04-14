import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text  import  CountVectorizer
nltk.download('wordnet')
from pathlib import Path
import re

ROOT_DIR = Path('.').parent.as_posix()
print(ROOT_DIR)

class NLPDataCleaning:
    def __init__( self ):
        self.data_path = Path(ROOT_DIR) / 'data'
        self.file_name = [file.as_posix() for file in self.data_path.glob('*.csv')]
        self.df = None
        self.col_name = 'text'

    def load_csv( self ):
        df = pd.read_csv( self.file_name[0], usecols=[self.col_name])
        return  df 
    
    def drop_nulls( self ):
        df = self.load_csv()
        print(f'before drop: {df.shape}')
        df.dropna(subset=[self.col_name], inplace=True)
        print(f'after drop: {df.shape}')
        return df
    
    def lower( self ):
        df = self.drop_nulls()
        return df[self.col_name].str.lower()
    
    def remove_stopwords(self):
        df = self.lower()
        stop_words = stopwords.words('english')
        df = df.apply( lambda x: ' '.join( word for word in x.split() if word not in stop_words ) )
        return df
    
    def clean_text( self ):
        
        df = self.remove_stopwords()
        df = df.apply(lambda x: re.sub(r"http\S+|www.\S+", "", x) ) ## remove links
        df = df.apply(lambda x: re.sub(r"\w+@\w+\.com", "", x )) ## remove email
        df = df.apply(lambda x: re.sub(r"[.,;:'\?\"!`]", '', x)) ## remove puntuation mark
        df = df.apply( lambda x: re.sub(r"[@#$%^&*\(\)_<>+-\[\]\{\}]" , "", x )) # remove special symbols
        df= df.apply(lambda x: re.sub(r"½m|½s|½t|½ï", "", x))
        return df

    def word_lemmatization( self ):
        
        df = self.clean_text()
        lemma = WordNetLemmatizer()
        df = df.apply( lambda x: ' '.join( lemma.lemmatize(word,'v') for word in x.split()  ))
        return df 
    
    def count_vectorize( self ):
        
        df = self.word_lemmatization()
        cv =  CountVectorizer()
        X = cv.fit_transform(df).toarray()
        return pd.DataFrame(X , columns=cv.get_feature_names_out())
    
    
if __name__ == '__main__':
    dc = NLPDataCleaning()
    df = dc.count_vectorize()
    print(df)
             