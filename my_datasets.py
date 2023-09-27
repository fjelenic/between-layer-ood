import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
from sklearn.datasets import fetch_20newsgroups



class AmazonReviewData:
    def __init__(self):
        self.name = "AmazonReview"
        self.nli = False
        self.categories = ['Gift_Card_v1_00', 'Software_v1_00', 'Video_Games_v1_00', 'Luggage_v1_00', 'Video_v1_00', 'Grocery_v1_00', 'Furniture_v1_00', 'Musical_Instruments_v1_00', 'Watches_v1_00', 'Tools_v1_00', 'Baby_v1_00', 'Jewelry_v1_00']
        self.num_out = len(self.categories)
        self.num_iter = 30
        self.frac = 0.0025
        self.train_size = 0.8
        self.min_len = 15
        self.seed = 42
        
    def load(self):
        df = pd.read_csv("amazon_data.csv")
        mapping = dict([(text, i) for i,text in enumerate(sorted(df[df["class"] == "X_train"]["text"].tolist()))])
        return df[df["class"] == "X_train"]["text"].tolist(), df[df["class"] == "X_test"]["text"].tolist(), df[df["class"] == "y_train"]["text"].tolist(), df[df["class"] == "y_test"]["text"].tolist(), mapping
"""        
    def load(self):
        data = []
        for i, lab in enumerate(self.categories):
            X = load_dataset('amazon_us_reviews', lab)["train"]
            df_temp = pd.DataFrame(X)
            df_temp["y"] = [i for _ in range(len(X['review_body']))]
            data.append(df_temp)
    
        df = pd.concat(data)
        df["lens"] = [len(x.split()) for x in df["review_body"]]
        df = df[df['lens'] >= self.min_len]
        df.drop_duplicates(subset="review_body", inplace=True, keep=False)

        df_sample = df.groupby('y', group_keys=False).sample(frac=self.frac, random_state=self.seed)
        X_train, X_test, y_train, y_test = train_test_split(df_sample['review_body'], df_sample['y'], train_size=self.train_size, random_state=self.seed, stratify=df_sample['y'])
        
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist(), mapping
"""

    
class AmazonReview2Data:
    def __init__(self):
        self.name = "AmazonReview2"
        self.nli = False
        self.categories = ['Gift_Card_v1_00', 'Software_v1_00', 'Video_Games_v1_00', 'Luggage_v1_00', 'Video_v1_00', 'Grocery_v1_00', 'Furniture_v1_00', 'Musical_Instruments_v1_00', 'Watches_v1_00', 'Tools_v1_00', 'Baby_v1_00', 'Jewelry_v1_00']
        self.num_out = 1
        self.num_iter = 30
        self.frac = 0.0025
        self.train_size = 0.8
        self.min_len = 15
        self.seed = 42
        
    def load(self):
        df = pd.read_csv("amazon2_data.csv")
        mapping = dict([(text, i) for i,text in enumerate(sorted(df[df["class"] == "X_train"]["text"].tolist()))])
        return df[df["class"] == "X_train"]["text"].tolist(), df[df["class"] == "X_test"]["text"].tolist(), df[df["class"] == "y_train"]["text"].tolist(), df[df["class"] == "y_test"]["text"].tolist(), mapping

"""
    def load(self):
        data = []
        for i, lab in enumerate(self.categories):
            X = load_dataset('amazon_us_reviews', lab)["train"]
            df_temp = pd.DataFrame(X)
            df_temp["y"] = [i for _ in range(len(X['review_body']))]
            data.append(df_temp)
    
        df = pd.concat(data)
        df["lens"] = [len(x.split()) for x in df["review_body"]]
        df = df[df['lens'] >= self.min_len]
        df.drop_duplicates(subset="review_body", inplace=True, keep=False)

        df_sample = df.groupby('y', group_keys=False).sample(frac=self.frac, random_state=self.seed)
        X_train, X_test, y_train, y_test = train_test_split(df_sample['review_body'], df_sample['y'], train_size=self.train_size, random_state=self.seed, stratify=df_sample['y'])
        
        X_train, X_test, y_train, y_test = [x.tolist() for x in [X_train, X_test, y_train, y_test]]
        
        X_train2, y_train2, X_test2, y_test2 = [],[],[],[]
        for i in range(len(X_train)):
            if y_train[i] in [5,10]:
                X_train2.append(X_train[i])
                y_train2.append(y_train[i]//5-1)
        for i in range(len(X_test)):
            if y_test[i] in [5,10]:
                X_test2.append(X_test[i])
                y_test2.append(y_test[i]//5-1)
        
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train2, X_test2, y_train2, y_test2, mapping
"""
    
    
class BigPatentData:
    def __init__(self):
        self.name = "BigPatent"
        self.nli = False
        self.num_out = 9
        self.num_iter = 30
        self.split_frac = {"train": 0.03, "test": 0.2}
        self.seed = 42
        
    def load(self):
        data = []
        for split in ["train", "test"]:
            X, y = [], []
            for i, lab in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "y"]):
                train = load_dataset('big_patent', lab, split=split)
                X.extend(train['abstract'])
                y.extend([i for _ in range(len(train['abstract']))])
            df_temp = pd.DataFrame({"X":X, "y":y})
            df_temp["split"] = split
            data.append(df_temp)
    
        df = pd.concat(data)
        df.drop_duplicates(subset="X", inplace=True, keep=False)

        df_train, df_test = df[df["split"] == "train"], df[df["split"] == "test"]
        df_train = df_train.groupby('y', group_keys=False).sample(frac=self.split_frac["train"], random_state=self.seed) 
        df_test = df_test.groupby('y', group_keys=False).sample(frac=self.split_frac["test"], random_state=self.seed)

        X_train, X_test, y_train, y_test = df_train["X"].tolist(), df_test["X"].tolist(), df_train["y"].tolist(), df_test["y"].tolist()
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train, X_test, y_train, y_test, mapping
    
    
class BigPatent2Data:
    def __init__(self):
        self.name = "BigPatent2"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30
        self.split_frac = {"train": 0.03, "test": 0.2}
        self.seed = 42
        
    def load(self):
        data = []
        for split in ["train", "test"]:
            X, y = [], []
            for i, lab in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "y"]):
                train = load_dataset('big_patent', lab, split=split)
                X.extend(train['abstract'])
                y.extend([i for _ in range(len(train['abstract']))])
            df_temp = pd.DataFrame({"X":X, "y":y})
            df_temp["split"] = split
            data.append(df_temp)
    
        df = pd.concat(data)
        df.drop_duplicates(subset="X", inplace=True, keep=False)

        df_train, df_test = df[df["split"] == "train"], df[df["split"] == "test"]
        df_train = df_train.groupby('y', group_keys=False).sample(frac=self.split_frac["train"], random_state=self.seed) 
        df_test = df_test.groupby('y', group_keys=False).sample(frac=self.split_frac["test"], random_state=self.seed)
        
        df_train = df_train[df_train["y"].isin([0,6])]
        df_test = df_test[df_test["y"].isin([0,6])]
        
        df_train["y"] = df_train["y"] // 6
        df_test["y"] = df_test["y"] // 6

        X_train, X_test, y_train, y_test = df_train["X"].tolist(), df_test["X"].tolist(), df_train["y"].tolist(), df_test["y"].tolist()
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train, X_test, y_train, y_test, mapping

    
class FilmGenreData:
    def __init__(self):
        self.name = "FilmGenre"
        self.nli = False
        self.num_out = 15
        self.num_iter = 30
        self.split_size = {"train": 500, "test": 400}
        self.split_frac = {"train": 0.5, "test": 0.15}
        self.seed = 42
        
    def load(self):
        df_train = pd.read_csv("film_genre_train.txt", engine="python", sep=" ::: ", names=["id", "movie", "genre", "summary"]) 
        df_train["test"] = False

        df_test = pd.read_csv("film_genre_test.txt", engine="python", sep=" ::: ", names=["id", "movie", "genre", "summary"])
        df_test["test"] = True

        df = pd.concat((df_train, df_test))
        df.drop_duplicates(subset="summary", inplace=True, keep=False)

        genres = ['drama', 'documentary', 'comedy', 'horror', 'thriller', 'action', 'western', 'reality-tv', 'adventure', 'family', 'music', 'romance', 'sci-fi', 'adult', 'crime']

        df.query(f"genre in {genres}", inplace=True)

        df['label'] = df["genre"].map({s:i for i,s in enumerate(genres)})

        df_train, df_test = df[df["test"] == False], df[df["test"] == True]

        #df_train = df_train.groupby('genre').sample(n=self.split_size["train"], random_state=self.seed) 
        #df_test = df_test.groupby('genre').sample(n=self.split_size["test"], random_state=self.seed) 
        
        df_train = df_train.groupby('genre').sample(frac=self.split_frac["train"], random_state=self.seed) 
        df_test = df_test.groupby('genre').sample(frac=self.split_frac["test"], random_state=self.seed)

        X_train, X_test, y_train, y_test = df_train["summary"].tolist(), df_test["summary"].tolist(), df_train["label"].tolist(), df_test["label"].tolist()
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train, X_test, y_train, y_test, mapping
    
    
class FilmGenre2Data:
    def __init__(self):
        self.name = "FilmGenre2"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30
        self.split_size = {"train": 500, "test": 400}
        self.split_frac = {"train": 0.5, "test": 0.15}
        self.seed = 42
        
    def load(self):
        df_train = pd.read_csv("film_genre_train.txt", engine="python", sep=" ::: ", names=["id", "movie", "genre", "summary"]) 
        df_train["test"] = False

        df_test = pd.read_csv("film_genre_test.txt", engine="python", sep=" ::: ", names=["id", "movie", "genre", "summary"])
        df_test["test"] = True

        df = pd.concat((df_train, df_test))
        df.drop_duplicates(subset="summary", inplace=True, keep=False)

        genres = ['drama', 'documentary', 'comedy', 'horror', 'thriller', 'action', 'western', 'reality-tv', 'adventure', 'family', 'music', 'romance', 'sci-fi', 'adult', 'crime']

        df.query(f"genre in {genres}", inplace=True)

        df['label'] = df["genre"].map({s:i for i,s in enumerate(genres)})

        df_train, df_test = df[df["test"] == False], df[df["test"] == True]

        #df_train = df_train.groupby('genre').sample(n=self.split_size["train"], random_state=self.seed) 
        #df_test = df_test.groupby('genre').sample(n=self.split_size["test"], random_state=self.seed) 
        
        df_train = df_train.groupby('genre').sample(frac=self.split_frac["train"], random_state=self.seed) 
        df_test = df_test.groupby('genre').sample(frac=self.split_frac["test"], random_state=self.seed)
        
        df_train = df_train[df_train["label"].isin([0,1])]
        df_test = df_test[df_test["label"].isin([0,1])]

        X_train, X_test, y_train, y_test = df_train["summary"].tolist(), df_test["summary"].tolist(), df_train["label"].tolist(), df_test["label"].tolist()
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train, X_test, y_train, y_test, mapping
    

class NewsGroupsData:
    def __init__(self):
        self.name = "NewsGroups"
        self.nli = False
        self.num_out = 20
        self.num_iter = 30
        
    def load(self):
        train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        X_train, X_test, y_train, y_test = self.clean(train.data, test.data, train.target.tolist(), test.target.tolist())
        mapping = dict([(text, i) for i,text in enumerate(sorted(X_train))])
        return X_train, X_test, y_train, y_test, mapping
    
    def clean(self, X_train, X_test, y_train, y_test):
        change = True
        while change:
            change = False
            for x in X_train:
                if x in X_test:
                    change = True
                    i_test = X_test.index(x)
                    X_test.pop(i_test)
                    y_test.pop(i_test)
        return X_train, X_test, y_train, y_test
    
    
class OneBillionData:
    def __init__(self):
        self.name = "OneBillion"
        self.nli = False
        self.num_out = None
        self.num_iter = None
        
    def load(self):
        #train = load_dataset('lm1b', split='train')
        test = load_dataset('lm1b', split='test')
        #mapping = dict([(text, i) for i,text in enumerate(sorted(train['text']))])
        return None, test["text"], None, None, None
    

class YelpData:
    def __init__(self):
        self.name = "Yelp"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30
        
    def load(self):
        train = load_dataset('yelp_polarity', split='train')
        test = load_dataset('yelp_polarity', split='test')
        mapping = dict([(text, i) for i,text in enumerate(sorted(train['text']))])
        return train['text'], test["text"], train['label'], test['label'], mapping
    
    
class SST2Data:
    def __init__(self):
        self.name = "SST2"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30
        
    def load(self):
        train = load_dataset('sst', split='train')
        test = load_dataset('sst', split='test')
        mapping = dict([(text, i) for i,text in enumerate(sorted(train['sentence']))])
        return train['sentence'], test["sentence"], self.rounding(train['label']), self.rounding(test['label']), mapping
    
    def rounding(self, y):
        return [round(yi) for yi in y]
    

class SubjectivityData:
    def __init__(self):
        self.name = "Subjectivity"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30
        
    def load(self):
        with open("obj", "r") as f:
            df_obj = [d.strip() for d in f.readlines()]
        with open("sub", "r", encoding="ISO-8859-1") as f:
            df_sub = [d.strip() for d in f.readlines()]
            
        X_raw = df_obj + df_sub
        y_raw = [0 for _ in range(len(df_obj))] + [1 for _ in range(len(df_sub))]
        random.seed(5)

        inds = list(range(len(X_raw)))
        random.shuffle(inds)
        X = [X_raw[i] for i in inds]
        y = [y_raw[i] for i in inds]
        
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
        
        mapping = dict([(text, i) for i,text in enumerate(sorted(train_X))])
        
        return train_X, test_X, train_y, test_y, mapping

    
class TrecData:
    def __init__(self):
        self.name = "TREC"
        self.nli = False
        self.num_out = 6
        self.num_iter = 30
    
    def load(self):
        train = load_dataset('trec', split='train')
        test = load_dataset('trec', split='test')
        mapping = dict([(text, i) for i,text in enumerate(sorted(train['text']))])
        return train['text'], test["text"], train['coarse_label'], test['coarse_label'], mapping


class AGNewsData:
    def __init__(self):
        self.name = "AG-News"
        self.nli = False
        self.num_out = 4
        self.num_iter = 30
    
    def load(self):
        train = load_dataset('ag_news', split='train')
        test = load_dataset('ag_news', split='test')
        
        train_X, train_y = self.subsample(train['text'], train['label'], 20_000)
        mapping = dict([(text, i) for i,text in enumerate(sorted(train_X))])
        
        return train_X, test["text"], train_y, test['label'], mapping
    
    def subsample(self, X, y, sample_size):
        train_X, _, train_y, _ = train_test_split(X, y, train_size=sample_size, random_state=5, stratify=y)
        return train_X, train_y
