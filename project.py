# import numpy as np
# import pandas as pd
# import re
# # from nltk.corpus import stopwords
# # from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# news_dataset = pd.read_csv('E:\Project_2\train.csv', encoding="unicode_escape",error_bad_lines=False)
# news_dataset
import pandas as pd
maldata = pd.read_csv("E:\Project_2\MalwareData.csv", sep ="|")
legit = maldata[0:41323].drop(["legitimate"], axis= 1)
mal = maldata[41323::].drop(["legitimate"], axis= 1)

print("The shape of the legit dataset is: %s sample, %s feature"%(legit.shape[0], legit.shape[1]))
print("The shape of the mal dataset is: %s sample, %s feature"%(mal.shape[0], mal.shape[1]))

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
# from sklearn import cross_validation
data_in = maldata.drop(['Name', 'md5', 'legitimate'], axis =1).values
labels = maldata['legitimate'].values
extratrees = ExtraTreesClassifier().fit(data_in, labels)
select = SelectFromModel(extratrees, prefit= True)
data_in_new = select.transform(data_in)
print(data_in.shape, data_in_new.shape)

import numpy as np
features = data_in_new.shape[1]
importances = extratrees.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(features):
    print("%d"%(f+1), maldata.columns[2+indices[f]],importances[indices[f]])

