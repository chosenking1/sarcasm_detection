# Author = Joshua

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data_1 = pd.read_json("sarcasm.json")
data_2 = pd.read_json("Sarcasm_Headlines_Dataset_v2.json")
data = pd.concat([data_1, data_2])
data.head()

data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# %%
model = BernoulliNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
# %%
user = "So this is how you can use machine learning to detect sarcasm by using the Python programming language"
# user = "granny starting to fear spiders in the garden might be real"
data = cv.transform([user]).toarray()
output = model.predict(data)
print(data)
print(output)
# %%
import pickle

# %%
model.predict(cv.transform(["joy in me"]).toarray())


# %%
class Concern:
    def predict_sarcasm(self):
        tokenized = cv.transform([self]).toarray()
        result = model.predict(tokenized)

        if result[0] == 1:
            return "It's a sarcasm!"
        else:
            return "It's not a sarcasm."


# %%
pickle.dump(Concern, open('sarcas.pkl', 'wb'))

# %%
man = pickle.load(open('sarcas.pkl', 'rb'))
man.predict_sarcasm("hope")
