from sklearn.naive_bayes import MultinomialNB


class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
