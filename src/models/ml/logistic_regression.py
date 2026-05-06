from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
