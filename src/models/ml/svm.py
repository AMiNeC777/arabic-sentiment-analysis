from sklearn.svm import LinearSVC


class SVMModel:
    def __init__(self):
        self.model = LinearSVC()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
