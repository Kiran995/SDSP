from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso

import numpy as np


class Model:
    def __init__(self, df, model_name, alpha):
        self.df = df
        self.model_name = model_name
        self.alpha = alpha

    def _fetch_data(self):
        X = self.df.drop('SalePrice', axis=1)
        y = self.df['SalePrice']
        return X, y

    def main(self):
        X, y = self._fetch_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=45)

        # import your model
        if self.model_name == 'LassoRegression':
            model = Lasso(alpha=self.alpha)
        elif self.model_name == 'LinearRegression':
            model = LinearRegression()

        # fitting and predict your model
        model.fit(X_train, y_train)

        model.predict(X_test)
        errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        return model, errors
