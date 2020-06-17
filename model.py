from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso

import numpy as np

class Model:
    def __init__(self, df, use_model):
        self.df = df
        self.use_model = use_model

    def main(self):
        X = self.df.drop('SalePrice', axis=1)
        y = self.df['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=45)

        # import your model
        if self.use_model == 'LassoRegression':
            model = Lasso(alpha=0.1)
        elif self.use_model == 'LinearRegression':
            model = LinearRegression()
        # fitting and predict your model
        model.fit(X_train, y_train)

        model.predict(X_test)
        errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        return model, errors