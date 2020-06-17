class Processing:
    def __init__(self, df, cat_col):
        self.df = df
        self.cat_col = cat_col

    def _convert_to_cat(self, col):
        return self.df[col].astype('category').cat.codes

    def main(self):
        for col in self.cat_col:
            self.df[col] = self._convert_to_cat(col)

        return self.df