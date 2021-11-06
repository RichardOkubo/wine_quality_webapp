import pickle


class WineQuality:
    def __init__(self):
        self.free_sufer_dioxide_scaler = pickle.load(open('../parameter/free_sulfur_dioxide_scaler.pkl', 'rb'))
        self.total_sufer_dioxide_scaler = pickle.load(open('../parameter/total_sulfur_dioxide_scaler.pkl', 'rb'))

    def data_preparation(self, df):
        df['free sulfur dioxide'] = self.free_sufer_dioxide_scaler.transform(df[['free sulfur dioxide']].values)
        df['total sulfur dioxide'] = self.total_sufer_dioxide_scaler.transform(df[['total sulfur dioxide']].values)
        return df
