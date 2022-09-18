# TODO: create classes for other functions, import and use here
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble, metrics

class MachineLearning:

    def __init__(self, csv_file) -> None:
        self.df = pd.read_csv(csv_file)
        self.df.drop(columns=self.df.columns[0], axis=1, inplace=True)
        pass

    def splitTestTrain(self):
        # split train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['clean_text'],self.df['vin'],test_size=0.3,random_state=0,shuffle=True)

    def gradientBoostClf(self):
        # split data into train and test
        self.splitTestTrain()

        # vectorize data
        X_train_GBC = self.X_train.values.reshape(-1)
        x_test_GBC = self.X_test.values.reshape(-1)
        vectorizer = CountVectorizer()
        X_train_GBC = vectorizer.fit_transform(X_train_GBC)
        x_test_GBC = vectorizer.transform(x_test_GBC)

        # Train the model
        model = ensemble.GradientBoostingClassifier(learning_rate=0.1,                                            
                                            n_estimators=2000,
                                            max_depth=9,
                                            min_samples_split=6,
                                            min_samples_leaf=2,
                                            max_features=8,
                                            subsample=0.9)
        model.fit(X_train_GBC, self.y_train)


        # Evaluate the model
        predicted_prob = model.predict_proba(x_test_GBC)[:,1]
        predicted = model.predict(x_test_GBC)

        accuracy = metrics.accuracy_score(predicted, self.y_test)
        print("Test accuracy: ", accuracy)
        print(metrics.classification_report(self.y_test, predicted, target_names=["0", "1"]))
        print("Test F-scoare: ", metrics.f1_score(self.y_test, predicted))
        

if __name__ == "__main__":
    csv_file = "./ml/data/dataclean_df_300.csv"
    ml = MachineLearning(csv_file)
    ml.gradientBoostClf()
