import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path
import shutil

import data_loader

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

class NeuralNetwork:
    def __init__(self, name="NNModel", reproc=True, trainon=0.9, epochs=200, traindf=pd.DataFrame(), testdf=pd.DataFrame()):
        """
        Creates a new neural network model that can be used to predict pricing in options
        Example Usage: mynetwork = NeuralNetwork("NeuralNetwork", False)
        :param name: str = Name model for saving purposes
        :param reproc: bool = Reprocess raw data? If processed folder has up-to-date data use False (default=True)
        :param trainon: float = By default =0.9, meaning 90% of data available is used to train, this is a percentage
        :param epochs: int = The number of epochs used (default=200 recommended)
        :param traindf: pd.DataFrame = can assign a training data df, if you already have cleaned and processed dataframe, default=processes data
        :param testdf: pd.DataFrame = can assign a test data df, if you already have cleaned and processed dataframe, default=processes data

        !!! You must call mynetwork.retrain() to create model
        !!! You must call mynetwork.save() to save model
        """
        self.y_test = None
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.preds = None
        self.mae = 0.0
        self.name = name
        self.BASE_DIR = Path(__file__).resolve().parent
        self.PROJECT_ROOT = self.BASE_DIR.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.TEST_DIR = self.DATA_DIR / "test"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.train = traindf
        self.test = testdf
        self.epochs = epochs
        self.available_data = len([f for f in self.RAW_DIR.iterdir() if f.is_file()])
        self.trainon = int(trainon*self.available_data)
        self.features = [
            'log_moneyness', 'time_sqrt', 'c_iv',
            'c_delta', 'c_gamma', 'c_vega', 'c_theta', 'c_rho'
        ]
        print("New Neural Network:\n-----------details-----------\n", self)
        if reproc:
            self.reprocess()
        if traindf.empty:
            self.maketrain()
        if testdf.empty:
            self.maketest()

    def __str__(self):
        return f"Neural network created:\n name={self.name},\n #train={self.trainon},\n #test={self.available_data-self.trainon},\n epochs={self.epochs},\n mae(0.0 on creation)={self.mae}"


    def reprocess(self):
        for item in self.PROCESSED_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        data_loader.process_each_file(raw_dir=self.RAW_DIR, processed_dir=self.PROCESSED_DIR, sep=",")
        self.available_data = len([f for f in self.RAW_DIR.iterdir() if f.is_file()])

        files = list(self.PROCESSED_DIR.glob("*.csv"))
        for f in files[:self.available_data-self.trainon]:
            shutil.move(str(f), self.TEST_DIR / f.name)

    def maketrain(self):
        dfs = []
        for csv_file in self.PROCESSED_DIR.glob("*.csv"):
            df = pd.read_csv(csv_file)
            dfs.append(df)

        self.train = pd.concat(dfs, ignore_index=True)
        self.train.columns = (self.train.columns.str.strip('[]'))
        self.train = self.train[
            (self.train['strike'] > 0) &
            (self.train['underlying_last'] > 0) &
            (self.train['dte'] >= 0)
            ].copy()
        self.train = self.train.dropna()

    def maketest(self):
        dfs = []
        if self.available_data-self.trainon == 0:
            print("Warning: NO TEST DATA FOUND")
        for csv_file in self.TEST_DIR.glob("*.csv"):
            df = pd.read_csv(csv_file)
            dfs.append(df)

        try:
            self.test = pd.concat(dfs, ignore_index=True)
        except:
            raise ValueError("Check the 'data/test' folder, make sure it is not empty!")
        self.test.columns = (self.test.columns.str.strip('[]'))
        self.test = self.test[
            (self.test['strike'] > 0) &
            (self.test['underlying_last'] > 0) &
            (self.test['dte'] >= 0)
            ].copy()
        self.test = self.test.dropna()

    def retrain(self):
        self.train['log_moneyness'] = np.log(self.train['underlying_last'] / self.train['strike'])
        self.train['time_sqrt'] = np.sqrt(self.train['dte'] / 365)

        target = 'c_last'
        X = self.train[self.features].values
        y = self.train[target].values.reshape(-1, 1)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        self.model = MLP(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

        self.model.eval()
        with torch.no_grad():
            preds_val = self.model(X_val_t)
            self.preds_val_unscaled = self.scaler_y.inverse_transform(preds_val.numpy())
            self.y_val_unscaled = self.scaler_y.inverse_transform(y_val_t.numpy())

    def predict(self):
        if self.model is None:
            raise ValueError("Model is not trained yet!")

        df_new = self.test.copy()
        df_new.columns = df_new.columns.str.strip('[]').str.lower()
        df_new['log_moneyness'] = np.log(df_new['underlying_last'] / df_new['strike'])
        df_new['time_sqrt'] = np.sqrt(df_new['dte'] / 365)

        X_new = df_new[self.features].values
        X_new_scaled = self.scaler_X.transform(X_new)
        X_new_t = torch.tensor(X_new_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(X_new_t)
            self.preds = self.scaler_y.inverse_transform(preds_scaled.numpy())
            self.y_test = df_new['c_last'].values.reshape(-1, 1)

        return self.preds

    def evaluate(self):
        if self.preds is None or self.y_test is None:
            raise ValueError("Run predict() first to generate predictions on test data!")

        mae = mean_absolute_error(self.y_test, self.preds)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.preds))
        r2 = r2_score(self.y_test, self.preds)

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R^2: {r2:.4f}")

    def save(self):
        torch.save(self.model.state_dict(), self.PROJECT_ROOT / "NNmodels" / f"{self.name+'.pth'}")
"""
mynetwork = NeuralNetwork("NeuralNetwork", True)
mynetwork.retrain()
mynetwork.save()
print(mynetwork.predict())
mynetwork.evaluate()
"""