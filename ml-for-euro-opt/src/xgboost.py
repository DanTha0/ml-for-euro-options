import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('SPX_options.csv')  #need to change to the csv file where our data is 
 
data['DaysToExpiration'] = (pd.to_datetime(data['Expiration']) - pd.to_datetime(data['Date'])).dt.days
data['OptionType'] = data['OptionType'].map({'C':1, 'P':0})
data['Moneyness'] = data['UnderlyingPrice'] / data['StrikePrice']

X = data[['StrikePrice', 'UnderlyingPrice', 'DaysToExpiration', 'OptionType', 'Moneyness']]
y = data['ImpliedVolatility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
