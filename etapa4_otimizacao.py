
import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("used_cars_price.csv")
if 'car_id' in df.columns:
    df = df.drop('car_id', axis=1)

for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

map_sim_nao = {'Sim': 1,'Não':0,'Yes':1,'No':0}
binarias = ['air_conditioning','power_steering','power_windows','abs_brakes','sunroof','parking_sensors','imported']
for col in binarias:
    if col in df.columns:
        df[col] = df[col].map(map_sim_nao)

df = pd.get_dummies(df, drop_first=True)
X = df.drop('price_brl',axis=1)
y = df['price_brl']

X_temp,X_test,y_temp,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train,X_val,y_train,y_val = train_test_split(X_temp,y_temp,test_size=0.25,random_state=42)
X_tune = pd.concat([X_train,X_val]); y_tune = pd.concat([y_train,y_val])

def metrics(y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    return {"R2": r2_score(y_true,y_pred),
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true,y_pred)}

baseline = LinearRegression()
baseline.fit(X_train,y_train)
bl_val = metrics(y_val, baseline.predict(X_val))

candidates = {
    "Ridge":{"estimator":Ridge(),"search":"grid","params":{"alpha":[0.1,1,10,50,100]}},
    "Lasso":{"estimator":Lasso(max_iter=5000),"search":"grid","params":{"alpha":[0.0001,0.001,0.01,0.1,1]}},
    "RandomForest":{
        "estimator":RandomForestRegressor(random_state=42),
        "search":"random",
        "params":{"n_estimators":[100,200,300], "max_depth":[None,10,20], "min_samples_split":[2,5], "min_samples_leaf":[1,2]},
        "n_iter":20
    }
}

try:
    import xgboost as xgb
    candidates["XGBoost"] = {
        "estimator":xgb.XGBRegressor(random_state=42,objective="reg:squarederror"),
        "search":"random",
        "params":{"n_estimators":[100,200,300],"max_depth":[3,4,6],"learning_rate":[0.01,0.05,0.1],"subsample":[0.6,0.8,1]},
        "n_iter":20
    }
except:
    pass

results = {}
for name,cfg in candidates.items():
    if cfg["search"]=="grid":
        gs = GridSearchCV(cfg["estimator"],cfg["params"],cv=5,scoring="neg_root_mean_squared_error",n_jobs=-1)
        gs.fit(X_tune,y_tune)
        results[name] = gs
    else:
        rs = RandomizedSearchCV(cfg["estimator"],cfg["params"],n_iter=cfg["n_iter"],cv=5,scoring="neg_root_mean_squared_error",n_jobs=-1,random_state=42)
        rs.fit(X_tune,y_tune)
        results[name] = rs

summary = []
for name,s in results.items():
    summary.append((name,-s.best_score_,s.best_params_))
summary.sort(key=lambda x: x[1])
best_name,_,_ = summary[0]
best_model = results[best_name].best_estimator_
best_model.fit(pd.concat([X_train,X_val]), pd.concat([y_train,y_val]))

joblib.dump(best_model,"models/modelo_final.joblib")

pred = best_model.predict(X_test)
final = metrics(y_test,pred)
with open("artifacts/test_metrics.json","w") as f: json.dump(final,f)
