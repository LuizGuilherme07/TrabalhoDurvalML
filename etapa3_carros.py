import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from matplotlib.ticker import FuncFormatter 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

try:
    df = pd.read_csv('used_cars_price.csv')
except FileNotFoundError:
    exit()

if 'car_id' in df.columns:
    df = df.drop('car_id', axis=1)

cols_numericas = df.select_dtypes(include=['float64', 'int64']).columns
for col in cols_numericas:
    df[col] = df[col].fillna(df[col].median())

cols_categoricas = df.select_dtypes(include=['object']).columns
for col in cols_categoricas:
    df[col] = df[col].fillna(df[col].mode()[0])

map_sim_nao = {'Sim': 1, 'Não': 0, 'Yes': 1, 'No': 0}
binarias = ['air_conditioning', 'power_steering', 'power_windows', 'abs_brakes', 
            'sunroof', 'parking_sensors', 'imported']

for col in binarias:
    if col in df.columns:
        df[col] = df[col].map(map_sim_nao)

df_model = pd.get_dummies(df, drop_first=True)

X = df_model.drop('price_brl', axis=1)
y = df_model['price_brl']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return r2_score(y_true, y_pred), np.sqrt(mse), mean_absolute_error(y_true, y_pred)

r2_train, rmse_train, mae_train = get_metrics(y_train, y_pred_train)
r2_val, rmse_val, mae_val = get_metrics(y_val, y_pred_val)

def fmt(x, pos):
    if x >= 1_000_000: return f'{x*1e-6:.1f}M'
    if x >= 1_000: return f'{x*1e-3:.0f}K'
    return f'{x:.0f}'

print(f"\n--- RESULTADOS ---")
print(f"TREINO    -> R2: {r2_train:.4f} | RMSE: R$ {rmse_train/1000:.1f}K | MAE: R$ {mae_train/1000:.1f}K")
print(f"VALIDACAO -> R2: {r2_val:.4f} | RMSE: R$ {rmse_val/1000:.1f}K | MAE: R$ {mae_val/1000:.1f}K")

if (r2_train - r2_val) > 0.10:
    print("STATUS: ALERTA (Possivel Overfitting)")
else:
    print("STATUS: SUCESSO (Modelo Generalizando Bem)")

plt.figure()
plt.subplots_adjust(bottom=0.25)
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(fmt)) 
ax.yaxis.set_major_formatter(FuncFormatter(fmt)) 

plt.scatter(y_val, y_pred_val, alpha=0.6, color='royalblue', label='Carros')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Linha Ideal')
plt.xlabel("price_brl (Valor Real)")
plt.ylabel("Predicted (Previsto)")
plt.title("Preço Real vs Previsto")
plt.legend()

txt1 = (
    "LEGENDA:\n"
    "• price_brl: Valor de tabela (CSV)\n"
    "• Predicted: Valor calculado pela IA\n"
    "• K = Mil (Ex: 50K = 50.000) | M = Milhão"
)
plt.figtext(0.5, 0.02, txt1, ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
plt.savefig("grafico_predito_vs_real.png")

plt.figure()
plt.subplots_adjust(bottom=0.25)
ax2 = plt.gca()
ax2.xaxis.set_major_formatter(FuncFormatter(fmt)) 

sns.histplot(y_val - y_pred_val, kde=True, color='purple')
plt.xlabel("Erro (R$)")
plt.title("Distribuição dos Erros")
plt.axvline(0, color='red', linestyle='--')

txt2 = (
    "LEGENDA:\n"
    "• Erro 0: IA acertou o preço\n"
    "• Esquerda: IA previu valor maior\n"
    "• K = Mil (Ex: 10K = 10.000)"
)
plt.figtext(0.5, 0.02, txt2, ha="center", fontsize=10, bbox={"facecolor":"purple", "alpha":0.1, "pad":5})
plt.savefig("grafico_residuos.png")

joblib.dump(model, 'modelo_baseline_carros.pkl')
print("\nConcluído.")