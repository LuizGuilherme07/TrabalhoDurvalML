import pandas as pd
import joblib

try:
    model = joblib.load('modelo_baseline_carros.pkl')
    df = pd.read_csv('used_cars_price.csv')
except:
    exit()

if 'car_id' in df.columns: df = df.drop('car_id', axis=1)
cols_num = df.select_dtypes(include=['float64', 'int64']).columns
for col in cols_num: df[col] = df[col].fillna(df[col].median())
cols_cat = df.select_dtypes(include=['object']).columns
for col in cols_cat: df[col] = df[col].fillna(df[col].mode()[0])
map_bin = {'Sim': 1, 'Não': 0, 'Yes': 1, 'No': 0}
for col in ['air_conditioning', 'power_steering', 'power_windows', 'abs_brakes', 'sunroof', 'parking_sensors', 'imported']:
    if col in df.columns: df[col] = df[col].map(map_bin)
df_model = pd.get_dummies(df, drop_first=True)


def formatar_grana(valor):
    valor_abs = abs(valor)
    if valor_abs >= 1_000_000:
        return f"R$ {valor/1_000_000:.1f}M" 
    elif valor_abs >= 1_000:
        return f"R$ {valor/1_000:.1f}K"  
    else:
        return f"R$ {valor:.2f}"          

nomes = df_model.drop('price_brl', axis=1).columns
tabela = pd.DataFrame({'Feature': nomes, 'Impacto': model.coef_})
tabela = tabela.sort_values(by='Impacto', ascending=False)

print("="*60)
print(f"PREÇO BASE (INTERCEPTO): {formatar_grana(model.intercept_)}")
print("="*60)

print("\nTOP 5 QUE AUMENTAM O PREÇO:")
print("-" * 60)
for i, row in tabela.head(5).iterrows():
    print(f"• {row['Feature']:<30} | {formatar_grana(row['Impacto'])}")

print("\nTOP 5 QUE DIMINUEM O PREÇO:")
print("-" * 60)
for i, row in tabela.tail(5).sort_values(by='Impacto', ascending=True).iterrows():
    print(f"• {row['Feature']:<30} | {formatar_grana(row['Impacto'])}")

print("\n" + "="*60)
print("LEGENDA DE VALORES:")
print("• K = Mil (Ex: 10K = R$ 10.000)")
print("• M = Milhão (Ex: 1M = R$ 1.000.000)")
print("="*60)