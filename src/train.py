import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Загрузка и очистка
df = pd.read_csv("data/CC GENERAL.csv").drop(columns=['CUST_ID'])

# Обработка (Чиним пропуски + Масштабируем)
X = StandardScaler().fit_transform(SimpleImputer().fit_transform(df))

# Обучение
model = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = model.fit_predict(X)

# Сохранение (модель и инструменты обработки)
joblib.dump(model, 'credit_model.v1')
joblib.dump(StandardScaler().fit(SimpleImputer().fit_transform(df)), 'scaler.pkl')
joblib.dump(SimpleImputer().fit(df), 'imputer.pkl')

# Визуализация
plt.scatter(df['BALANCE'], df['PURCHASES'], c=clusters, cmap='viridis', alpha=0.5)
plt.colorbar(label='Кластер')
plt.show()

print("Готово. Модели сохранены.")