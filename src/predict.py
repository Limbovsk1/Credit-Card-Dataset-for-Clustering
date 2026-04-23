import joblib
import numpy as np
import warnings

# Отключаем лишние уведомления
warnings.filterwarnings("ignore", category=UserWarning)

# Загрузка
model = joblib.load('credit_model.v1')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Данные (17 чисел)
raw = np.array([[1500, 0.8, 500, 0, 500, 0, 0.5, 0, 0.4, 0, 0, 10, 5000, 200, 150, 0.1, 12]])

# Расчет
res = model.predict(scaler.transform(imputer.transform(raw)))[0]

print(f"Результат: Кластер №{res}")