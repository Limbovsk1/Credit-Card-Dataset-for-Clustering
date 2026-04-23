import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Загрузка моделей и данных
df = pd.read_csv("data/CC GENERAL.csv")
model = joblib.load('credit_model.v1')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

st.title("Сегментация клиентов банка")

col1, col2 = st.columns(2)

with col1:
    st.header("Ввод данных")
    b = st.number_input("Баланс", value=1000.0)
    p = st.number_input("Покупки", value=500.0)
    c = st.number_input("Наличные", value=0.0)
    l = st.number_input("Лимит", value=3000.0)
    t = st.slider("Срок (мес)", 6, 12, 12)

    if st.button("Определить кластер"):
        # Создаем пустую строку со всеми колонками (17 шт) и заполняем нужные
        inp = pd.DataFrame([[0.0]*17], columns=imputer.feature_names_in_)
        inp.update(pd.DataFrame([[b, p, c, l, t]], columns=['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'TENURE']))
        
        res = model.predict(scaler.transform(imputer.transform(inp)))[0]
        st.success(f"Клиент отнесен к кластеру №{res}")
        
        desc = ["Обычные", "VIP/Активные", "Наличники", "Экономные", "Новые"]
        st.info(desc[res] if res < len(desc) else "Особая группа")

with col2:
    st.header("Визуализация")
    sample = df.dropna().sample(1000)
    
    # Предсказание для графика (быстрый расчет)
    X_s = imputer.transform(sample.drop('CUST_ID', axis=1))
    colors = model.predict(scaler.transform(X_s))

    fig, ax = plt.subplots()
    sc = ax.scatter(sample['BALANCE'], sample['PURCHASES'], c=colors, cmap='viridis', alpha=0.5, s=15)
    
    if 'res' in locals():
        ax.scatter(b, p, color='red', s=200, marker='X', label='Клиент')
        ax.legend()

    plt.colorbar(sc, label='Кластер')
    st.pyplot(fig)