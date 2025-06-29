import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Настройка страницы
st.set_page_config(page_title="Прогноз стоимости недвижимости", layout="wide")


# Загрузка данных
@st.cache_data
def load_data():
    try:
        # Проверяем несколько возможных путей к файлу
        possible_paths = [
            "realty_data.csv",  # В той же папке
            os.path.join("data", "realty_data.csv"),  # В папке data/
            os.path.join(os.path.dirname(__file__), "realty_data.csv")  # Абсолютный путь
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data = pd.read_csv(path)
                st.success(f"Данные успешно загружены из: {path}")
                return data

        st.error("Файл realty_data.csv не найден. Проверьте расположение файла.")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None


# Основной код
data = load_data()

if data is not None:
    # Проверяем необходимые столбцы
    required_cols = {'total_square', 'rooms', 'price', 'district', 'object_type'}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        st.error(f"Отсутствуют обязательные столбцы: {missing}")
    else:
        # Очистка данных
        data_clean = data.dropna(subset=list(required_cols))

        # Преобразование категориальных признаков
        data_clean = pd.get_dummies(data_clean, columns=['district', 'object_type'])

        # Выбираем признаки для модели
        features = ['total_square', 'rooms'] + \
                   [col for col in data_clean.columns if col.startswith(('district_', 'object_type_'))]

        # Подготовка данных
        X = data_clean[features]
        y = data_clean['price']

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Интерфейс
        st.title("Калькулятор стоимости недвижимости")

        col1, col2 = st.columns(2)
        with col1:
            total_square = st.number_input("Общая площадь (кв.м)",
                                           min_value=30,
                                           max_value=500,
                                           value=70)
            rooms = st.selectbox("Количество комнат",
                                 options=sorted(data_clean['rooms'].unique()))

        with col2:
            district = st.selectbox("Район",
                                    options=data['district'].unique())
            object_type = st.selectbox("Тип объекта",
                                       options=data['object_type'].unique())

        if st.button("Рассчитать стоимость"):
            try:
                # Создаем DataFrame с введенными данными
                input_data = pd.DataFrame([[total_square, rooms]],
                                          columns=['total_square', 'rooms'])

                # Добавляем dummy-переменные
                for col in features:
                    if col.startswith('district_'):
                        input_data[col] = 1 if col == f'district_{district}' else 0
                    elif col.startswith('object_type_'):
                        input_data[col] = 1 if col == f'object_type_{object_type}' else 0

                # Упорядочиваем столбцы как в обучающих данных
                input_data = input_data[features]

                # Прогнозирование
                price = model.predict(input_data)[0]
                st.success(f"### Предсказанная стоимость: {price:,.2f} руб.")

                # Дополнительная информация
                with st.expander("Показать статистику по району"):
                    district_stats = data[data['district'] == district]['price'].describe()
                    st.write(district_stats)

            except Exception as e:
                st.error(f"Ошибка предсказания: {str(e)}")

# Отображение сырых данных
if st.checkbox("Показать исходные данные"):
    st.write(data)