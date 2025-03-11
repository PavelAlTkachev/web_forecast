import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from pandas.tseries.offsets import DateOffset
import io

# Задаем начальную настройку
st.title("Прогноз продаж и анализ остатков")

# Загрузка данных
sales_file = st.file_uploader("Загрузите файл с продажами", type=["xlsx"])
remains_file = st.file_uploader("Загрузите файл с остатками", type=["xlsx"])

# Выбор периода анализа
start_date = pd.to_datetime(st.sidebar.date_input("Начальная дата", pd.to_datetime("2021-01-01")))
end_date = pd.to_datetime(st.sidebar.date_input("Конечная дата", pd.to_datetime("2025-01-01")))

v_min = st.sidebar.slider("Минимальное значение v_total", -1.0, 1.0, -0.3, step=0.05)
v_max = st.sidebar.slider("Максимальное значение v_total", -1.0, 1.0, 0.3, step=0.05)

# Выбор параметров прогноза
forecast_type = st.sidebar.selectbox(
    "Выберите тип прогноза",
    [
        "Ультракороткий прогноз",
        "Короткий прогноз",
        "Средний прогноз",
        "Длинный прогноз"
    ]
)

# Параметры прогноза с несколькими сдвигами
forecast_params = {
    "Ультракороткий прогноз": (1, [1, 2, 3]),
    "Короткий прогноз": (3, [3, 6, 12]),
    "Средний прогноз": (6, [6, 12, 24]),
    "Длинный прогноз": (12, [12, 24, 36])
}
period_X, shifts = forecast_params[forecast_type]

# Загрузка данных
if sales_file and remains_file:
    try:
        sales_df = pd.read_excel(sales_file)
        remains_df = pd.read_excel(remains_file)

        # Проверка обязательных столбцов
        if not set(["product_number", "sale_date", "quantity"]).issubset(sales_df.columns) or \
           not set(["product_number", "remain_date", "product_amount"]).issubset(remains_df.columns):
            st.error("Отсутствуют обязательные столбцы в одном из файлов.")
            st.stop()

        # Преобразование дат
        sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])
        remains_df['remain_date'] = pd.to_datetime(remains_df['remain_date'])

        # Фильтрация данных по выбранному периоду
        sales_df = sales_df[(sales_df['sale_date'] >= start_date) & (sales_df['sale_date'] <= end_date)]
        remains_df = remains_df[(remains_df['remain_date'] >= start_date) & (remains_df['remain_date'] <= end_date)]

        # Агрегация данных по месяцам
        sales_df['month'] = sales_df['sale_date'].dt.to_period('M').dt.to_timestamp()
        remains_df['month'] = remains_df['remain_date'].dt.to_period('M').dt.to_timestamp()
        print('remains', remains_df['product_amount'])
        sales_monthly = sales_df.groupby(['product_number', 'month'], as_index=False)['quantity'].sum()
        remains_monthly = remains_df.groupby(['product_number', 'month'], as_index=False)['product_amount'].mean()

        # Слияние данных
        merged_df = pd.merge(sales_monthly, remains_monthly, on=['product_number', 'month'], how='left')
        # Прогноз с учетом сдвигов
        forecast_results = []
        for product, group in sales_monthly.groupby('product_number'):
            group = group.set_index('month').sort_index()
            full_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='MS')
            group = group.reindex(full_index, fill_value=0)
            group.index.name = 'month'
            group = group.reset_index()
            group['product_number'] = product
            # Расчет скользящего прогноза
            group['FPB'] = group['quantity'].rolling(window=period_X, min_periods=period_X).sum()
            for i, shift in enumerate(shifts, 1):
                group[f'FP{i}'] = group['FPB'].shift(shift)
                group[f'v{i}'] = (
                            (1 - group[f'FP{i}'] / group['FPB'])).fillna(0)

            group['v_total'] = group[
                [f'v{i}' for i in range(1, len(shifts) + 1)]].mean(axis=1)

            # Применяем ограничение на v_total
            group['v_total'] = group['v_total'].clip(lower=v_min, upper=v_max)

            group['forecast'] = round((group['FPB'] * (1 + group['v_total'])), 0)
            group['forecast_date'] = group['month'] + DateOffset(
                months=period_X)
            forecast_results.append(group)

        forecast_df = pd.concat(forecast_results, ignore_index=True)
        merged_df = pd.merge(merged_df, forecast_df[['product_number', 'month', 'forecast', 'forecast_date']], on=['product_number', 'month'], how='left')


        # Месячный сдвиг фактических данных - корректируем сдвиг для разных типов прогноза
        def calculate_shifted_sales(group, period_X):
            shifted_sales = []
            for i in range(len(group)):
                future_sales = group['quantity'].iloc[
                               i:i + period_X].sum()
                shifted_sales.append(future_sales)
            return shifted_sales

        print('per', period_X)
        # Коррекция сдвига фактических продаж
        if period_X == 1:
            merged_df['shifted_sales_quantity'] = merged_df.groupby('product_number')['quantity'].shift(-period_X)
            print('working')
        else:
            merged_df['shifted_sales_quantity'] = merged_df.groupby(
            'product_number').apply(calculate_shifted_sales,
                                    period_X=period_X).explode().values
        # Расчет ошибки прогноза
        merged_df['error'] = np.abs(merged_df['forecast'] - merged_df['shifted_sales_quantity'])
        merged_df['MAPE'] = (merged_df['error'] / merged_df['shifted_sales_quantity']).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Выбор товара для анализа
        product_list = merged_df['product_number'].unique()
        selected_product = st.sidebar.selectbox("Выберите товар", product_list)
        product_data = merged_df[merged_df['product_number'] == selected_product]

        # Графики
        base = alt.Chart(product_data).encode(x=alt.X('forecast_date:T', axis=alt.Axis(format='%m.%y')))
        forecast_line = base.mark_line(color='red').encode(y='forecast:Q', tooltip=['forecast'])
        actual_line = base.mark_line(color='blue').encode(y='shifted_sales_quantity:Q', tooltip=['shifted_sales_quantity'])
        remain_line = base.mark_line(color='green').encode(y='product_amount:Q', tooltip=['product_amount'])

        # Точки на линиях
        forecast_points = base.mark_point(color='red', filled=True).encode(
            y='forecast:Q')
        actual_points = base.mark_point(color='blue', filled=True).encode(
            y='shifted_sales_quantity:Q')
        remain_points = base.mark_point(color='green', filled=True).encode(
            y='product_amount:Q')

        chart = (forecast_line + actual_line + remain_line + forecast_points + actual_points + remain_points).properties(
            title='Прогноз, фактические продажи и остатки',
            width=800,
            height=500
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        st.subheader("Таблица с прогнозами, ошибками и остатками")
        st.dataframe(merged_df, use_container_width=True)

        # Проверяем, что merged_df не пустой
        if not merged_df.empty:
            # Создаем объект байтового потока
            output = io.BytesIO()

            # Сохраняем DataFrame в байтовый поток как Excel
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                merged_df.to_excel(writer, index=False, sheet_name="Прогноз")

            # Получаем бинарные данные
            output.seek(0)

            # Кнопка для скачивания файла
            st.download_button(
                label="Скачать таблицу в Excel",
                data=output,
                file_name="прогноз_продаж.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Ошибка в обработке данных: {e}")
