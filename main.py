import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from pandas.tseries.offsets import DateOffset

# Установка отображения таблиц в широком формате
st.title("Прогноз продаж и анализ остатков")

# ============
# Функция проверки данных
# ============
def validate_data(df, required_columns, name):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Файл {name} не содержит обязательные колонки: {', '.join(missing_cols)}")
        return False
    return True

# ============
# Загрузка файлов
# ============
sales_file = st.file_uploader("Загрузите файл с продажами", type=["xlsx"])
remains_file = st.file_uploader("Загрузите файл с остатками", type=["xlsx"])

# ============
# Параметры расчёта
# ============
st.sidebar.header("Параметры расчёта")
period_X = st.sidebar.number_input("Период суммирования продаж (X месяцев)", min_value=1, value=1, step=1)
offset_Y = st.sidebar.number_input("Сдвиг FP1 (Y месяцев)", min_value=1, value=1, step=1)
offset_Z = st.sidebar.number_input("Сдвиг FP2 (Z месяцев)", min_value=1, value=2, step=1)
offset_W = st.sidebar.number_input("Сдвиг FP3 (W месяцев)", min_value=1, value=3, step=1)

# ============
# Выбор периода анализа
# ============
start_date = st.sidebar.date_input("Начальная дата", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("Конечная дата", pd.to_datetime("2025-01-01"))

if sales_file and remains_file:
    try:
        # ============
        # Чтение файлов
        # ============
        sales_df = pd.read_excel(sales_file)
        remains_df = pd.read_excel(remains_file)

        # Проверка данных
        if not validate_data(sales_df, ["product_number", "sale_date", "quantity"], "Продажи") or \
           not validate_data(remains_df, ["product_number", "remain_date", "product_amount"], "Остатки"):
            st.stop()

        sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date'])
        remains_df['remain_date'] = pd.to_datetime(remains_df['remain_date'])

        # Фильтрация по датам
        sales_df = sales_df[(sales_df['sale_date'] >= pd.to_datetime(start_date)) &
                            (sales_df['sale_date'] <= pd.to_datetime(end_date))]
        remains_df = remains_df[(remains_df['remain_date'] >= pd.to_datetime(start_date)) &
                                (remains_df['remain_date'] <= pd.to_datetime(end_date))]

        # ============
        # Агрегируем продажи и остатки по месяцам
        # ============
        # Обработка продаж
        sales_df['month'] = sales_df['sale_date'].dt.to_period('M').dt.to_timestamp()
        sales_monthly = sales_df.groupby(['product_number', 'month'], as_index=False)['quantity'].sum()
        if period_X > 1:
            sales_monthly = sales_monthly.groupby(['product_number', pd.Grouper(key='month', freq=f'{period_X}M')])['quantity'].sum().reset_index()

        # Обработка остатков
        remains_df['month'] = remains_df['remain_date'].dt.to_period('M').dt.to_timestamp()
        remains_monthly = remains_df.groupby(['product_number', 'month'], as_index=False)['product_amount'].mean()
        if period_X > 1:
            remains_monthly = remains_monthly.groupby(['product_number', pd.Grouper(key='month', freq=f'{period_X}M')])['product_amount'].mean().reset_index()

        # ============
        # Расчёт прогнозов
        # ============
        forecast_results = []
        for product, group in sales_monthly.groupby('product_number'):
            group = group.set_index('month').sort_index()
            full_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq='MS')
            group = group.reindex(full_index, fill_value=0)
            group.index.name = 'month'
            group = group.reset_index()
            group['product_number'] = product

            group['FPB'] = group['quantity'].rolling(window=period_X, min_periods=period_X).sum()
            group['FP1'] = group['FPB'].shift(offset_Y)
            group['FP2'] = group['FPB'].shift(offset_Z)
            group['FP3'] = group['FPB'].shift(offset_W)

            group['v1'] = (1 - group['FP1'] / group['FPB']).fillna(0)
            group['v2'] = ((1 - group['FP2'] / group['FPB']) / 2).fillna(0)
            group['v3'] = ((1 - group['FP3'] / group['FPB']) / 3).fillna(0)

            group['v_total'] = group['v1'] * 0.5 + group['v2'] * 0.3 + group['v3'] * 0.2
            group['forecast'] = group['FPB'] * (1 + group['v_total'])
            group['forecast_date'] = group['month'] + DateOffset(months=period_X)

            forecast_results.append(group)

        forecast_df = pd.concat(forecast_results, ignore_index=True)

        # ============
        # Анализ ошибок прогноза (MAPE)
        # ============
        actual_sales_df = sales_monthly.rename(columns={'month': 'forecast_date', 'quantity': 'actual_sales'})
        merged_df = pd.merge(forecast_df, actual_sales_df, on=['product_number', 'forecast_date'], how='left')
        merged_df['error'] = np.abs(merged_df['forecast'] - merged_df['actual_sales'])
        merged_df['MAPE'] = (merged_df['error'] / merged_df['actual_sales']).replace([np.inf, -np.inf], np.nan).fillna(0)

        # ============
        # Объединение с остатками
        # ============
        remains_forecast = remains_monthly.rename(columns={'month': 'forecast_date', 'product_amount': 'remain'})
        merged_df = pd.merge(merged_df, remains_forecast, on=['product_number', 'forecast_date'], how='left')
        merged_df['remain'] = merged_df['remain'].fillna(0)

        # ============
        # Выбор товара
        # ============
        product_list = merged_df['product_number'].unique()
        selected_product = st.sidebar.selectbox("Выберите товар", product_list)

        product_data = merged_df[merged_df['product_number'] == selected_product]

        # ============
        # Визуализация
        # ============
        # График прогноза, продаж и остатков
        base = alt.Chart(product_data).encode(x='forecast_date:T')
        forecast_line = base.mark_point(color='red').encode(y='forecast:Q', tooltip=['forecast'])
        actual_line = base.mark_line(color='blue').encode(y='actual_sales:Q', tooltip=['actual_sales'])
        remain_line = base.mark_line(color='green').encode(y='remain:Q', tooltip=['remain'])

        chart = (forecast_line + actual_line + remain_line).properties(
            title='Прогноз, фактические продажи и остатки',
            width=600,
            height=800
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # Гистограмма ошибок
        st.subheader("Гистограмма ошибок прогноза")
        error_chart = alt.Chart(product_data).mark_bar().encode(
            x=alt.X('MAPE:Q', bin=True, title="Ошибка (MAPE)"),
            y='count()'
        )
        st.altair_chart(error_chart, use_container_width=True)

        # ============
        # Отображение таблицы
        # ============
        st.subheader("Таблица с прогнозами, ошибками и остатками")
        st.dataframe(merged_df, use_container_width=True)

        # Сохранение таблицы в Excel
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name="Прогнозы")
            processed_data = output.getvalue()
            return processed_data

        excel_data = to_excel(merged_df)
        st.download_button(
            label="Скачать таблицу в Excel",
            data=excel_data,
            file_name="sales_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Логирование
        st.text("Расчёт завершен. Данные готовы к анализу.")

    except Exception as e:
        st.error(f"Ошибка в обработке данных: {e}")