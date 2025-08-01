import warnings
import pandas as pd
from numpy import where

warnings.filterwarnings('ignore')

# Функция для загрузки параметров из файла
file_params = 'parameters.txt'

def load_params_from_file(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Удаляем пробелы и символы новой строки
            line = line.strip()
            if line:  # Проверяем, что строка не пустая
                key, value = line.split('=')
                params[key.strip()] = float(value.strip()) if '.' in value else int(value.strip())
    return params

# Загрузка параметров из файла
params = load_params_from_file(file_params)

# Присваиваем значения переменным
zip_length = params['zip_length']
countAggreg_rec = params['countAggreg_rec']
successRate_rec = params['successRate_rec']
countAggreg_not_rec = params['countAggreg_not_rec']
successRate_not_rec = params['successRate_not_rec']
duplicate_agreg_drop = params['duplicate_agreg_drop']

# Проверка значений
print("zip_length: ", zip_length)
print("countAggreg_rec: ", countAggreg_rec)
print("successRate_rec:" , successRate_rec)
print("countAggreg_not_rec: ", countAggreg_not_rec)
print("successRate_not_rec: ", successRate_not_rec)
print("duplicate_agreg_drop: ", duplicate_agreg_drop)

print('Загрузка данных из файлов...')
df_test = pd.read_excel('test.xlsx')
df_train = pd.read_excel('train.xlsx')

print('Объединение фреймов...')
# Объединяем df_test и df_train по столбцу 'Agreg_ID'
merged_df = pd.merge(df_test, df_train[['Agreg_ID', 'countAggreg', 'countApp', 'successRate', 'Recomendation', 'notRecomended', 'max_traffic_source']], 
                     on='Agreg_ID', how='left', suffixes=('', '_train'))

# Переименовываем столбцы из df_train
merged_df.rename(columns={
    'max_traffic_source_train': 'TRAIN_max_traffic_source',
    'countAggreg_train': 'TRAIN_countAggreg',
    'countApp_train': 'TRAIN_countApp',
    'successRate_train': 'TRAIN_successRate',
    'Recomendation_train': 'TRAIN_Recomendation',
    'notRecomended_train': 'TRAIN_notRecomended'
}, inplace=True)

# Заполняем отсутствующие значения 'Не найдено'
for col in ['TRAIN_countApp', 'TRAIN_successRate', 'TRAIN_Recomendation', 'TRAIN_notRecomended']:
    merged_df[col] = merged_df[col].fillna('Not found')
    
merged_df = merged_df.drop_duplicates()

print('Добавление столбцов countAggregRecTrainAggregInTEST, SuccessRateRecTrainAggregInTEST')
# Возвращает словарь: {Agreg_ID: (countAggreg, successRate)}
def get_agreg_info(df):
    unique_rows = df.drop_duplicates(subset='Agreg_ID')[['Agreg_ID', 'countAggreg', 'successRate']]
    return dict(zip(unique_rows['Agreg_ID'], zip(unique_rows['countAggreg'], unique_rows['successRate'])))

# Заводим словари Агрегат : (countAggreg, successRate)
info_test = get_agreg_info(df_test)
info_train = get_agreg_info(df_train)

keys = merged_df['SERVICE'] + '_' + merged_df['TRAIN_Recomendation'].astype(str) + '_' + merged_df['cutted_zip_code'].astype(str)

mapped = keys.map(info_test)

merged_df['countAggregRecTrainggregInTEST'] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else 'Not found').where(merged_df['TRAIN_Recomendation'] != 0, '-')
merged_df['SuccessRateRecTrainAggregInTEST'] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else 'Not found').where(merged_df['TRAIN_Recomendation'] != 0, '-')

print('Добавление столбцов countAggregRecTrainAggregInTRAIN, SuccessRateRecTrainAggregInTRAIN')

#keys = merged_df['SERVICE'] + '_' + merged_df['TRAIN_Recomendation'].astype(str) + '_' + merged_df['cutted_zip_code'].astype(str)

mapped = keys.map(info_train)

merged_df['countAggregRecTrainAggregInTRAIN'] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else 'Not found').where(merged_df['TRAIN_Recomendation'] != 0, '-')
merged_df['SuccessRateRecTrainAggregInTRAIN'] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else 'Not found').where(merged_df['TRAIN_Recomendation'] != 0, '-')

print('Добавление столбцов countAggregRecTestAggregInTRAIN, SuccessRateRecTestAggregInTRAIN')

keys = merged_df['SERVICE'] + '_' + merged_df['Recomendation'].astype(str) + '_' + merged_df['cutted_zip_code'].astype(str)

mapped = keys.map(info_train)

merged_df['countAggregRecTestAggregInTRAIN'] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else 'Not found').where(merged_df['Recomendation'] != 0, '-')
merged_df['SuccessRateRecTestAggregInTRAIN'] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else 'Not found').where(merged_df['Recomendation'] != 0, '-')

print('Добавление столбцов Alternative_Recomendation, Alternative_notRecomended')

# Функции для альтернативных рекомендаций
def assign_traffic_source_and_recommendation(df, df2, countAggreg_rec, successRate_rec):
    # Создаем словарь для быстрого поиска по паре (SERVICE, cutted_zip_code)
    lookup = df2.set_index(['SERVICE', 'cutted_zip_code'])['max_traffic_source'].to_dict()

    # Формируем ключи для каждой строки df
    keys = list(zip(df['SERVICE'], df['cutted_zip_code']))

    # Ищем соответствующий traffic_source в словаре, если нет — ставим '-'
    df['Alternative_Recomendation'] = [lookup.get(key, '-') for key in keys]

    return df

'''
def assign_traffic_source_and_recommendation(df, df2, countAggreg_rec, successRate_rec):
    """
    Для каждой группы (SERVICE, cutted_zip_code) выбирает max_traffic_source
    и добавляет столбец 'Recomendation' на основе successRate.
    
    Параметры:
        df: DataFrame — исходный датафрейм (например, df_train или df_test)
        countAggreg_rec: int/float — минимальное значение countAggreg
        successRate_rec: float — минимальное значение successRate
    
    Возвращает:
        df — модифицированный датафрейм с колонками 'max_traffic_source' и 'Recomendation'
    """
    #df['TRAIN_successRate'] = df['TRAIN_successRate'].astype(float)
    
    # Приводим все значения к строке и сравниваем
    #mask_valid = df['TRAIN_successRate'].astype(str).str.lower() != 'not found'

    # Применяем маску
    #df_valid = df[mask_valid].copy()

    # Теперь можно безопасно преобразовать и отфильтровать
    #df_valid['TRAIN_successRate'] = df_valid['TRAIN_successRate'].astype(float)

    qualified = df2[
        (df2['countAggreg'] >= countAggreg_rec) &
        (df2['successRate'] >= successRate_rec)
    ]

    
    # Индексы с максимальным successRate по группам
    idx = qualified.groupby(['SERVICE', 'cutted_zip_code'])['TRAIN_successRate'].idxmax()
    
    # Сопоставление: (SERVICE, ZIP) → TRAFFIC_SOURCE
    max_ts_series = df2.loc[idx, ['SERVICE', 'cutted_zip_code', 'TRAFFIC_SOURCE']] \
        .set_index(['SERVICE', 'cutted_zip_code'])['TRAFFIC_SOURCE']
    
    # Присваиваем max_traffic_source по индексам
    df['Alternative_Recomendation'] = df2.set_index(['SERVICE', 'cutted_zip_code']) \
        .index.map(max_ts_series.to_dict()).fillna(0)
    
    # Рекомендация, если успех ниже порога
   
    df['Recomendation'] = df['max_traffic_source'].where(
        df['successRate'] < successRate_rec, 
        0
    )
    

    return df
'''
    
merged_df = assign_traffic_source_and_recommendation(merged_df, df_train, countAggreg_rec, successRate_rec)


# Добавление нового столбца
merged_df['Alternative_notRecomended'] = where(
    (merged_df['notRecomended'] == 'remove') & (merged_df['Alternative_Recomendation'] == 0),
    'remove',
    0
)

'''

# Формируем ключи
merged_keys = (
    merged_df['SERVICE'] + '_' +
    merged_df['TRAIN_Recomendation'].astype(str) + '_' +
    merged_df['cutted_zip_code'].astype(str)
)

# Извлекаем значения из словаря
mapped = merged_keys.map(info_test)

# Распаковываем значения из словаря
count_vals = mapped.map(lambda x: x['countAggreg'] if isinstance(x, dict) else 'Not Found')
success_vals = mapped.map(lambda x: x['successRate'] if isinstance(x, dict) else 'Not Found')

# Если TRAIN_Recomendation == 0, то проставляем 0, иначе — полученные значения
merged_df['countAggregRecTrainAggregInTEST'] = count_vals.where(merged_df['TRAIN_Recomendation'] != 0, 0)
merged_df['SuccessRateRecTrainAggregInTEST'] = success_vals.where(merged_df['TRAIN_Recomendation'] != 0, 0)
'''
if duplicate_agreg_drop:
    merged_df = merged_df.drop_duplicates(subset='Agreg_ID', keep='first')
    
print(merged_df)
print('Запись в итоговый файл testresult.xlsx')
merged_df.to_excel('testresult.xlsx', index=False)

input('Result file (testresult.xlsx) was successfully generated, press ENTER to exit\n')
