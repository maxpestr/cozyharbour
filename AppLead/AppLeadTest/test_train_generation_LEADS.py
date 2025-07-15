# -*- coding: utf-8 -*-

#Загрузка файла
import pandas as pd
import warnings
import random

warnings.filterwarnings('ignore')
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
zipSub_length = params['zipSub_length']
countAggregSub_rec = params['countAggregSub_rec']
successRateSub_rec = params['successRateSub_rec']
countAggregSub_not_rec = params['countAggregSub_not_rec']
successRateSub_not_rec = params['successRateSub_not_rec']
partition_index = params['partition_index']

# Проверка значений
print("zipSub_length: ", zipSub_length)
print("countAggregSub_rec: ", countAggregSub_rec)
print("successRateSub_rec:" , successRateSub_rec)
print("countAggregSub_not_rec: ", countAggregSub_not_rec)
print("successRateSub_not_rec: ", successRateSub_not_rec)
print("partition_index: ", partition_index)


#filename = '_отсорт-preleads_20240726.xlsx'
filename = input('Input file name\n')
print('Loading data file...')
df = pd.read_excel(filename)#, usecols=['CAMPAIGN', 'SERVICE', 'TRAFFIC_SOURCE', 'LEAD_ID', 'ZIP_CODE', 'IS_APPOINTMENT_SET'])
print('Processing...')
print('Filtering...')
#FILTERING
df = df.loc[(df['CAMPAIGN_CURRENT_TYPE'] == 'Appointments') | (df['CAMPAIGN_CURRENT_TYPE'] == 'Leads')]
df = df.dropna(subset=['ZIP_CODE', 'SERVICE', 'TRAFFIC_SOURCE'])

#Обрезаем zip коды
df['cutted_zip_code'] = df['ZIP_CODE'].astype(str).str[:zipSub_length]


#Генерируем уникальный ID группы на основе SERVICE, TRAFFIC_SOURCE и cutted_zip_code
#unique_IDs = df[['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code']].drop_duplicates().to_numpy().tolist()
'''
def create_group_ID(row):
    #group_key = unique_IDs.index([row['SERVICE'], row['TRAFFIC_SOURCE'], row['cutted_zip_code']])
    group_key = str(row['SERVICE'])+'_'+ str(row['TRAFFIC_SOURCE'])+'_'+str(row['cutted_zip_code'])
    return group_key
'''
print('Unique agregates IDs generation...')
#df['Agreg_ID'] = df.apply(create_group_ID, axis = 1)
df['Agreg_ID'] = df['SERVICE'] + '_' + df['TRAFFIC_SOURCE'] + '_' + df['cutted_zip_code'] 
'''
test_density = -0.33

while test_density <= 0 or test_density > 50:
    test_density = int(input('Enter test percentage (without \'%\')\n'))

test_density = test_density / 100

list_test = []

for i in df.index.tolist():
    if random.random() < test_density:
        list_test.append(i)

print(len(list_test))
#print(len(list_train))

df_test = df.loc[df.index.isin(list_test)]
df_train = df.loc[~df.index.isin(list_test)]
'''

df_train = df.loc[df.index <= partition_index]
df_test = df.loc[df.index > partition_index]
print('Test and Train samples generation...')
#df_train = df.loc[df.index <= 3*len(df)//4]
#df_test = df.loc[df.index > 3*len(df)//4]

print('\nTrain\n', df_train, '\n')
print('Test\n', df_test, '\n')
#print('test.xlsx')
#считаем агрегаты с одинаковым cutted_zip_code
df_train['countAggreg'] = df_train.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'])['LEAD_ID'].transform('count')
#считаем, сколько удач (Submission-ов) для каждого агрегата
df_train['countSub'] = df_train.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'])['IS_SUBMITTED'].transform(lambda x: x.sum())

#вычисляем конверсию на агрегате
df_train['successRateSub'] = df_train['countSub']/df_train['countAggreg']

#функция для нахлждения максимального traffic_source для набора 'SERVICE', 'cutted_zip_code'
def assign_traffic_source_and_recommendation(df, countAggreg_rec, successRate_rec):
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
    # Отбор строк, удовлетворяющих условиям
    qualified = df[(df['countAggreg'] >= countAggregSub_rec) & 
                   (df['successRateSub'] >= successRateSub_rec)]
    
    # Индексы с максимальным successRateSub по группам
    idx = qualified.groupby(['SERVICE', 'cutted_zip_code'])['successRateSub'].idxmax()
    
    # Сопоставление: (SERVICE, ZIP) → TRAFFIC_SOURCE
    max_ts_series = df.loc[idx, ['SERVICE', 'cutted_zip_code', 'TRAFFIC_SOURCE']] \
        .set_index(['SERVICE', 'cutted_zip_code'])['TRAFFIC_SOURCE']
    
    # Присваиваем max_traffic_source по индексам
    df['max_traffic_source'] = df.set_index(['SERVICE', 'cutted_zip_code']) \
        .index.map(max_ts_series.to_dict()).fillna(0)
    
    # Рекомендация, если успех ниже порога
    df['Recomendation'] = df['max_traffic_source'].where(
        df['successRateSub'] < successRateSub_rec, 
        0
    )

    return df

#функция для нахождения агрегатов под удаление
def assign_not_recommended(df, countAggregSub_not_rec, successRateSub_not_rec):
    """
    Отмечает строки как 'remove', если они удовлетворяют условиям:
    - countAggreg > порога
    - successRate < порога
    - Recomendation == 0

    Все остальные строки получают 0 в колонке 'notRecomended'.
    """
    condition = (
        (df['countAggreg'] > countAggregSub_not_rec) &
        (df['successRateSub'] < successRateSub_not_rec) &
        (df['Recomendation'] == 0)
    )

    df['notRecomended'] = pd.Series('remove', index=df.index).where(condition, 0)

    return df


print('(train) max_traffic_source sample searching...')

df_train = assign_traffic_source_and_recommendation(df_train, countAggregSub_rec, successRateSub_rec)

print('(train) not recomended Agregs searching...')

df_train = assign_not_recommended(df_train, countAggregSub_not_rec, successRateSub_not_rec)

#print(df_train, '\n')

#считаем агрегаты с одинаковым cutted_zip_code
df_test['countAggreg'] = df_test.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'])['LEAD_ID'].transform('count')
#считаем, сколько удач (Submission-ов) для каждого агрегата
df_test['countSub'] = df_test.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'])['IS_SUBMITTED'].transform(lambda x: x.sum())

#вычисляем конверсию на агрегате
df_test['successRateSub'] = df_test['countSub']/df_test['countAggreg']

#функция для нахождения наилучшего source, удовлетворяющего критериям.
#если таковых нет - 0

print('(test) max_traffic_source sample searching...')

df_test = assign_traffic_source_and_recommendation(df_test, countAggregSub_rec, successRateSub_rec)

print('(test) not recomended Agregs searching...')

df_test = assign_not_recommended(df_test, countAggregSub_not_rec, successRateSub_not_rec)

print('Writing to files...')

df_test.to_excel('test.xlsx', index=False)
df_train.to_excel('train.xlsx', index=False)
input('Test and train files were successfully generated, press ENTER to exit\n')
#print(df_test, '\n')
#print(df_train)
