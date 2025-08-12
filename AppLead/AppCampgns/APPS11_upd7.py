# -*- coding: utf-8 -*-
#Загрузка файла
import warnings
import pandas as pd
from pyexcelerate import Workbook

warnings.filterwarnings('ignore')
#Check

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
INCLUDE_IN_SET_RATE_CONSIDER = params['INCLUDE_IN_SET_RATE_CONSIDER']
zip_length = params['zip_length']
countAggreg_rec = params['countAggreg_rec']
successRate_rec = params['successRate_rec']
countAggreg_not_rec = params['countAggreg_not_rec']
successRate_not_rec = params['successRate_not_rec']
duplicate_agreg_campaign_drop = params['duplicate_agreg_campaign_drop']

# Проверка значений
print("INCLUDE_IN_SET_RATE_CONSIDER: ", INCLUDE_IN_SET_RATE_CONSIDER,"\n")
print("zip_length: ", zip_length)
print("countAggreg_rec: ", countAggreg_rec)
print("successRate_rec:" , successRate_rec)
print("countAggreg_not_rec: ", countAggreg_not_rec)
print("successRate_not_rec: ", successRate_not_rec)
print("duplicate_agreg_campaign_drop: ", duplicate_agreg_campaign_drop)

#from openpyxl import load_workbook
#filename = '_отсорт-preleads_20240726.xlsx'
filename = input('Enter data filename\n')
print('Loading data file...')
colslist = ['STATE', 'COUNTY', 'CAMPAIGN', 'SERVICE', 'TRAFFIC_SOURCE', 'LEAD_ID', 'ZIP_CODE', 'IS_APPOINTMENT_SET', 'DISTRIBUTION_ALGORITHM_VERSION', 'COMPETING_CAMPAIGNS', 'IS_SUBMITTED']
df = pd.read_excel(filename)#, usecols=colslist)
print('Processing...')

#UPDATED FILTERING
df = df.loc[(df['CAMPAIGN_CURRENT_TYPE'] == 'Appointments') | (df['CAMPAIGN_CURRENT_TYPE'] == 'Leads')]
df = df.dropna(subset=['ZIP_CODE', 'SERVICE', 'TRAFFIC_SOURCE'])

'''
#Задаем параметры
zip_length = 3
countAggreg_rec = 7
successRate_rec = 0.25
countAggreg_not_rec = 15
successRate_not_rec = 0.1
'''

#Обрезаем zip коды
df['cutted_zip_code'] = df['ZIP_CODE'].astype(str).str[:zip_length]
#считаем агрегаты с одинаковым cutted_zip_code
df['countAggreg'] = df.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'])['LEAD_ID'].transform('count')
#считаем, сколько удач (appointment-ов) для каждого агрегата
df['countApp'] = df.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'])['IS_APPOINTMENT_SET'].transform(lambda x: x.sum())

#вычисляем конверсию на агрегате
df['successRate'] = df['countApp']/df['countAggreg']
'''
#функция для нахождения наилучшего source, удовлетворяющего критериям.
#если таковых нет - 0
def choose_max_traffic_source(group):

    filtered_group = group[(group['countAggreg'] >= countAggreg_rec) & (group['successRate'] >= successRate_rec)]
    if not filtered_group.empty:
        max_index = filtered_group['successRate'].idxmax()
        max_traffic_source = filtered_group.loc[max_index, 'TRAFFIC_SOURCE']
    else:
        max_traffic_source = 0
    group['max_traffic_source'] = max_traffic_source
    return group['max_traffic_source']

max_traffic_source = df.groupby(['SERVICE', 'cutted_zip_code'], group_keys=False).apply(choose_max_traffic_source)#, include_groups=False)

# Заполнение столбца Recomendation
df['Recomendation'] = df.apply(
    lambda row: max_traffic_source[row.name] if row['successRate'] < successRate_rec else 0,
    axis=1
)

#функция для нахождения агрегатов под удаление
def notRecomended(group):
    filtered_group = group[(group['countAggreg'] > countAggreg_not_rec) & (group['successRate'] < successRate_not_rec) & (group['Recomendation'] == 0)]
    if not filtered_group.empty:
        answer = 'remove'
    else:
        answer = 0
    group['notRecomended'] = answer
    return group

df = df.groupby(['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code'], group_keys=False).apply(notRecomended)#,  include_groups=False)
'''
#Генерирует уникальный ID группы на основе SERVICE, TRAFFIC_SOURCE и cutted_zip_code
unique_IDs = df[['SERVICE', 'TRAFFIC_SOURCE', 'cutted_zip_code']].drop_duplicates().to_numpy().tolist()

def create_group_ID(row):
    #group_key = unique_IDs.index([row['SERVICE'], row['TRAFFIC_SOURCE'], row['cutted_zip_code']])
    group_key = str(row['SERVICE'])[7:]+'_'+str(row['TRAFFIC_SOURCE'])[6:]+'_'+str(row['cutted_zip_code'])
    return group_key

df['Agreg_ID'] = df.apply(create_group_ID, axis = 1)


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
    qualified = df[(df['countAggreg'] >= countAggreg_rec) & 
                   (df['successRate'] >= successRate_rec)]
    
    # Индексы с максимальным successRate по группам
    idx = qualified.groupby(['SERVICE', 'cutted_zip_code'])['successRate'].idxmax()
    
    # Сопоставление: (SERVICE, ZIP) → TRAFFIC_SOURCE
    max_ts_series = df.loc[idx, ['SERVICE', 'cutted_zip_code', 'TRAFFIC_SOURCE']] \
        .set_index(['SERVICE', 'cutted_zip_code'])['TRAFFIC_SOURCE']
    
    # Присваиваем max_traffic_source по индексам
    df['max_traffic_source'] = df.set_index(['SERVICE', 'cutted_zip_code']) \
        .index.map(max_ts_series.to_dict()).fillna(0)
    
    # Рекомендация, если успех ниже порога
    df['Recomendation'] = df['max_traffic_source'].where(
        df['successRate'] < successRate_rec, 
        0
    )

    return df

#функция для нахождения агрегатов под удаление
def assign_not_recommended(df, countAggreg_not_rec, successRate_not_rec):
    """
    Отмечает строки как 'remove', если они удовлетворяют условиям:
    - countAggreg > порога
    - successRate < порога
    - Recomendation == 0

    Все остальные строки получают 0 в колонке 'notRecomended'.
    """
    condition = (
        (df['countAggreg'] > countAggreg_not_rec) &
        (df['successRate'] < successRate_not_rec) &
        (df['Recomendation'] == 0)
    )

    df['notRecomended'] = pd.Series('remove', index=df.index).where(condition, 0)

    return df


print('max_traffic_source sample searching...')

df = assign_traffic_source_and_recommendation(df, countAggreg_rec, successRate_rec)

print('not recomended Agregs searching...')

df = assign_not_recommended(df, countAggreg_not_rec, successRate_not_rec)


print('Agregates metrics are completed, now evaluating for campaigns')
"""---

---

Теперь приступаем к кампаниям. Расшифровка аббревиатур:

CRA - campaign rate app
CCM - conv.campaign_metric
C_ij, A_ij - думаю, ясно
D_ij - знаменатель дроби для взвешенного среднего
CAMPAIGN_success - количество appointmentod (вообще) для каждой кампании
CAMPAIGN_count - сколько раз встречается каждая кампания всего
"""

if not INCLUDE_IN_SET_RATE_CONSIDER:
    df['IS_SUBMITTED_C'] = df['IS_SUBMITTED']
    df['IS_APPOINTMENT_SET_C'] = df['IS_APPOINTMENT_SET']
else:
    df['IS_SUBMITTED_C'] = df['IS_SUBMITTED']*df['INCLUDE_IN_SET_RATE']
    df['IS_APPOINTMENT_SET_C'] = df['IS_APPOINTMENT_SET']*df['INCLUDE_IN_SET_RATE']

# Подсчет значений Campaign
campaign_counts = df[df['IS_SUBMITTED_C'] == True].groupby('CAMPAIGN')['DISTRIBUTION_ALGORITHM_VERSION'].count().reset_index()
campaign_counts.rename(columns={'DISTRIBUTION_ALGORITHM_VERSION': 'countCampaign'}, inplace=True)
'''
В подсчет идут не все campaign а только у которых верно IS_SUBMITTED
-----СДЕЛАНО
'''

df['countAggreg'] = df.groupby(['Agreg_ID'])['Agreg_ID'].transform('count')
#df['countzip'] = df.groupby('Agreg_ID').size().reset_index(name='countzip')['countzip']

#Считаем общее количество для каждого CAMPAIGN'а
CAMPAIGN_count = df.groupby(['CAMPAIGN'])['IS_SUBMITTED_C'].transform(lambda x: x.sum())
'''
В подсчет идут не все campaign а только у которых верно IS_SUBMITTED
------СДЕЛАНО
'''

CAMPAIGN_success = df.groupby(['CAMPAIGN'])['IS_APPOINTMENT_SET_C'].transform(lambda x: x.sum())
'''
ТУТ МЕНЯТЬ НЕ НАДО, т.к. IS_APPOINTMENT_SET => IS_SUBMITTED
'''

df['CRA'] = CAMPAIGN_success / CAMPAIGN_count
'''
CAMPAIGN_count считается уже по-другому (------СДЕЛАНО):
В подсчет идут не все campaign а только у которых верно IS_SUBMITTED
'''
print('Campaign rate app completed')

# Подсчет значений True в 'IS_SUBMITTED' с группировкой по 'STATE', 'COUNTY', 'TRAFFIC_SOURCE', 'cutted_zip_code'
submissions_counts = df.groupby(['Agreg_ID'])['IS_SUBMITTED_C'].sum().reset_index()
submissions_counts.rename(columns={'IS_SUBMITTED_C': 'countAggreg_Submitted'}, inplace=True)

# Объединение с исходным DataFrame на основе 'STATE', 'COUNTY', 'TRAFFIC_SOURCE', 'cutted_zip_code'
df = df.merge(submissions_counts, on=['Agreg_ID'], how='left')



df['CCM'] = df[df['IS_SUBMITTED_C'] == True].groupby(['Agreg_ID'])['CRA'].transform(lambda x: x.sum()) / df[df['IS_SUBMITTED_C'] == True]['countAggreg_Submitted']
'''
В знаменатель и числитель дроби идут только записи с IS_SUBMITTED_TRUE, например только агрегаты с IS_SUBMITTED = True идут в новый CountAggreg
'''
print('conv.campaign_metric completed')

df['C_ij'] = df[df['IS_SUBMITTED_C'] == True].groupby(['Agreg_ID', 'CAMPAIGN'])['Agreg_ID'].transform('count')
'''
---------Снова считаем только по IS_SUBMITTED = True
'''

df['A_ij'] = df.groupby(['CAMPAIGN', 'Agreg_ID'])['IS_APPOINTMENT_SET_C'].transform(lambda x: x.sum()) / df['C_ij']
'''
IS_APPOINTMENT_SET => IS_Submitted поэтому ничего не надо менять
'''

#Вычисляем знаменатель
df['D_ij'] = df['C_ij'] * abs(df['A_ij'] - df['CRA'])
#И итоговое вычисление
df['weighted average deviation'] = df[df['IS_SUBMITTED_C'] == True].drop_duplicates(subset=['Agreg_ID', 'CAMPAIGN'], keep='last').groupby(['CAMPAIGN'])['D_ij'].transform(lambda x: x.sum()) / CAMPAIGN_count
print('weighted average deviation completed')
#df['weighted average deviation'] = df.groupby(['Agreg_ID', 'CAMPAIGN'])['weighted average deviation'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['weighted average deviation'] = df[df['IS_SUBMITTED_C'] == True].groupby(['Agreg_ID', 'CAMPAIGN'])['weighted average deviation'].transform(lambda x: x.fillna(x.mean()))
#??????
# Подсчет значений Campaign
campaign_counts = df[df['IS_SUBMITTED_C'] == True].groupby('CAMPAIGN')['DISTRIBUTION_ALGORITHM_VERSION'].count().reset_index()
campaign_counts.rename(columns={'DISTRIBUTION_ALGORITHM_VERSION': 'countCampaign'}, inplace=True)
'''
-------countCampaign теперь тоже только с IS_SUBMITTED
'''
# Объединение с исходным DataFrame
df = df.merge(campaign_counts, on = ['CAMPAIGN'], how='left')

# Подсчет суммы всех значений COMPETING_CAMPAIGNS
competing_score_all = df[df['IS_SUBMITTED_C']== True].groupby('CAMPAIGN')['COMPETING_CAMPAIGNS'].sum().reset_index()
competing_score_all.rename(columns={'COMPETING_CAMPAIGNS': 'competingScoreAll'}, inplace=True)

df = df.merge(competing_score_all, on = ['CAMPAIGN'], how='left')

df['Aver_campaign_competing'] = df['competingScoreAll'] / df['countCampaign']
print('Aver_campaign_competin completed')
# Создание столбца successSub
#df['successSub'] = df['countSubmitted'] / df['countAggreg']#/ df['countZip']
'''
теперь successSub - НЕ НУЖЕНы
------УБРАЛ
'''

# Подсчет значений True в 'IS_APPOINTMENT_SET' с группировкой по 'Campaign'
appointment_counts = df.groupby(['CAMPAIGN'])['IS_APPOINTMENT_SET_C'].sum().reset_index()
appointment_counts.rename(columns={'IS_APPOINTMENT_SET_C': 'countCampaignApp'}, inplace=True)

# Объединение с исходным DataFrame на основе 'Campaign'
df = df.merge(appointment_counts, on=['CAMPAIGN'], how='left')

# Создание столбца CampaignRate
df['CampaignRateApp'] = df['countCampaignApp'] / df['countCampaign']
'''
    ТЕПЕРЬ СРЕДИ SUBMITTED
    '''
print('CampaignRateApp completed')
# Подсчет значений True в 'IS_SUBMITTED' с группировкой по 'Campaign'
#appointment_counts = df.groupby(['CAMPAIGN'])['IS_SUBMITTED'].sum().reset_index()
#appointment_counts.rename(columns={'IS_SUBMITTED': 'countCampaignSub'}, inplace=True)
'''
У нас теперь все IS_SUBMITTED
'''
'''
IS_SUBMITTED - ысе рассматриваемое множество
ренейминг в новой версии видимо не нужен
'''
# Объединение с исходным DataFrame на основе 'Campaign'
#df = df.merge(appointment_counts, on=['CAMPAIGN'], how='left')

# Создание столбца CampaignRate
#df['CampaignRateSub'] = df['countCampaignApp'] / df['countCampaign']
'''
    ТЕПЕРЬ count Appointment СРЕДИ SUBMITTED
'''

df.drop(['C_ij', 'A_ij', 'D_ij'], axis=1, inplace=True)

df['CRA'] = df['CRA'].fillna(-1)

print('campaign rate app final analysis completed')

df.rename(columns={'CRA': 'campaign rate app'}, inplace=True)
'''
def fill_ccm(row):
    # Проверяем, если значение в 'CCM' NaN
    if pd.isna(row['CCM']):
        # Находим первую строку с таким же 'Agreg_ID', где 'CCM' не NaN
        matching_value = df[df['Agreg_ID'] == row['Agreg_ID']]['CCM'].dropna().iloc[0] if not df[df['Agreg_ID'] == row['Agreg_ID']]['CCM'].dropna().empty else -1
        return matching_value
    else:
        return row['CCM']

df['CCM'] = df.apply(fill_ccm, axis=1)
'''
# Составляем словарь для Agreg_ID и соответствующих значений CCM
ccm_dict = df.dropna(subset=['CCM']).groupby('Agreg_ID')['CCM'].first().to_dict()

# Теперь для каждой строки проверяем, если CCM == NaN, то присваиваем значение из словаря
df['CCM'] = df['Agreg_ID'].map(ccm_dict).fillna(-1)

df.rename(columns={'CCM': 'conv.campaign_metric'}, inplace=True)

print('conv.campaign_metric final analysis completed')

# Составляем словарь для CAMPAIGN и соответствующих значений weighted average deviation
wad_dict = df.dropna(subset=['weighted average deviation']).groupby('CAMPAIGN')['weighted average deviation'].first().to_dict()

# Теперь для каждой строки проверяем, если CCM == NaN, то присваиваем значение из словаря
df['weighted average deviation'] = df['CAMPAIGN'].map(wad_dict).fillna(-1)

# Составляем словарь для CAMPAIGN и соответствующих значений count campaign
cc_dict = df.dropna(subset=['countCampaign']).groupby('CAMPAIGN')['countCampaign'].first().to_dict()

# Теперь для каждой строки проверяем, если CCM == NaN, то присваиваем значение из словаря
df['countCampaign'] = df['CAMPAIGN'].map(cc_dict).fillna(-1)

# Составляем словарь для CAMPAIGN и соответствующих значений competingScoreAll
csa_dict = df.dropna(subset=['competingScoreAll']).groupby('CAMPAIGN')['competingScoreAll'].first().to_dict()

# Теперь для каждой строки проверяем, если CCM == NaN, то присваиваем значение из словаря
df['competingScoreAll'] = df['CAMPAIGN'].map(csa_dict).fillna(-1)

# Составляем словарь для CAMPAIGN и соответствующих значений Average campaign competing
acc_dict = df.dropna(subset=['Aver_campaign_competing']).groupby('CAMPAIGN')['Aver_campaign_competing'].first().to_dict()

# Теперь для каждой строки проверяем, если CCM == NaN, то присваиваем значение из словаря
df['Aver_campaign_competing'] = df['CAMPAIGN'].map(acc_dict).fillna(-1)

# Составляем словарь для CAMPAIGN и соответствующих значений CampaignRateApp
cra_dict = df.dropna(subset=['CampaignRateApp']).groupby('CAMPAIGN')['CampaignRateApp'].first().to_dict()

# Теперь для каждой строки проверяем, если CCM == NaN, то присваиваем значение из словаря
df['CampaignRateApp'] = df['CAMPAIGN'].map(cra_dict).fillna(-1)

#Новые константы
#threshold_Aver_campaign_competing = 1.5
#lower_threshold_for_successCompApp = 0.405
#возможно заменить на App 
#threshold_for_coeff_conv_campaign_metric = 0.9
'''
Три константы - в файл
'''

threshold_Aver_campaign_competing = params['threshold_Aver_campaign_competing']
threshold_for_coeff_conv_campaign_metric = params['threshold_for_coeff_conv_campaign_metric']

print("threshold_Aver_campaign_competing: ", threshold_Aver_campaign_competing)
print("threshold_for_coeff_conv_campaign_metric: ", threshold_for_coeff_conv_campaign_metric)

countAggregCompApp_rec = params['countAggregCompApp_rec']
successRateCompApp_rec = params['successRateCompApp_rec']
countAggregCompApp_not_rec = params['countAggregCompApp_not_rec']
successRateCompApp_not_rec = params['successRateCompApp_not_rec']

# Проверка значений
print("countAggregCompApp_rec: ", countAggregCompApp_rec)
print("successRateCompApp_rec:" , successRateCompApp_rec)
print("countAggregCompApp_not_rec: ", countAggregCompApp_not_rec)
print("successRateCompApp_not_rec: ", successRateCompApp_not_rec)


# Условие столбца competingAlgorithm_control
df['competingAlgorithm_control'] = (
    (df['countAggreg_Submitted'] > countAggregCompApp_rec) &
    (df['campaign rate app'] < successRateCompApp_rec) &
    (df['Aver_campaign_competing'] > threshold_Aver_campaign_competing)
).replace({True: 'Check competition algorithm', False: 0})

# Условие столбца transfer to Type "leads"
df['transfer_to_Type_leads'] = (
    (df['countAggreg_Submitted'] > countAggregCompApp_not_rec)&
    (df['campaign rate app'] < successRateCompApp_not_rec )
    ).replace({True: 'option to change current type to leads', False: 0})


# Условие столбца campaign_noneffectivity
df['campaign_noneffectivity'] = (
    (df['countAggreg_Submitted'] > countAggregCompApp_not_rec) &
    (df['campaign rate app'] < successRateCompApp_not_rec ) &
    (df['conv.campaign_metric'] < df['campaign rate app']*threshold_for_coeff_conv_campaign_metric)
).replace({True: 'The campaign is probably noneffective', False: 0})

df.drop(['IS_SUBMITTED_C', 'IS_APPOINTMENT_SET_C'], axis=1, inplace=True)

if duplicate_agreg_campaign_drop:
    df = df.drop_duplicates(subset=['Agreg_ID', 'CAMPAIGN'], keep='first')

file_path = input('Enter output filename\n')
print('Writing to file...')
'''
Столбец CountAggreg не меняется, просто фигурирует больше countSub
'''
values = [df.columns] + list(df.values)
wb = Workbook()
wb.new_sheet('Sheet1', data=values)
wb.save(file_path)
'''
df = df.drop(columns=colslist)

workbook = load_workbook(filename=filename)
sheet = workbook.active  # Выбираем активный лист

# Определяем, с какого столбца начать добавление новых данных
start_col = sheet.max_column + 1

for col_index, column_name in enumerate(df.columns, start=start_col):
        sheet.cell(row=1, column=col_index, value=column_name)

# Записываем новые столбцы в лист
for row_index, value in enumerate(df.values, start=2):  # начинаем с первой строки
    for col_index, item in enumerate(value, start=start_col):
        sheet.cell(row=row_index, column=col_index, value=item)

workbook.save(file_path)
workbook.close()
'''

#df_existing = pd.read_excel(filename)
#df_combined = pd.concat([df_existing, df], ignore_index=True)
#df.to_excel(file_path, index=False)

print('The result was successfully saved in '+file_path+'\n')
input('Press ENTER to exit\n')
