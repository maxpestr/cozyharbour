import PySimpleGUI as sg
import openpyxl
import pandas as pd
import numpy as np


def check_digit(string):
    if string.isdigit():
        return True
    else:
        try:
            float(string)
            return True
        except ValueError:
            return False

def get_parameters():
    default_settings = {}
    try:
        with open('params.txt', 'r') as file:
            for line in file:
                if len(line) > 3:
                    key, value = line.strip().split('::')
                    default_settings[key.strip()] = value.strip()
    except FileNotFoundError:
        pass  # File doesn't exist, default_settings will remain empty
    column1 = [
        [sg.Text('Enter the value of parameter a')],
        [sg.InputText(default_text=default_settings.get('a', '2'), key='a')],
        [sg.Text('Enter the value of parameter b')],
        [sg.InputText(default_text=default_settings.get('b', '-1'), key='b')],
        [sg.Text('Enter the value of parameter omega')],
        [sg.InputText(default_text=default_settings.get('omega', '1'), key='omega')],
        [sg.Text('Enter the value of parameter q')],
        [sg.InputText(default_text=default_settings.get('q', '-1'), key='q')],
        [sg.Text('Enter the value of parameter r')],
        [sg.InputText(default_text=default_settings.get('r', '-1'), key='r')],
        [sg.Text('Enter the value of parameter s')],
        [sg.InputText(default_text=default_settings.get('s', '-1'), key='s')],
        [sg.Text('Enter the probability of randomly creating a new cluster during transfer')],
        [sg.InputText(default_text=default_settings.get('alpha', '0.2'), key='alpha')],
        [sg.Text('Enter the probability of corrective translation')],
        [sg.InputText(default_text=default_settings.get('tp', '0.5'), key='tp')],
        [sg.Text('Enter the number of series')],
        [sg.InputText(default_text=default_settings.get('ser', '10'), key='ser')],
        [sg.Text('Enter the maximum series length')],
        [sg.InputText(default_text=default_settings.get('maxlen', '5'), key='maxlen')],
        [sg.Text('Enter the power parameter before p')],
        [sg.InputText(default_text=default_settings.get('p_tf', '-0.5'), key='p_tf')],
        [sg.Text('Save file in xls(x) or csv(c)?')],
        [sg.InputText(default_text=default_settings.get('s1', 'x'), key='s1')],
        [sg.Text(' Длина скользящего окна:')],
        [sg.InputText(default_text=default_settings.get('n_lust', '10'), key='n_lust')],
        [sg.Text('Enter the value of "величина падения"')],
        [sg.InputText(default_text=default_settings.get('r_threshold', '5'), key='r_threshold')],
        [sg.Text('Enter the value of "верхняя граница"')],
        [sg.InputText(default_text=default_settings.get('up_lim', '20'), key='up_lim')],
    ]

    column2=[
        [sg.Text('Введите значение aver_sum_c_thresh:')],
        [sg.InputText(default_text=default_settings.get('aver_sum_c_thresh', '0'), key='a_th')],
        [sg.Text('Введите значение awer_vol_c_term_thresh:')],
        [sg.InputText(default_text=default_settings.get('awer_vol_c_term_thresh', '0'), key='o_th')],
        [sg.Text('Введите значение boundary_point_num_term_thresh:')],
        [sg.InputText(default_text=default_settings.get('boundary_point_num_term_thresh', '0'), key='b_th')],
        [sg.Text('Введите значение nearest_dist_bond_term_thresh:')],
        [sg.InputText(default_text=default_settings.get('nearest_dist_bond_term_thresh', '0'), key='n_th')],
        [sg.Text('Введите значение farthest_dist_bond_term_thresh:')],
        [sg.InputText(default_text=default_settings.get('farthest_dist_bond_term_thresh', '0'), key='f_th')],
        [sg.Button('Submit')]
    ]
    
    layout = [
        [sg.Column(column1), sg.VSeperator(), sg.Column(column2)],
        #[sg.Button('Submit')]
    ]

    window = sg.Window('Enter Parameters', layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Submit':
            break

    window.close()
    with open('params.txt', 'w') as file:
        for key, value in values.items():
            file.write(f"{key} :: {value}\n")
    return values

def process_data(input_file, output_file):
    wb = openpyxl.Workbook()
    ws = wb.active

    # Озаглавливаем колонки
    headers = {
        "NUM_SER": int,
        "CORRECTION_TYPE": str,
        "point_to_trans_num": int,
        "boundary_points_number": int,
        "bpn^p": float,
        "New_TF": float,
        "Old_TF": float,
        "FINAL_STATUS": str,
        "mult.term ^ q": float,
        "mean_near ^ r": float,
        "fath_term ^ s" : float,
        "pre_tf+": float,
        "pre_tf-": float,
        "is_new": str,
        "clust_trans": int,
        "trans_points_nums": str,
        "real ser. length": int,
        "is_start_search": str,
        "is_end_search": str,
        "abs_delta_tf": float,
        "curr_delta_tf": float
    }
    ws.append(list(headers.keys()))

    # Открываем txt файл для чтения
    with open(input_file, 'r') as file:
        lines = file.readlines()

    series_data = {}  # Словарь для хранения данных текущей серии

    for line in lines:
        line = line.strip()  # Убираем пробельные символы с начала и конца строки
        if line.startswith("!!!!"):  # Обнаружен конец блока, записываем данные в Excel
            row_data = [series_data.get(header, '') for header in headers.keys()]
            for i, value in enumerate(row_data):
                header = list(headers.keys())[i]
                try:
                    row_data[i] = headers[header](value)
                    if headers[header] == float:
                        row_data[i] = round(float(value), 6)
                except ValueError:
                    pass
            ws.append(row_data)
            series_data = {}  # Очищаем данные для нового блока
        elif line.startswith("NUM_SER"):
            series_data['NUM_SER'] = line.split()[1]
        elif line.startswith("CORRECTING") or line.startswith("SIMPLE"):
            series_data['CORRECTION_TYPE'] = line.split()[0]
        elif line.startswith("point to trans num"):
            series_data['point_to_trans_num'] = line.split()[-1]
        elif line.startswith("bpn^"):
            series_data['bpn^p'] = line.split('=')[1].strip()
        elif line.startswith("boundary_points_number"):
            series_data['boundary_points_number'] = line.split('=')[1].strip()
        elif line.startswith("New_TF"):
            series_data['New_TF'] = line.split('=')[1].strip()
        elif line.startswith("Old_TF"):
            series_data['Old_TF'] = line.split('=')[1].strip()
        elif line.startswith("CANCELED") or line.startswith("ACCEPTED"):
            series_data['FINAL_STATUS'] = line.strip()
        elif line.startswith("mult.term"):
            series_data['mult.term ^ q'] = line.split('=')[1].strip()
        elif line.startswith("mean_near"):
            series_data['mean_near ^ r'] = line.split('=')[1].strip()
        elif line.startswith("farth"):
            series_data['fath_term ^ s'] = line.split('=')[1].strip()
        elif line.startswith("is_new?"):
            series_data['is_new'] = line.split('?')[1].strip()        
        elif line.startswith("pre_tf+"):
            series_data['pre_tf+'] = line.split('=')[1].strip()
        elif line.startswith("pre_tf-"):
            series_data['pre_tf-'] = line.split('=')[1].strip()
        elif line.startswith("diff_b"):
            series_data['diff_b'] = line.split('=')[1].strip()
        elif line.startswith("clust_trans"):
            series_data['clust_trans'] = line.split(':')[1].strip()
        elif line.startswith("trans_points_nums"):
            series_data['trans_points_nums'] = line.split(':')[1].strip()    
        elif line.startswith("real ser. length"):
            series_data['real ser. length'] = line.split(':')[1].strip()
        elif line.startswith("slow growth"):
            series_data['is_start_search'] = line.split('!')[1].strip()
        elif line.startswith("abs_delta_tf"):
            series_data['abs_delta_tf'] = line.split('!')[1].strip()
        elif line.startswith("curr_delta_tf"):
            series_data['curr_delta_tf'] = line.split('!')[1].strip()
        elif line.startswith("new params"):
            series_data['is_end_search'] = line.split('!')[1].strip()

    # Сохраняем xlsx файл
    wb.save(output_file)   

def write_to_excel(ws, series_data):
    row = [series_data.get(header, "") for header in ws[1]]  # Получаем значения для каждой колонки
    ws.append(row)  # Добавляем значения в новую строку

def get_user_options():
    layout = [
        [sg.Text('Использовать внешний excel-файл для построения графа?')],
        [sg.Radio('Да', "RADIO1", default=True, key='t_g_option'),
         sg.Radio('Нет', "RADIO1", default=False)],
        
        [sg.Text('Использовать интеллектуальный поиск при медленном росте?')],
        [sg.Radio('Да', "RADIO2", default=True, key='int_search_option'),
         sg.Radio('Нет', "RADIO2", default=False)],
        
        [sg.Text('Использовать геометрическое условие в nearest_term?')],
        [sg.Radio('Да', "RADIO3", default=True, key='check_geom_option'),
         sg.Radio('Нет', "RADIO3", default=False)],
        
        [sg.Submit()]
    ]

    window = sg.Window('Настройки', layout)


    while True:
        event, values = window.read()
        if event in (None, 'Exit', 'Cancel'):
            break
        if event == 'Submit':
            tg_option = 1 if values['t_g_option'] else 0
            int_search_option = 1 if values['int_search_option'] else 0
            check_geom_option = 1 if values['check_geom_option'] else 0
            break

    window.close()
    return tg_option, int_search_option, check_geom_option
     
def assign_C_roles_ui(df):
    c_values = [(0,0)]
    numeric_cols = [col for col in df.columns if col not in ['Clust_ID', 'ID'] and pd.api.types.is_numeric_dtype(df[col])]
    
    # Create layout for the window
    layout = [
        [sg.Text('Select columns to mark as C and enter C-value or select option for each column:')],
        [sg.Column([[sg.Checkbox(col, key=f'C_{col}'),
                     # Input for C-values or option for 'use_clust_mean'
                     sg.InputText(default_text='0.0', key=f'C_value_{col}'),
                     sg.Checkbox('use_clust_mean', key=f'use_clust_mean_{col}')] for col in numeric_cols])],
        [sg.Checkbox('Use clust sum?', key='length_sum_selected')],  # Checkbox for 'Use clust sum?'
        [sg.InputText(default_text='0.0', key='c_sum_value')],  # Input for 'Use clust sum?' value
        [sg.Checkbox('use_mean', key=f'use_mean1')],
        [sg.Button('Assign Roles')]
    ]

    # Create the window
    window = sg.Window('Assign Column Roles (C)', layout)

    # Event loop for assigning roles C
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        if event == 'Assign Roles':
            for col in numeric_cols:
                if values[f'C_{col}']:
                    new_col_name = col + '_CC'
                    df.rename(columns={col: new_col_name}, inplace=True)
                    c_value = float(values[f'C_value_{col}']) if values[f'C_value_{col}'] else None
                    use_clust_mean = values[f'use_clust_mean_{col}']
                    c_values.append([c_value, use_clust_mean])
            c_sum = float(values.get('c_sum_value', 0.0))
            use_clust_mean = values['use_mean1']
            c_values.append([c_sum,use_clust_mean])
            break

    window.close()
    if values['length_sum_selected']:
        return 1, c_values
    else:
        return 0, c_values

def assign_column_roles_ui(df):
    # Identify numeric columns except for clust_id and ID
    numeric_cols = [col for col in df.columns if col not in ['Clust_ID', 'ID'] and pd.api.types.is_numeric_dtype(df[col])]

    # Create a layout for the window
    layout = []
    drs=[]
    for col in numeric_cols:
        if col.startswith('feature'):
            default_role = 'F +'
        elif col.startswith('X'):
            default_role='X'
        else:
            default_role='S'
        layout.append([sg.Text(f"Assign role to numeric column '{col}':"), 
                       sg.Radio('F +', group_id=col, key=f'F1_{col}', default=default_role),
                       sg.Radio('F -', group_id=col, key=f'F2_{col}', default=default_role),
                       sg.Radio('X and F+', group_id=col, key=f'XF1_{col}', default=default_role),
                       sg.Radio('X and F-', group_id=col, key=f'XF2_{col}', default=default_role),
                       sg.Radio('X', group_id=col, key=f'X_{col}', default=default_role),
                       sg.Radio('S', group_id=col, key=f'S_{col}', default=default_role)])

    layout.append([sg.Button('Next')])

    # Create the window
    window = sg.Window('Assign Column Roles (excluding C)', layout)

    # Event loop for the first part (assigning roles except C)
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        if event == 'Next':
            roles = {}
            for col in numeric_cols:
                for role in ['F1', 'F2', 'XF1', 'XF2', 'X', 'S']:
                    if values[f'{role}_{col}']:
                        new_col_name = role + '_' + col
                        roles[col] = new_col_name
            # Update column names based on user-assigned roles
            df.rename(columns=roles, inplace=True)
            break

    window.close()


def clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')    

def print_iteration(iteration, n):
    clear_screen()
    print(f"Текущая итерация: {iteration} из {n}")

def connected_components_autosplit_ui():
    # Create layout for the window
    layout = [
        [sg.Checkbox('Use connected_components_autosplit?', key='fc')],  # Checkbox for 'Use connected_components_autosplit?'
        [sg.Button('Submit')]
    ]

    # Create the window
    window = sg.Window('Сonnected components autosplit', layout)

    # Event loop for assigning roles C
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        if event == 'Submit':
            fc = values['fc']
            break

    window.close()
    return 1 if fc else 0
