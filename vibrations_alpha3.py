# -*- coding: utf-8 -*-
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from pandas import DataFrame
from pandas import IndexSlice
from pandas.io.excel import read_excel
import os

f_tables='output_tables.txt'
f_out='output_vibr.txt'

#код для вывода ascii-таблиц

def print_pretty_table(data, cell_sep=' | ', header_separator=True):
    rows = len(data)
    cols = len(data[0])

    col_width = []
    for col in range(cols):
        columns = [data[row][col] for row in range(rows)]
        col_width.append(len(max(columns, key=len)))

    separator = "-+-".join('-' * n for n in col_width)

    for i, row in enumerate(range(rows)):
        if i == 1 and header_separator:
            print(separator)

        result = []
        for col in range(cols):
            item = data[row][col].rjust(col_width[col])
            result.append(item)

        print(cell_sep.join(result))
    print('\n')
    
def write_pretty_table_to_txt_to_excel(data,sheet_name='Sheet1',filename='output.xlsx'):
    rows = len(data)
    cols = len(data[0])

    col_width = []
    for col in range(cols):
        columns = [data[row][col] for row in range(rows)]
        col_width.append(len(max(columns, key=len)))

    df = pd.DataFrame(data, columns=[f'Column {i+1}' for i in range(cols)])
    df.to_excel(filename, sheet_name=sheet_name, index=False)

def write_pretty_table_to_txt(data, filename=f_tables, cell_sep=' | ', header_separator=True):
    rows = len(data)
    cols = len(data[0])

    col_width = []
    for col in range(cols):
        columns = [data[row][col] for row in range(rows)]
        col_width.append(len(max(columns, key=len)))

    separator = "-+-".join('-' * n for n in col_width)

    with open(filename, 'a') as file:
        for i, row in enumerate(range(rows)):
            if i == 1 and header_separator:
                file.write(separator + '\n')

            result = []
            for col in range(cols):
                item = data[row][col].rjust(col_width[col])
                result.append(item)

            file.write(cell_sep.join(result) + '\n')
        file.write('\n')

def write_to_file(data, filename=f_out, encoding = 'UTF-8'):
    with open(filename, 'a') as file:
        if isinstance(data, str):
            file.write(data)
        elif isinstance(data, int) or isinstance(data, float):
            file.write(str(data))
        else:
            raise ValueError("Unsupported data type. Only strings and numbers are supported.")

def get_okno_parameters():
    # Define the layout of the window

    layout = [
        [sg.Text("Скалярные параметры")],
        [sg.Text("M (количество моторных масс):"), sg.InputText("6", key="M")],
        [sg.Text("G (модуль сдвига):"), sg.InputText("8.3e10", key="G")],
        [sg.Text("dk (диаметр кривошипа):"), sg.InputText("5e-2", key="dk")],
        [sg.Text("J_M (момент инерции маховика демпфера):"), sg.InputText("1e-2", key="J_M")],
        [sg.Text("I (количество цилиндров):"), sg.InputText("8", key="I")],
        [sg.Text("Mvp (масса возвратно-поступательно движущихся элементов):"), sg.InputText("0.7", key="Mvp")],
        [sg.Text("R (радиус кривошипа):"), sg.InputText("4e-2", key="R")],
        [sg.Text("lambd (параметр подобия):"), sg.InputText("0.5", key="lambd")],
        [sg.Text("n_nom (номинальная частота, об/м):"), sg.InputText("1e4", key="n_nom")],
        [sg.Text("tau (тактность):"), sg.InputText("4", key="tau")],
        [sg.Text("a_coef (коэффициент a):"), sg.InputText("1", key="a_coef")],
        [sg.Text("b_coef (коэффициент b):"), sg.InputText("1", key="b_coef")],
        [sg.Text("c_coef (коэффициент c):"), sg.InputText("1", key="c_coef")],
        [sg.Text("eps (коэффициент демпфирования):"), sg.InputText("5", key="eps")],
        [sg.Text("n_min (минимальная частота, об/м):"), sg.InputText("6000", key="n_min")],
        [sg.Text("n_max (минимальная частота, об/м):"), sg.InputText("12000", key="n_max")],
        [sg.Text("V (объем цилиндра):"), sg.InputText("0.5e-3", key="V")],
        [sg.Text("Ne_nom (номинальная мощность):"), sg.InputText("1e5", key="Ne_nom")],
        [sg.Text("g_e (эффективный расход топлива):"), sg.InputText("0.5", key="g_e")],
        [sg.Text("Имя файла, содержащего J,C,Pg,M_n, Excel:")],
        [sg.InputText("testdata.xlsx", key="filename")],
        [sg.Text("Имя папки, в которую будут складываться результаты:")],
        [sg.Button("Отправить")],
    ]

    # Create the window
    window = sg.Window("Параметры двигателя", layout)

    # Display and interact with the window
    event, values = window.read()

    N=72 
    K=24
    # Get the input values
    M = int(values["M"])
    G = float(values["G"])
    dk = float(values["dk"])
    J_M = float(values["J_M"])
    I = int(values["I"])
    Mvp = float(values["Mvp"])
    R = float(values["R"])
    lambd = float(values["lambd"])
    n_nom = float(values["n_nom"])
    tau = int(values["tau"])
    a_coef = float(values["a_coef"])
    b_coef = float(values["b_coef"])
    c_coef = float(values["c_coef"])
    eps = float(values["eps"])
    n_min = float(values["n_min"])
    n_max = float(values["n_max"])
    V = float(values["V"])
    Ne_nom = float(values["Ne_nom"])
    g_e = float(values["g_e"])
    filename = values["filename"]
    
    #if not os.path.exists(outfolder):
    #    os.makedirs(outfolder)

    # Read the data from the Excel file
    J = read_excel(filename, sheet_name=0, header=None).iloc[:, 0] #.tolist()
    C = read_excel(filename, sheet_name=1, header=None).iloc[:, 0] #.tolist()
    P_g = read_excel(filename, sheet_name=2, header=None).iloc[:, 0] #.tolist()
    M_n = read_excel(filename, sheet_name=3, header=None).iloc[:, 0] #.tolist()
    # Close the window
    window.close()
    return N,K,M,G,dk,J_M,I,Mvp,R,lambd,n_nom,tau,a_coef,b_coef,c_coef,eps,n_min,n_max,V,Ne_nom,g_e,filename, J,C,P_g,M_n

N,K,M,G,dk,J_M,I,Mvp,R,lambd,n_nom,tau,a_coef,b_coef,c_coef,eps,n_min,n_max,V,Ne_nom,g_e,filename,J,C,Pg,M_n = get_okno_parameters()


#________________________________________________________________________________________#
# 1. Эквивалентная расчетная схема#
open(f_tables, "w")
open(f_out, "w")

J_p=np.pi*dk**4/32

print('Полярный момент инерции:\n')
write_to_file('Полярный момент инерции:\n')
print(J_p)
write_to_file(J_p)
print('\n')
write_to_file('\n')

str1=['i']+list(map(str, range(1,M+1)))

str2=['J_i,кг*м^2']+list(map(str, J.tolist()))

table_data=[str1,str2]
print_pretty_table(table_data)
write_pretty_table_to_txt(table_data)

# или такие варианты:
#write_pretty_table_to_txt_to_excel(table_data)
#write_pretty_table_to_file(data, filename='output.txt', cell_sep=' | ', header_separator=True):


l=np.zeros(M-1); #здесь будут длины l_i,i+1 между массами
#вычисляем длины
l=G*J_p/C

#таблица для C и l

str1=['i,i+1']
for i in range(1,M):
    nums=str(i)+','+str(i+1)
    str1.append(nums)

str2=['C_i,i+1, Н*м/рад']+list(map(str, np.round(C,5).tolist()))
str3=['l_i,i+1, м']+list(map(str, np.round(l,4).tolist()))

table_data=[str1,str2,str3]
print_pretty_table(table_data)
write_pretty_table_to_txt(table_data, f_tables)

#3. Частота собственных колебаний

J_sum=np.sum(J[0:M-1])

l_sum=np.sum(l[0:M-2])+l[M-2]

C_sum=G*J_p/l_sum

#Приближенное значение частоты
w_01=np.sqrt(C_sum*(J_sum+J_M)/(J_sum*J_M))

#Функция формирует массив a[i]
def find_a(w):
    a=np.ones(M)
    for i in range(1, M):
      aJ=a*J
      a[i]=a[i-1]-w**2*np.sum(aJ[0:i])/C[i-1]
    return a

find_a(w_01)

def min_to_find(w):
    Ja=J*find_a(w)
    return abs(np.sum(Ja)/(np.max(abs(Ja))))

min_to_find(w_01*2.1)

w_s=w_01
while (w_s>0 and min_to_find(w_s)>0.05):
    w_s=w_s-w_01*0.01

if min_to_find(w_s)>0.05:
      while (w_s<10*w_01 and min_to_find(w_s)>0.05):
          w_s=w_s+w_01*0.01

if min_to_find(w_s)>0.05:
    write_to_file('Частота собственных колебаний не найдена')

#Итоговая ошибка для собственных колебаний

min_to_find(w_s)

#Частота собственных колебаний

print('Частота собственных колебаний:\n')
print(str(round(w_s))+'\n')
write_to_file('Частота собственных колебаний:\n')
write_to_file(str(round(w_s))+'\n')

a=find_a(w_s) #Получаем массив a_i для частоты собственных колебаний

#Таблица для a_i
str1=['i']+list(map(str, range(1,M+1)))
str2=['a_i']+list(map(str, (np.round(a,3)).tolist()))
print_pretty_table([str1,str2])
write_pretty_table_to_txt([str1,str2])

#4 Главные и сильные гармоники
K_m_main_1=I*1; K_m_main_2=I*2; K_m_main_3=I*3; #Главные гармоники
K_m_strong_1=0.5*I*1; K_m_strong_2=0.5*I*2; K_m_strong_3=0.5*I*3; #Сильные гармоники

#5 Резонансные частоты
n_res_strong_1=30*w_s/(np.pi*K_m_strong_1)
n_res_main_1=30*w_s/(np.pi*K_m_main_1)

n_res_strong_2=30*w_s/(np.pi*K_m_strong_2)
n_res_main_2=30*w_s/(np.pi*K_m_main_2)

n_res_strong_3=30*w_s/(np.pi*K_m_strong_3)
n_res_main_3=30*w_s/(np.pi*K_m_main_3)

#Строим таблицы
data=[['K_m', 'n_res, об/м']]
data.append(['K_m_strong_1']+[str(round(n_res_strong_1))])
data.append(['K_m_main_1']+[str(round(n_res_main_1))])

data.append(['K_m_strong_2']+[str(round(n_res_strong_2))])
data.append(['K_m_main_2']+[str(round(n_res_main_2))])

data.append(['K_m_strong_3']+[str(round(n_res_strong_3))])
data.append(['K_m_main_3']+[str(round(n_res_main_3))])

print_pretty_table(data)
write_pretty_table_to_txt(data)

#6 Гармонический анализ крутящего момента

#Номинальные период и круговая частота
T=120/n_nom*tau/4
w_nom=2*np.pi/T

phi=np.linspace(10,720,72)
j=Mvp*R*w_nom**2*(np.cos(phi)+lambd*np.cos(2*phi))
beta=np.arcsin(lambd*np.sin(phi))

Pj=-Mvp*j
M_rot_g=Pg*np.sin(phi+beta)*R/np.cos(beta)
M_rot_j=Pj*np.sin(phi+beta)*R/np.cos(beta)

#Разложение в ряд (газовая составляющая)
Ag=np.zeros(K)
Bg=np.zeros(K)
for k in range(K):
    for i in range(N):
        Ag[k]+=2/N*M_rot_g[i]*np.cos((k+1)*2*np.pi*(i+1)/N)
        Bg[k]+=2/N*M_rot_g[i]*np.sin((k+1)*2*np.pi*(i+1)/N)

M_rot_g_sqr=np.sqrt(Ag**2+Bg**2)
phi_g=np.arctan(Ag/Bg)

#Средний крутящий газовый момент
M_rot_g_mean=1/N*sum(M_rot_g)

print('Средний крутящий газовый момент:\n')
print(M_rot_g_mean)
print('\n')
              
write_to_file('Средний крутящий газовый момент:\n')
write_to_file(M_rot_g_mean)
write_to_file('\n')

#Строим таблицу для газовой составляющей
data=[['k', 'М_кр_газ_k', 'phi_газ_k']]
for k in range(K):
    data.append([str(k+1), str(round(M_rot_g_sqr[k],2)), str(round(phi_g[k],2))])
print_pretty_table(data)
write_pretty_table_to_txt(data)

#Разложение в ряд (инерционная составляющая)
Aj=np.zeros(K)
Bj=np.zeros(K)
for k in range(K):
    for i in range(N):
        Aj[k]+=2/N*M_rot_j[i]*np.cos((k+1)*2*np.pi*(i+1)/N)
        Bj[k]+=2/N*M_rot_j[i]*np.sin((k+1)*2*np.pi*(i+1)/N)

M_rot_j_sqr=np.sqrt(Aj**2+Bj**2)
phi_j=np.arctan(Aj/Bj)

#Средний крутящий инерционный момент
M_rot_j_mean=1/N*sum(M_rot_j)

print('Средний крутящий инерционный момент:\n')
print(M_rot_j_mean)
print('\n')

write_to_file('Средний крутящий инерционный момент:\n')
write_to_file(M_rot_j_mean)
write_to_file('\n')

#Строим таблицу для инерционной составляющей
data=[['k', 'М_кр_инерц_k', 'phi_инерц_k']]
for k in range(K):
    data.append([str(k+1), str(round(M_rot_j_sqr[k],2)), str(round(phi_j[k],2))])
print_pretty_table(data)
write_pretty_table_to_txt(data)

#Среднее давление в рабочем диапазоне
n_diap=np.linspace(n_min, n_max, 10)

P_e=(30*tau/(V*I))*(Ne_nom/n_nom)*(a_coef+b_coef*n_diap/n_nom+c_coef*(n_diap/n_nom)**2)
P_e_nom=(30*tau/(V*I))*(Ne_nom/n_nom)*(a_coef+b_coef+c_coef)

M_rot_n_g_strong_1=M_rot_g_sqr[round(K_m_strong_1)]-1*P_e/P_e_nom
M_rot_n_g_strong_2=M_rot_g_sqr[round(K_m_strong_2)]-1*P_e/P_e_nom
M_rot_n_g_strong_3=M_rot_g_sqr[round(K_m_strong_3)-1]*P_e/P_e_nom

M_rot_n_g_main_1=M_rot_g_sqr[round(K_m_main_1)-1]*P_e/P_e_nom
M_rot_n_g_main_2=M_rot_g_sqr[round(K_m_main_2)-1]*P_e/P_e_nom
M_rot_n_g_main_3=M_rot_g_sqr[round(K_m_main_3)-1]*P_e/P_e_nom

#Строим таблицы

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_g']+np.round(M_rot_n_g_strong_1).astype(int).astype(str).tolist()]
data=str1+str2

print('Первая сильная гармоника\n')
print_pretty_table(data)
write_to_file('Первая сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_g']+np.round(M_rot_n_g_main_1).astype(int).astype(str).tolist()]
data=str1+str2

print('Первая главная гармоника\n')
print_pretty_table(data)
write_to_file('Первая главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_g']+np.round(M_rot_n_g_strong_2).astype(int).astype(str).tolist()]
data=str1+str2

print('Вторая сильная гармоника\n')
print_pretty_table(data)
write_to_file('Вторая сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_g']+np.round(M_rot_n_g_main_2).astype(int).astype(str).tolist()]
data=str1+str2

print('Вторая главная гармоника\n\n')
print_pretty_table(data)
write_to_file('Вторая главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_g']+np.round(M_rot_n_g_strong_3).astype(int).astype(str).tolist()]
data=str1+str2

print('Третья сильная гармоника\n\n')
print_pretty_table(data)
write_to_file('Третья сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_g']+np.round(M_rot_n_g_main_3).astype(int).astype(str).tolist()]
data=str1+str2

print('Третья главная гармоника\n\n')
print_pretty_table(data)
write_to_file('Третья главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

#Вычисляем гармоники инерционных составляющих
M_rot_n_j_strong_1=M_rot_j_sqr[round(K_m_strong_1)-1]*(n_diap/n_nom)**2
M_rot_n_j_strong_2=M_rot_j_sqr[round(K_m_strong_2)-1]*(n_diap/n_nom)**2
M_rot_n_j_strong_3=M_rot_j_sqr[round(K_m_strong_3)-1]*(n_diap/n_nom)**2

M_rot_n_j_main_1=M_rot_j_sqr[round(K_m_main_1)-1]*(n_diap/n_nom)**2
M_rot_n_j_main_2=M_rot_j_sqr[round(K_m_main_2)-1]*(n_diap/n_nom)**2
M_rot_n_j_main_3=M_rot_j_sqr[round(K_m_main_3)-1]*(n_diap/n_nom)**2

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_j']+np.round(M_rot_n_j_strong_1, 3).astype(str).tolist()]
data=str1+str2

print('Первая сильная гармоника\n')
print_pretty_table(data)
write_to_file('Первая сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_j']+np.round(M_rot_n_j_main_1, 3).astype(str).tolist()]
data=str1+str2

print('Первая главная гармоника\n')
print_pretty_table(data)
write_to_file('Первая главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_j']+np.round(M_rot_n_j_strong_2, 3).astype(str).tolist()]
data=str1+str2

print('Вторая сильная гармоника\n')
print_pretty_table(data)
write_to_file('Вторая сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_j']+np.round(M_rot_n_j_main_2, 3).astype(str).tolist()]
data=str1+str2

print('Вторая главная гармоника\n\n')
print_pretty_table(data)
write_to_file('Вторая главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_j']+np.round(M_rot_n_j_strong_3, 3).astype(str).tolist()]
data=str1+str2

print('Третья сильная гармоника\n\n')
print_pretty_table(data)
write_to_file('Третья сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['M_K_j']+np.round(M_rot_n_j_main_3, 3).astype(str).tolist()]
data=str1+str2

print('Третья главная гармоника\n\n')
print_pretty_table(data)
write_to_file('Третья главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)

#7. Амплитуда колебаний моторных масс
A1_strong_1=(M_rot_n_j_strong_1+M_rot_n_g_strong_1)*sum(a)/(eps*K_m_strong_1*(2*np.pi*n_res_strong_1/60)*sum(a**2))
A1_main_1=(M_rot_n_j_main_1+M_rot_n_g_main_1)*sum(a)/(eps*K_m_main_1*(2*np.pi*n_res_main_1/60)*sum(a**2))

A1_strong_2=(M_rot_n_j_strong_2+M_rot_n_g_strong_2)*sum(a)/(eps*K_m_strong_2*(2*np.pi*n_res_strong_2/60)*sum(a**2))
A1_main_2=(M_rot_n_j_main_2+M_rot_n_g_main_2)*sum(a)/(eps*K_m_main_2*(2*np.pi*n_res_main_2/60)*sum(a**2))

A1_strong_3=(M_rot_n_j_strong_3+M_rot_n_g_strong_3)*sum(a)/(eps*K_m_strong_3*(2*np.pi*n_res_strong_3/60)*sum(a**2))
A1_main_3=(M_rot_n_j_main_3+M_rot_n_g_main_3)*sum(a)/(eps*K_m_main_3*(2*np.pi*n_res_main_3/60)*sum(a**2))

#Строим таблицу для A_i

A_strong_1=np.zeros((M, 10))
for i in range(M):
    A_strong_1[i]=a[i]*A1_strong_1
data=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M):
    data.append(['A_strong_1['+str(i+1)+']']+np.round(A_strong_1[i], 6).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

A_main_1=np.zeros((M, 10))
for i in range(M):
    A_main_1[i]=a[i]*A1_main_1
data=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M):
    data.append(['A_main_1['+str(i+1)+']']+np.round(A_main_1[i], 6).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

A_strong_2=np.zeros((M, 10))
for i in range(M):
    A_strong_2[i]=a[i]*A1_strong_2
data=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M):
    data.append(['A_strong_2['+str(i+1)+']']+np.round(A_strong_2[i], 6).astype(str).tolist())
    
print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

A_main_2=np.zeros((M, 10))
for i in range(M):
    A_main_2[i]=a[i]*A1_main_2
data=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M):
    data.append(['A_main_2['+str(i+1)+']']+np.round(A_main_2[i], 6).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

A_strong_3=np.zeros((M, 10))
for i in range(M):
    A_strong_3[i]=a[i]*A1_strong_3
data=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M):
    data.append(['A_strong_3['+str(i+1)+']']+np.round(A_strong_3[i], 6).astype(str).tolist())
    
print_pretty_table(data)
write_pretty_table_to_txt(data)

write_to_file('\n', filename=f_tables)
A_main_3=np.zeros((M, 10))
for i in range(M):
    A_main_3[i]=a[i]*A1_main_3
data=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M):
    data.append(['A_main_3['+str(i+1)+']']+np.round(A_main_3[i], 6).astype(str).tolist())
    
print_pretty_table(data)
write_pretty_table_to_txt(data)

#Упругие моменты

M_upr_strong_1=np.zeros((M-1, 10))
for i in range(M-1):
    for j in range(10):
        M_upr_strong_1[i][j]=C[i]*(A_strong_1[i+1][j]-A_strong_1[i][j])
#Строим таблицу
data=[['M_upr']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M-1):
    data.append(['M_upr_strong_1['+str(i+1)+']']+np.round(M_upr_strong_1[i],2).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

M_upr_main_1=np.zeros((M-1, 10))
for i in range(M-1):
    for j in range(10):
        M_upr_main_1[i][j]=C[i]*(A_main_1[i+1][j]-A_main_1[i][j])
#Строим таблицу
data=[['M_upr']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M-1):
    data.append(['M_upr_main_1['+str(i+1)+']']+np.round(M_upr_main_1[i],2).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

M_upr_strong_2=np.zeros((M-1, 10))
for i in range(M-1):
    for j in range(10):
        M_upr_strong_2[i][j]=C[i]*(A_strong_2[i+1][j]-A_strong_2[i][j])
#Строим таблицу
data=[['M_upr']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M-1):
    data.append(['M_upr_strong_2['+str(i+1)+']']+np.round(M_upr_strong_2[i],2).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

M_upr_main_2=np.zeros((M-1, 10))
for i in range(M-1):
    for j in range(10):
        M_upr_main_2[i][j]=C[i]*(A_main_2[i+1][j]-A_main_2[i][j])
#Строим таблицу
data=[['M_upr']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M-1):
    data.append(['M_upr_main_2['+str(i+1)+']']+np.round(M_upr_main_2[i],2).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

M_upr_strong_3=np.zeros((M-1, 10))
for i in range(M-1):
    for j in range(10):
        M_upr_strong_3[i][j]=C[i]*(A_strong_3[i+1][j]-A_strong_3[i][j])
#Строим таблицу
data=[['M_upr']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M-1):
    data.append(['M_upr_strong_3['+str(i+1)+']']+np.round(M_upr_strong_3[i],2).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

M_upr_main_3=np.zeros((M-1, 10))
for i in range(M-1):
    for j in range(10):
        M_upr_main_3[i][j]=C[i]*(A_main_3[i+1][j]-A_main_3[i][j])
#Строим таблицу
data=[['M_upr']+np.round(n_diap).astype(int).astype(str).tolist()]
for i in range(M-1):
    data.append(['M_upr_main_3['+str(i+1)+']']+np.round(M_upr_main_3[i],2).astype(str).tolist())

print_pretty_table(data)
write_pretty_table_to_txt(data)
write_to_file('\n', filename=f_tables)

#Дополнительные касательные напряжения для каждой гармоники

max_abs_value = np.amax(np.abs(M_upr_strong_1))
max_abs_index = np.argmax(np.abs(M_upr_strong_1))
max_value = M_upr_strong_1.flatten()[max_abs_index]
max_value
write_to_file('Дополнительное касательное напряжение для первой сильной гармоники:')
write_to_file(round(max_value*16/(np.pi*dk**3)))
write_to_file('\n')

max_abs_value = np.amax(np.abs(M_upr_main_1))
max_abs_index = np.argmax(np.abs(M_upr_main_1))
max_value = M_upr_main_1.flatten()[max_abs_index]
max_value
write_to_file('Дополнительное касательное напряжение для первой главной гармоники:')
write_to_file(round(max_value*16/(np.pi*dk**3)))
write_to_file('\n')

max_abs_value = np.amax(np.abs(M_upr_strong_2))
max_abs_index = np.argmax(np.abs(M_upr_strong_2))
max_value = M_upr_strong_2.flatten()[max_abs_index]
max_value
write_to_file('Дополнительное касательное напряжение для второй сильной гармоники:')
write_to_file(round(max_value*16/(np.pi*dk**3)))
write_to_file('\n')

max_abs_value = np.amax(np.abs(M_upr_main_2))
max_abs_index = np.argmax(np.abs(M_upr_main_2))
max_value = M_upr_main_2.flatten()[max_abs_index]
max_value
write_to_file('Дополнительное касательное напряжение для второй главной гармоники:')
write_to_file(round(max_value*16/(np.pi*dk**3)))
write_to_file('\n')

max_abs_value = np.amax(np.abs(M_upr_strong_3))
max_abs_index = np.argmax(np.abs(M_upr_strong_3))
max_value = M_upr_strong_3.flatten()[max_abs_index]
max_value
write_to_file('Дополнительное касательное напряжение для третьей сильной гармоники:')
write_to_file(round(max_value*16/(np.pi*dk**3)))
write_to_file('\n')

max_abs_value = np.amax(np.abs(M_upr_main_3))
max_abs_index = np.argmax(np.abs(M_upr_main_3))
max_value = M_upr_main_3.flatten()[max_abs_index]
max_value
write_to_file('Дополнительное касательное напряжение для третьей главной гармоники:')
write_to_file(round(max_value*16/(np.pi*dk**3)))
write_to_file('\n')

#9 Потери мощности и перерасход топлива

N_vibr_strong_1=(M_rot_g_sqr[round(K_m_strong_1)-1]+M_rot_j_sqr[round(K_m_strong_1)-1])*sum(a)*A1_strong_1*K_m_strong_1*w_s/2
N_vibr_main_1=(M_rot_g_sqr[round(K_m_main_1)-1]+M_rot_j_sqr[round(K_m_main_1)-1])*sum(a)*A1_main_1*K_m_main_1*w_s/2

N_vibr_strong_2=(M_rot_g_sqr[round(K_m_strong_2)-1]+M_rot_j_sqr[round(K_m_strong_2)-1])*sum(a)*A1_strong_2*K_m_strong_2*w_s/2
N_vibr_main_2=(M_rot_g_sqr[round(K_m_main_2)-1]+M_rot_j_sqr[round(K_m_main_2)-1])*sum(a)*A1_main_2*K_m_main_2*w_s/2

N_vibr_strong_3=(M_rot_g_sqr[round(K_m_strong_3)-1]+M_rot_j_sqr[round(K_m_strong_3)-1])*sum(a)*A1_strong_3*K_m_strong_3*w_s/2
N_vibr_main_3=(M_rot_g_sqr[round(K_m_main_3)-1]+M_rot_j_sqr[round(K_m_main_3)-1])*sum(a)*A1_main_1*K_m_main_3*w_s/2

N_sum=N_vibr_strong_1+N_vibr_main_1+N_vibr_strong_2+N_vibr_main_2+N_vibr_strong_3+N_vibr_main_3

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_strong_1']+np.round(N_vibr_strong_1, 3).astype(str).tolist()]
data=str1+str2

print('Первая сильная гармоника\n')
print_pretty_table(data)
write_to_file('Первая сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)
write_to_file('\n')

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_main_1']+np.round(N_vibr_main_1, 3).astype(str).tolist()]
data=str1+str2

print('Первая главная гармоника\n')
print_pretty_table(data)
write_to_file('Первая главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)
write_to_file('\n')

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_strong_2']+np.round(N_vibr_strong_2, 3).astype(str).tolist()]
data=str1+str2

print('Вторая сильная гармоника\n')
print_pretty_table(data)
write_to_file('Вторая сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)
write_to_file('\n')

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_main_2']+np.round(N_vibr_main_2, 3).astype(str).tolist()]
data=str1+str2

print('Вторая главная гармоника\n')
print_pretty_table(data)
write_to_file('Вторая главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)
write_to_file('\n')

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_strong_3']+np.round(N_vibr_strong_3, 3).astype(str).tolist()]
data=str1+str2

print('Третья сильная гармоника\n')
print_pretty_table(data)
write_to_file('Третья сильная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)
write_to_file('\n')

str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_main_3']+np.round(N_vibr_main_3, 3).astype(str).tolist()]
data=str1+str2

print('Третья главная гармоника\n')
print_pretty_table(data)
write_to_file('Третья главная гармоника\n\n', filename=f_tables)
write_pretty_table_to_txt(data)
write_to_file('\n')

N_vibr_sum=N_vibr_strong_1+N_vibr_main_1+N_vibr_strong_2+N_vibr_main_2+N_vibr_strong_3+N_vibr_main_3
str1=[['n']+np.round(n_diap).astype(int).astype(str).tolist()]
str2=[['N_vibr_sum']+np.round(N_vibr_sum, 3).astype(str).tolist()]
data=str1+str2
print('Суммарная мощность колебаний\n')
write_to_file('Суммарная мощность колебаний\n\n', f_tables)
print_pretty_table(data)
write_pretty_table_to_txt(data)

plt.figure(1)
plt.plot(n_diap, A1_main_1)
plt.figure(2)
plt.plot(n_diap, A1_strong_1)

plt.figure(3)
plt.subplot(2,3,1)
plt.plot(n_diap, A1_main_1)
plt.title='A1_main_1'
plt.subplot(2,3,2)
plt.plot(n_diap, A1_main_2)
plt.title='A1_main_2'
plt.subplot(2,3,3)
plt.plot(n_diap, A1_main_3)
plt.title='A1_main_3'

plt.subplot(2,3,4)
plt.plot(n_diap, A1_strong_1)
plt.title='A1_strong_1'
plt.subplot(2,3,5)
plt.plot(n_diap, A1_strong_2)
plt.title='A1_strong_2'
plt.subplot(2,3,6)
plt.plot(n_diap, A1_strong_3)
plt.title='A1_strong_3'

plt.show()
input()
