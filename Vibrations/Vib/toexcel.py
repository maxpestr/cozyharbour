import pandas as pd
from openpyxl import Workbook

# Создание DataFrame
data = {'a': ['Some text', 'Longer text', 'Short text'],
        'b': ['Another text', 'Even longer text', 'Text']}
df = pd.DataFrame(data)

# Создание нового Excel-файла
wb = Workbook()
ws = wb.active

# Запись DataFrame в файл формата xlsx
for row in df.itertuples(index=False):
    ws.append(row)

# Настройка ширины столбцов
for column_cells in ws.columns:
    length = max(len(str(cell.value)) for cell in column_cells)
    ws.column_dimensions[column_cells[0].column_letter].width = length

# Сохранение файла
wb.save('output.xlsx')
