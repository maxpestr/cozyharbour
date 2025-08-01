import pandas as pd
import warnings
from pyexcelerate import Workbook

warnings.filterwarnings('ignore')

filename = input('Enter data filename\n')
print('Loading data file...')
df = pd.read_excel(filename)
print('Processing...')

#FILTERING
df = df.loc[(df['CAMPAIGN_CURRENT_TYPE'] == 'Appointments') | (df['CAMPAIGN_CURRENT_TYPE'] == 'Leads')]
df = df.dropna(subset=['ZIP_CODE', 'SERVICE', 'TRAFFIC_SOURCE'])
df = df[df['INCLUDE_IN_SET_RATE'] == True]

#WRITING FILTERED FILE
file_path = input('Enter output filename\n')
print('Writing to file...')

values = [df.columns] + list(df.values)
wb = Workbook()
wb.new_sheet('Sheet1', data=values)
wb.save(file_path)

print('The result was successfully saved\n')
input('Press ENTER to exit\n')
