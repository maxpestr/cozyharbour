from openpyxl import Workbook

numbers = [1, 2, 3, 4, 5]

# Create a new workbook
workbook = Workbook()

# Get the active sheet
sheet = workbook.active

# Write the numbers to the sheet
for i, number in enumerate(numbers, start=1):
    sheet.cell(row=i, column=1, value=number)

# Save the workbook to a file
workbook.save('numbers.xlsx')
