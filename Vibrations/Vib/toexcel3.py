import pandas as pd

numbers = [1, 2, 3, 4, 5]

# Create a DataFrame from the list of numbers
df = pd.DataFrame(numbers, columns=['Numbers'])

# Save the DataFrame to an Excel file
df.to_excel('numbers.xlsx', index=False)
