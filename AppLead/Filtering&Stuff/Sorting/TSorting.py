import pandas as pd

def process_excel_file():
    try:
        # Get filename from user
        filename = input("Введите имя Excel файла: ")
            
            # Load Excel file
        print("Загрузка...")
        try:
            df = pd.read_excel(filename)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{filename}' не найден!")
            return
        except Exception as e:
            print(f"Ошибка при загрузке файла: {str(e)}")
            return
        
        # Check if required column exists
        if 'VISIT_AT_UTC' not in df.columns:
            print("Ошибка: Столбец 'VISIT_AT_UTC' не найден в файле!")
            return
        
        # Create a new column for sorting without modifying the original
        df['SORT_VISIT_AT_UTC'] = pd.to_datetime(df['VISIT_AT_UTC'], errors='coerce')
        
        # Sort dataframe
        print("Сортировка...")
        df.sort_values('SORT_VISIT_AT_UTC', inplace=True)
        
        # Drop the temporary sorting column
        df.drop(columns=['SORT_VISIT_AT_UTC'], inplace=True)
        
        # Ask for the output file name
        output_file = input("Введите имя для сохранения объединенного файла (с расширением .xlsx): ")
        print("Сохранение...")
        df.to_excel(output_file, index=False)
        
        print("Готово! Файл успешно обработан.")
    
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

# Run the script
if __name__ == "__main__":
    process_excel_file()
