import pandas as pd

def merge_and_sort_excel_files():
    try:
        # Ask for the first file
        file1 = input("Введите имя первого Excel файла: ")
        print("Загрузка первого файла...")
        try:
            df1 = pd.read_excel(file1)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file1}' не найден!")
            return
        except Exception as e:
            print(f"Ошибка при загрузке первого файла: {str(e)}")
            return

        # Ask for the second file
        file2 = input("Введите имя второго Excel файла: ")
        print("Загрузка второго файла...")
        try:
            df2 = pd.read_excel(file2)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file2}' не найден!")
            return
        except Exception as e:
            print(f"Ошибка при загрузке второго файла: {str(e)}")
            return

        # Ensure both files have the same columns
        if not all(df1.columns == df2.columns):
            print("Ошибка: Столбцы в файлах не совпадают!")
            return

        # Merge the two DataFrames
        print("Объединение файлов...")
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Check if the required column exists
        if 'VISIT_AT_UTC' not in combined_df.columns:
            print("Ошибка: Столбец 'VISIT_AT_UTC' не найден в объединенных данных!")
            return

        # Create a temporary column for sorting
        print("Сортировка...")
        combined_df['SORT_VISIT_AT_UTC'] = pd.to_datetime(combined_df['VISIT_AT_UTC'], errors='coerce')
        combined_df.sort_values('SORT_VISIT_AT_UTC', inplace=True)
        combined_df.drop(columns=['SORT_VISIT_AT_UTC'], inplace=True)

        # Ask for the output file name
        output_file = input("Введите имя для сохранения объединенного файла (с расширением .xlsx): ")
        print("Сохранение...")
        combined_df.to_excel(output_file, index=False)

        print("Готово! Объединенный и отсортированный файл сохранен как:", output_file)

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

# Run the script
if __name__ == "__main__":
    merge_and_sort_excel_files()
