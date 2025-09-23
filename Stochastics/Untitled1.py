# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import math

def sequence_f(max_n: int = 100, f_next: float = 1.0) -> dict:
    """
    Вычисляет {n: f_n} для n=2..max_n
    по рекуррентной формуле f_n = sqrt(1 + n*f_{n+1}),
    начиная с заданного f_{max_n+1} = f_next.
    
    max_n : максимальное n
    f_next: граничное значение f_{max_n+1}
    """
    values = {max_n + 1: f_next}
    def compute(n: int):
        if n in values:
            return values[n]
        fn1 = compute(n + 1)
        values[n] = math.sqrt(1 + n * fn1)
        return values[n]
    
    # Заполняем для n от max_n вниз до 2
    for n in range(max_n, 1, -1):
        compute(n)
    return {k: values[k] for k in range(2, max_n + 1)}

# Пример использования:
if __name__ == "__main__":
    result = sequence_f(max_n=20, f_next=1.0)
    # Выведем первые несколько минимальных членов
    for n in range(2, 11):
        print(f"f_{n} = {result[n]:.6f}")


# %%
print(1+1)

# %%
