"""
Лабораторна робота №5 — Завдання 2.4
Обчислення відносної важливості ознак за допомогою регресора AdaBoost

Запуск:
    python3 LR_5_task_4.py

Примітка: використовується синтетичний датасет нерухомості
(аналог Boston/California Housing, сумісний із будь-яким оточенням).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ── Генерація синтетичного датасету нерухомості ──────────────────────────────
# 13 ознак — аналог класичного Boston Housing
np.random.seed(42)
N = 506

feature_names = np.array([
    'CRIM',   # рівень злочинності
    'ZN',     # частка жилих зон
    'INDUS',  # частка промислових зон
    'CHAS',   # приналежність до р. Чарльз
    'NOX',    # концентрація NO
    'RM',     # середня кількість кімнат
    'AGE',    # частка старих будинків
    'DIS',    # відстань до центрів зайнятості
    'RAD',    # доступ до магістралей
    'TAX',    # податкова ставка
    'PTRATIO',# учнів на вчителя
    'B',      # частка темношкірих
    'LSTAT',  # % нижчих верств населення
])

CRIM    = np.random.exponential(3.6, N)
ZN      = np.random.uniform(0, 100, N)
INDUS   = np.random.uniform(0.5, 27, N)
CHAS    = np.random.binomial(1, 0.07, N).astype(float)
NOX     = np.random.uniform(0.38, 0.87, N)
RM      = np.random.normal(6.28, 0.7, N)
AGE     = np.random.uniform(2.9, 100, N)
DIS     = np.random.exponential(3.8, N) + 1
RAD     = np.random.choice(range(1, 25), N).astype(float)
TAX     = np.random.uniform(187, 711, N)
PTRATIO = np.random.uniform(12.6, 22, N)
B       = np.random.uniform(0.32, 396.9, N)
LSTAT   = np.random.exponential(12, N)

X_raw = np.column_stack([CRIM, ZN, INDUS, CHAS, NOX, RM,
                          AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT])

# Цільова змінна — ціна (MEDV), залежить від ознак
y_raw = (
    -0.1 * CRIM + 0.05 * ZN - 0.04 * INDUS + 2.5 * CHAS
    - 17 * NOX + 3.8 * RM - 0.01 * AGE - 1.5 * DIS
    - 0.01 * RAD - 0.01 * TAX - 0.9 * PTRATIO + 0.009 * B
    - 0.52 * LSTAT + np.random.normal(0, 2, N) + 22
)
y_raw = np.clip(y_raw, 5, 50)  # ціни у тисячах доларів


# ── Перемішування даних ──────────────────────────────────────────────────────
X, y = shuffle(X_raw, y_raw, random_state=7)

# ── Розбиття даних на навчальний та тестовий набори ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# ── Модель на основі регресора AdaBoost ──────────────────────────────────────
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400,
    random_state=7
)
regressor.fit(X_train, y_train)

# ── Обчислення показників ефективності регресора AdaBoost ────────────────────
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nADABOOST REGRESSOR")
print(f"Mean squared error      = {round(mse, 2)}")
print(f"Explained variance score = {round(evs, 2)}")

# ── Вилучення важливості ознак ───────────────────────────────────────────────
feature_importances = regressor.feature_importances_

# ── Нормалізація значень важливості ознак ────────────────────────────────────
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# ── Сортування та перестановка значень ──────────────────────────────────────
index_sorted = np.flipud(np.argsort(feature_importances))

# ── Розміщення міток уздовж осі X ───────────────────────────────────────────
pos = np.arange(index_sorted.shape[0]) + 0.5

# ── Побудова стовпчастої діаграми ────────────────────────────────────────────
plt.figure(figsize=(11, 6))
plt.bar(pos, feature_importances[index_sorted], align='center', color='steelblue')
plt.xticks(pos, feature_names[index_sorted], rotation=30, ha='right', fontsize=11)
plt.ylabel('Відносна важливість (%)', fontsize=12)
plt.title('Оцінка важливості ознак за допомогою регресора AdaBoost\n(Housing Dataset)', fontsize=13)
plt.tight_layout()

# ── Виведення таблиці важливості ────────────────────────────────────────────
print("\nВідносна важливість ознак (відсортовано):")
print(f"{'Ознака':<12} {'Важливість (%)':>15}  Інтерпретація")
print("-" * 60)
descriptions = {
    'CRIM':    'рівень злочинності',
    'ZN':      'частка жилих зон',
    'INDUS':   'частка промислових зон',
    'CHAS':    'приналежність до р. Чарльз',
    'NOX':     'концентрація NO',
    'RM':      'середня кількість кімнат',
    'AGE':     'частка старих будинків',
    'DIS':     'відстань до центрів зайнятості',
    'RAD':     'доступ до магістралей',
    'TAX':     'податкова ставка',
    'PTRATIO': 'учнів на вчителя',
    'B':       'частка темношкірих',
    'LSTAT':   '% нижчих верств населення',
}
for i in index_sorted:
    name = feature_names[i]
    print(f"{name:<12} {feature_importances[i]:>14.2f}  {descriptions[name]}")

plt.show()
