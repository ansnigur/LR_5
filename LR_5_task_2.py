"""
Лабораторна робота №5 — Завдання 2.2
Обробка дисбалансу класів

Запуск без балансування:
    python3 -W ignore LR_5_task_2.py
Запуск з балансуванням:
    python3 LR_5_task_2.py balance
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities import visualize_classifier


# ── Завантаження вхідних даних ──────────────────────────────────────────────
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# ── Поділ вхідних даних на два класи на підставі міток ──────────────────────
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

print(f"Кількість точок класу 0: {len(class_0)}")
print(f"Кількість точок класу 1: {len(class_1)}")

# ── Візуалізація вхідних даних ───────────────────────────────────────────────
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75,
            facecolors='black', edgecolors='black', linewidth=1,
            marker='x', label='Class-0')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
            facecolors='white', edgecolors='black', linewidth=1,
            marker='o', label='Class-1')
plt.title('Вхідні дані (дисбаланс класів)')
plt.legend()
plt.tight_layout()

# ── Розбиття даних на навчальний та тестовий набори ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# ── Класифікатор на основі гранично випадкових лісів ────────────────────────
# Параметр balance керує врахуванням дисбалансу класів
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        # Додаємо class_weight='balanced' для компенсації дисбалансу
        params['class_weight'] = 'balanced'
        title_suffix = '(з балансуванням)'
    else:
        raise TypeError("Invalid input argument; should be 'balance'")
else:
    title_suffix = '(без балансування)'

# ── Створення, навчання і візуалізація класифікатора ─────────────────────────
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train,
                     f'Extra Trees — Training dataset {title_suffix}')

# ── Передбачення та візуалізація для тестового набору ───────────────────────
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test,
                     f'Extra Trees — Test dataset {title_suffix}')

# ── Обчислення показників ефективності класифікатора ─────────────────────────
class_names = ['Class-0', 'Class-1']

print("\n" + "#" * 40)
print(f"\nClassifier performance on training dataset {title_suffix}\n")
print(classification_report(
    y_train, classifier.predict(X_train), target_names=class_names
))
print("#" * 40 + "\n")

print("#" * 40)
print(f"\nClassifier performance on test dataset {title_suffix}\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

plt.show()
