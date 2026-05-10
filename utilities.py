"""
Допоміжний модуль — функція візуалізації класифікатора
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_classifier(classifier, X, y, title=''):
    """
    Візуалізує межі рішень класифікатора на 2D площині.
    """
    # Визначаємо крок сітки та межі
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    step_size = 0.01

    # Будуємо сітку точок
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )

    # Передбачаємо клас для кожної точки сітки
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Кольорова карта для фону та точок
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel1, shading='auto')

    # Малюємо точки даних
    markers = ['s', 'o', '^', 'D', 'v']
    colors  = ['black', 'blue', 'red', 'green', 'purple']
    unique_labels = np.unique(y)

    for idx, label in enumerate(unique_labels):
        mask = (y == label)
        plt.scatter(
            X[mask, 0], X[mask, 1],
            s=75,
            facecolors='white',
            edgecolors=colors[idx % len(colors)],
            linewidth=1,
            marker=markers[idx % len(markers)],
            label=f'Class-{int(label)}'
        )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
