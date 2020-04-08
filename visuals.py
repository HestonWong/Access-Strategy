from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import pydotplus


def feature_plot(importances, X_train, y_train, n_top=10):
    indices = np.argsort(importances)[::-1]  # 返回排序后的索引，[i:j:-1]取i到j步进为-1
    columns = X_train.columns.values[indices][:n_top]
    values = importances[indices][:n_top]

    fig = plt.figure(figsize=(12, 5))
    plt.title('Top N Feature Importances')
    rects = plt.bar(np.arange(n_top), values, width=0.6, align='center', color='#99CCFF', label='Feature Weight')

    # 使条形图更高以适合文本标签
    axes = plt.gca()
    axes.set_ylim([0, np.max(values) * 1.1])

    # 在每个条形加上标签
    delta = np.max(values) * 0.02

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0,
                 height + delta,
                 '%.2f' % height,
                 ha='center',
                 va='bottom')

    # 检测x轴便签是否过长
    rotation = 30  # 旋转角度
    for i in columns:
        if len(i) > 15:
            rotation = 90
            break

    plt.xticks(np.arange(n_top), columns, rotation=rotation)
    plt.xlim((-0.5, n_top - 0.5))  # x轴坐标的限制
    plt.xlabel('Weight', fontsize=12)
    plt.ylabel('Feature', fontsize=12)

    plt.legend(loc='upper center')  # 图例标签位置在顶部中间
    plt.tight_layout()
    plt.show()
    return columns


def roc(fpr_train, tpr_train, fpr_test, tpr_test):
    plt.plot(fpr_train, tpr_train, label='train')
    plt.plot(fpr_test, tpr_test, label='test')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


def graph(model, X, y, p=0, w=0):
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=X.columns,
                                    class_names=['good', 'bad'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('tree.png')
    if p:
        print(tree.export_graphviz(model, out_file=None))
    elif w:
        with open('tree_graphviz.dot', 'w') as writer:
            tree.export_graphviz(model, out_file=writer)
    else:
        pass
