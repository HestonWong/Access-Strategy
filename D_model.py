from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, recall_score, make_scorer
import numpy as np
import visuals as vs


def Model_train(X_train, X_test, y_train, y_test, n=10):
    estimator = DecisionTreeClassifier(class_weight='balanced')

    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    param = {'max_depth': [4, 5, 6, 7],
             'min_samples_leaf': np.linspace(0.01, 0.05, 20, dtype=float)}
    scoring = make_scorer(recall)
    model = GridSearchCV(estimator=estimator, param_grid=param, scoring=scoring, cv=4, verbose=True, n_jobs=12)
    model.fit(X_train, y_train)
    print('最佳模型参数：\n', model.best_estimator_, '\n')

    model = model.best_estimator_
    importances = model.fit(X_train, y_train).feature_importances_

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    print('AUC_train:{:.3f} | AUC_test:{:.3f}'.format(auc(fpr_train, tpr_train), auc(fpr_test, tpr_test)))
    print('KS_train:{:.3f} | KS_test:{:.3f}'.format(max(abs(tpr_train - fpr_train)), max(abs(tpr_test - fpr_test))))
    print('Recall_train:{:.3f} | Recall_test:{:.3f}'.format(recall(y_train, y_train_pred), recall(y_test, y_test_pred)))
    col_n = vs.feature_plot(importances, X_train, y_train, n_top=n)
    vs.roc(fpr_train, tpr_train, fpr_test, tpr_test)
    vs.graph(model, X_train, y_train, w=True)

    print('模型训练完成', '\n')
    return y_train_pred, y_train_prob, y_test_pred, y_test_prob, col_n
