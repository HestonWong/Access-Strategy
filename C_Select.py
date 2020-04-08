def Select_feature(X_train, X_test, y_train, y_test, iv=0.01, corr=0.7):
    from toad.selection import select, stepwise

    # IV、corr综合筛选
    X_train, dropped = select(X_train, target=y_train, iv=iv, corr=corr, return_drop=True)
    X_test = X_test[X_train.columns]
    print('IV筛选不通过的特征为：\n', dropped['iv'], '\n',
          'corr筛选不通过的特征为：\n', dropped['iv'])
    print('IV/corr综合筛选后剩余{}个特征'.format(X_train.shape[1]))

    # 逻辑双向逐步回归筛选
    # X_train = stepwise(X_train, target=y_train, estimator='ols', direction='both', criterion='aic')
    # X_test = X_test[X_train.columns]
    # print('stepwise筛选后剩余{}个特征'.format(X_train.shape[1]))
    # print('特征筛选已完成', '\n')
    return X_train, X_test, y_train, y_test

