def fea_engn(data):
    # 第一次借贷时间锚点
    data['rec_cr_line_month'] = data['earliest_cr_line'].dt.month + (data['earliest_cr_line'].dt.year - 1900) * 12
    data.drop('earliest_cr_line', axis=1, inplace=True)

    # 证实收入
    data['verified_inc'] = data[['annual_inc', 'verification_status']].apply(lambda x: x[0] if x[1] == 'Verified' else 0, axis=1)
    data['source_verified_inc'] = data[['annual_inc', 'verification_status']].apply(lambda x: x[0] if x[1] == 'Source_Verified' else 0, axis=1)
    data['not_verified_inc'] = data[['annual_inc', 'verification_status']].apply(lambda x: x[0] if x[1] == 'Not_Verified' else 0, axis=1)

    # 当前借贷负债收入比
    data['dti2'] = data['installment'] * 12 / (data['annual_inc'] + 1)
    return data
    print('特征加工完成')


def Combiner_Trans(X_train, X_test, y_train, y_test, exclude_lsit=[], new_bins={}, category=0, numeric=0, plt_show=0):
    from toad.transform import Combiner
    import matplotlib.pyplot as plt
    import pandas as pd
    comb = Combiner()

    if category:  # 离散变量分箱
        category = X_train.columns[X_train.dtypes == 'object'].tolist()
        comb.fit(X_train[category], y=y_train, method='chi', min_samples=0.1, empty_separate=False, exclude=exclude_lsit)
        bins = comb.export()

        if new_bins:  # 如果指定分箱则执行
            comb.set_rules(new_bins)
        print('离散变量分箱结果：\n', bins)

        if plt_show:
            for i in category:  # 观察分箱情况
                bin_plot(pd.concat([X_train[i], y_train], axis=1), x=i, target='loan_status')
                plt.show()

        X_train[category] = comb.transform(X_train[category], labels=True)
        X_test[category] = comb.transform(X_test[category], labels=True)
        Combiner_category = pd.concat([X_train[category], y_train], axis=1)
        Combiner_category.to_csv('Combiner_category.csv')
        print('离散变量分箱结果已保存')

    elif numeric:  # 数值变量分箱
        numeric = X_train.columns[X_train.dtypes != 'object'].tolist()
        comb.fit(X_train[numeric], y=y_train, method='chi', min_samples=0.1, empty_separate=False, exclude=exclude_lsit)
        bins = comb.export()

        if new_bins:  # 如果指定分箱则执行
            combiner.set_rules(new_bins)
        print('数值变量分箱结果：\n', bins)

        if plt_show:
            for i in category:  # 观察分箱情况
                bin_plot(pd.concat([X_train[i], y_train], axis=1), x=i, target='loan_status')
                plt.show()

        X_train[numeric] = comb.transform(X_train[numeric], labels=True)
        X_test[numeric] = comb.transform(X_test[numeric], labels=True)
        Combiner_numeric = pd.concat([X_train[numeric], y_train], axis=1)
        Combiner_numeric.to_csv('Combiner_numeric.csv')
        print('数值变量分箱结果已保存')

    else:  # 全部变量分箱
        comb.fit(X_train, y=y_train, method='chi', min_samples=0.1, empty_separate=False, exclude=exclude_lsit)
        bins = comb.export()

        if new_bins:  # 如果指定分箱则执行
            combiner.set_rules(new_bins)
        print('数值变量分箱结果：\n', bins)

        if plt_show:
            for i in X_train.columns:  # 观察分箱情况
                bin_plot(pd.concat([X_train[i], y_train], axis=1), x=i, target='loan_status')
                plt.show()

        X_train = comb.transform(X_train, labels=True)
        X_test = comb.transform(X_test, labels=True)
        Combiner_all = pd.concat([X_train, y_train], axis=1)
        Combiner_all.to_csv('Combiner_all.csv')
        print('全部变量分箱结果已保存')
    print('数据分箱完成')
    return X_train, X_test, y_train, y_test


def WOE_Trans(X_train, X_test, y_train, y_test):
    from toad.transform import WOETransformer

    WOE_t = WOETransformer()
    category = X_train.columns[X_train.dtypes == 'object'].tolist()
    X_train[category] = WOE_t.fit_transform(X_train[category], y=y_train)
    X_test[category] = WOE_t.fit_transform(X_test[category], y=y_test)
    X_train[category].to_csv('WOE_category.csv')
    print('分类变量WOE编码已保存')
    print('数据WOE编码完成')
    return X_train, X_test, y_train, y_test
