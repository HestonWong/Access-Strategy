from pandas.api.types import CategoricalDtype
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行


# 定义回调函数，去除每列数据中的异常值，异常值为均值两边超过三个标准差的数据
def exclude_extreme(s):
    mean = s.mean()
    sigma = s.std()
    ceiling = mean + 3 * sigma
    floor = mean - 3 * sigma
    s_copy = s.copy()
    s_copy[s_copy > ceiling] = ceiling
    s_copy[s_copy < floor] = floor
    print('.', end='')
    return s_copy


def feature(data):
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # 缺失

    # 删除贷后 和没用的特征
    d = ['desc', 'emp_title', 'tax_liens', 'last_pymnt_d', 'title', 'last_credit_pull_d', 'issue_d', 'zip_code', 'grade',
         'debt_settlement_flag', 'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'last_pymnt_amnt', 'total_pymnt', 'total_pymnt_inv',
         'total_rec_int', 'total_rec_late_fee', 'int_rate', 'term', 'next_pymnt_d', 'il_util', 'pymnt_plan', 'initial_list_status',
         'application_type','hardship_flag', 'disbursement_method', 'out_prncp', 'out_prncp_inv', 'funded_amnt', 'funded_amnt', 'loan_amnt']
    data = data.drop(d, axis=1)
    missing_80percent = list(data.columns[data.isnull().sum() > len(data) * 0.8])
    data.drop(missing_80percent, axis=1, inplace=True)
    missing_40percent = list(data.columns[data.isnull().sum() > len(data) * 0.4])
    fill_max = ['mths_since_last_delinq', 'mths_since_recent_revol_delinq', 'mths_since_recent_bc_dlq', 'mths_since_last_major_derog']
    data[fill_max] = data[fill_max].fillna(data[fill_max].max())

    # 初步填充缺失值，后期根据情况精细调整
    fill_0 = ['mo_sin_old_il_acct']
    fill_none = []
    fill_median = ['mths_since_recent_inq', 'mths_since_rcnt_il', 'bc_util', 'percent_bc_gt_75', 'bc_open_to_buy', 'mths_since_recent_bc', 'all_util',
                   'dti', 'total_bal_il', 'max_bal_bc']
    fill_mode = ['emp_length', 'revol_util', 'num_tl_120dpd_2m', 'total_cu_tl', 'inq_last_6mths', 'open_acc_6m', 'open_act_il', 'open_il_12m',
                 'open_il_24m', 'open_rv_12m', 'open_rv_24m', 'inq_fi', 'inq_last_12m']
    data[fill_0] = data[fill_0].apply(lambda x: x.fillna(0))
    data[fill_none] = data[fill_none].apply(lambda x: x.fillna('None'))
    data[fill_median] = data[fill_median].apply(lambda x: x.fillna(x.median()))
    data[fill_mode] = data[fill_mode].apply(lambda x: x.fillna(x.mode()[0]))  # [0]以防有两个众数
    # missing2 = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
    # missing1 = data.columns[data.isnull().sum() != 0].tolist()
    # print(missing1, '\n', missing2)

    # 删除只有唯一值的特征
    # print(data.columns[data.nunique() == 1].tolist())
    d = ['policy_code']
    data.drop(d, axis=1, inplace=True)
    print('缺失处理完成')

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # 清洗

    # 字符型清洗
    data[['revol_util']] = data[['revol_util']].apply(lambda y: y.str.rstrip('%').astype(float) / 100)
    # col = data.columns[data.dtypes == 'object'].tolist()
    # print(data.info())
    # print(data[col].nunique())

    # 有序分类型
    # data['emp_length'] = data['emp_length'].astype(CategoricalDtype(categories=[
    #     '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], ordered=True))

    # 观察数值数据分布情况，并把异常变量控制在3倍标准差以内
    # data.select_dtypes(include=[np.number]).hist(alpha=0.7, figsize=(16,16), bins=20)
    # print(data.columns[data.dtypes != 'object'].tolist())
    col = ['annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
           'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il',
           'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
           'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
           'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
           'mort_acc', 'mths_since_recent_bc', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
           'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
           'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
           'total_il_high_credit_limit']

    print('异常值处理中：')
    data[col] = data[col].apply(lambda x: exclude_extreme(x))
    print('异常值处理完成')

    # 观察存疑数据列
    # for i in data.columns[data.dtypes != 'object']:
    #     plt.title(i)
    #     data[i].hist(alpha=0.7, figsize=(16, 16), bins=10)
    #     plt.show()
    # for i in ['annual_inc', 'dti']:
    #     plt.title(i)
    #     plt.boxplot(data[i], whis=3, flierprops={'marker': 'D', 'markerfacecolor': 'grey'})
    #     plt.show()

    # 删除灰色区域,加大区分度
    data['loan_status'] = data['loan_status'].map({
        'Fully Paid': 0,
        'Current': 0,
        'Charged Off': 1,
        'Late (31-120 days)': 1,
        'Late (16-30 days)': 1,
        'In Grace Period': 2,
        'Default': 2})
    data = data[data['loan_status'].isin([0, 1])]
    print('数据清洗完成')
    return data
