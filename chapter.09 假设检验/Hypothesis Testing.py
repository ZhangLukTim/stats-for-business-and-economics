import numpy as np
from scipy.stats import norm, t


def p_value(x_bar, mu, sigma, n, how):
    """ 计算sigma已知情况下的p-值
    总体均值假设检验，当sigma已知的情况下计算p-value

    Params
    ------
    x_bar: 样本均值
    mu: 总体均值，即目标值
    sigma: 总体方差
    n: 样本容量
    how: 假设检验方法，可选择 ( 'up', 'down', 'double' )

    Return
    ------
    (检验统计量, p-值)

    """
    z = (x_bar - mu) / (sigma / pow(n, 0.5))
    if how == 'up':
        p = norm.sf(z)
    elif how == 'down':
        p = norm.cdf(z)
    elif how == 'double':
        p = norm.sf(abs(z)) * 2
    else:
        pass

    return z, p

def p_valut_t(x_bar, mu, s, n, how):
    """ 计算sigma未知情况下的p-值
    总体均值假设检验，当sigma未知的情况下计算p-value

    Params
    ------
    x_bar: 样本均值
    mu: 总体均值，即目标值
    s: 样本方差
    n: 样本容量
    how: 假设检验方法，可选择 ( 'up', 'down', 'double' )

    Return
    ------
    (检验统计量, p-值)

    """
    t_dist = t(n-1)
    t_val = (x_bar - mu) / (s / np.sqrt(n))
    if how == 'up':
        p = t.sf(z)
    elif how == 'down':
        p = t.cdf(z)
    elif how == 'double':
        p = t.sf(abs(z)) * 2
    else:
        pass

    return t_val, p

def p_value_p(ratio, target, n, how):
    """ 计算总体比率的p-值
    总体比率假设检验

    Params
    ------
    ratio: 样本中目标值的比率
    target: 需要检验的目标比率
    n: 样本容量
    how: 假设检验方法，可选择 ( 'up', 'down', 'double' )

    Return
    ------
    (检验统计量, p-值)
    """

    z = ( ratio - target ) / np.sqrt(target * ( 1 - target ) / n)
    if how == 'up':
        p = norm.sf(z)
    elif how == 'down':
        p = norm.cdf(z)
    elif how == 'double':
        p = norm.sf(abs(z)) * 2
    else:
        pass
    return z, p


# 计算正态分布上某个区间的面积大小
def norm_inter_prob(a=-np.inf, b=np.inf, mu=0, sigma=1):
    norm_dist = norm(mu, sigma)
    return norm_dist.cdf(b) - norm_dist.cdf(a)


# 已知显著性水平和第二类错误的上限，以及样本均值、方差、实际均值
# 估算进行统计检验需要的样本大小
def get_sample_size(alpha, beta, sigma, mu0, mu1, how):
    if how == 'double':
        z0 = norm.isf(alpha / 2)
    elif how == 'up' or how == 'down':
        z0 = norm.isf(alpha)
    else:
        print("how参数错误.")
        return -1

    z1 = norm.isf(beta)
    n = pow((z0 + z1) * sigma / ( mu0 - mu1 ), 2) 
    return n


def p_value_pair(x1, x2, mu, sigma1, sigma2, n1, n2, how):
    """ 计算sigma已知情况下的p-值
    总体均值假设检验，当sigma已知的情况下计算p-value

    Params
    ------
    x_bar: 样本均值
    mu: 总体均值，即目标值
    sigma: 总体方差
    n: 样本容量
    how: 假设检验方法，可选择 ( 'up', 'down', 'double' )

    Return
    ------
    (检验统计量, p-值)

    """
    z = (x1 - x2 - mu) / ( pow( pow(sigma1, 2) / n1 + pow(sigma2, 2) / n2, 0.5) )
    if how == 'up':
        p = norm.sf(z)
    elif how == 'down':
        p = norm.cdf(z)
    elif how == 'double':
        p = norm.sf(abs(z)) * 2
    else:
        pass

    return z, p
