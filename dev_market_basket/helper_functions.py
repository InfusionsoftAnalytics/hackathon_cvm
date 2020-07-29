# PACKAGES
import datetime as datetime
import os as os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 15)


def joint_prob(df, col1, col2, ui_col=None):
    '''
    Calculates join prob
    :param df: dataframe with indicator columns
    :param col1: name of first column
    :param col2: name of second column
    :return: joint probability
    '''
    if(ui_col==None):
        df['combo']=df[col1]*df[col2]
        num=df['combo'].sum()
        denom=df.shape[0]
        return(num/denom)
    else:
        df['combo'] = df[col1] * df[col2]
        num = df[df.ui==ui_col]['combo'].sum()
        denom = df[df.ui==ui_col].shape[0]
        return (num / denom)


def marginal_prob(df, col1, ui_col=None):
    '''
    :param df: dataframe with indicator columns
    :param col1: name of column to get frequency of
    :return: marginal prob/frequency of column
    '''
    if (ui_col == None):
        num=df[col1].sum()
        denom=df.shape[0]
        return(num/denom)
    else:
        num = df[df.ui==ui_col][col1].sum()
        denom = df[df.ui==ui_col].shape[0]
        return (num / denom)


def get_pmi(df, col1, col2, ui_col=None):
    '''
    Calcualtes pmi
    :param df: dataframe with indicator columns
    :param col1: name of first column
    :param col2: name of second column
    :return: pmi for two columns, https://en.wikipedia.org/wiki/Pointwise_mutual_information
    '''
    if (ui_col == None):
        num= joint_prob(df, col1, col2)
        denom=marginal_prob(df, col1)*marginal_prob(df, col2)
        log_return = np.log(num/denom)
        return(log_return)
    else:
        num= joint_prob(df, col1, col2, ui_col)
        denom=marginal_prob(df, col1, ui_col)*marginal_prob(df, col2, ui_col)
        log_return = np.log(num/denom)
        return(log_return)

def get_npmi(df, col1, col2, ui_col=None):
    '''
    Calcualtes pmi
    :param df: dataframe with indicator columns
    :param col1: name of first column
    :param col2: name of second column
    :return: normalized pointwise mutual info
    '''
    if (ui_col == None):
        num= get_pmi(df, col1, col2)
        denom=-np.log(joint_prob(df, col1, col2))
        return(num/denom)
    else:
        num= get_pmi(df, col1, col2, ui_col)
        denom=-np.log(joint_prob(df, col1, col2, ui_col))
        return(num/denom)


def make_cvm_flags(df):
    '''
    This function adds indicator columns for cvm usage
    :param df: dataframe of daily app data
    :return: dataframe with new flag columns and a list of the names of the new columns
    '''

    og_col_names= list(df.columns.values)

    df['lead_webform']=np.where(df.webform_count>=1, 1, 0)
    df['note_added_contact']=np.where(df.total_tasks_completed>=1, 1, 0)
    df['task_completed']=np.where(df.total_tasks_completed >=1, 1, 0)
    df['client_opens_email']=np.where(df.total_emails_opened >=1, 1, 0)
    df['appointment_scheduled']=np.where(df.appointments_scheduled >=1, 1, 0)
    df['quote_accepted']=np.where(df.total_quotes >=1, 1, 0)
    df['invoice_paid']=np.where(df.quote_paid >=1, 1, 0)
    df['sms_sent_or_received']=np.where(df.sms_total_incoming_events+df.sms_total_outgoing_events >=1, 1, 0)
    df['manual_stage_prog']=np.where(df.contact_count_stages_manual >=1, 1, 0)
    df['new_pipeline_stage']=np.where(df.count_deal_pipeline >=1, 1, 0)
    df['opp_moves_stages']=np.where(df.count_contact_opportunity_stage_move >=1, 1, 0)

    new_col_names = list(df.columns.values)
    added_columns_list = list(np.setdiff1d(new_col_names,og_col_names))

    return(df, added_columns_list)




#https://www.statsmodels.org/stable/_modules/statsmodels/stats/proportion.html
def proportion_confint(count, nobs, alpha=0.05, method='normal'):
    '''confidence interval for a binomial proportion

    Parameters
    ----------
    count : int or array_array_like
        number of successes, can be pandas Series or DataFrame
    nobs : int
        total number of trials
    alpha : float in (0, 1)
        significance level, default 0.05
    method : {'normal', 'agresti_coull', 'beta', 'wilson', 'binom_test'}
        default: 'normal'
        method to use for confidence interval,
        currently available methods :

         - `normal` : asymptotic normal approximation
         - `agresti_coull` : Agresti-Coull interval
         - `beta` : Clopper-Pearson interval based on Beta distribution
         - `wilson` : Wilson Score interval
         - `jeffreys` : Jeffreys Bayesian Interval
         - `binom_test` : experimental, inversion of binom_test

    Returns
    -------
    ci_low, ci_upp : float, ndarray, or pandas Series or DataFrame
        lower and upper confidence level with coverage (approximately) 1-alpha.
        When a pandas object is returned, then the index is taken from the
        `count`.

    Notes
    -----
    Beta, the Clopper-Pearson exact interval has coverage at least 1-alpha,
    but is in general conservative. Most of the other methods have average
    coverage equal to 1-alpha, but will have smaller coverage in some cases.

    The 'beta' and 'jeffreys' interval are central, they use alpha/2 in each
    tail, and alpha is not adjusted at the boundaries. In the extreme case
    when `count` is zero or equal to `nobs`, then the coverage will be only
    1 - alpha/2 in the case of 'beta'.

    The confidence intervals are clipped to be in the [0, 1] interval in the
    case of 'normal' and 'agresti_coull'.

    Method "binom_test" directly inverts the binomial test in scipy.stats.
    which has discrete steps.

    TODO: binom_test intervals raise an exception in small samples if one
       interval bound is close to zero or one.

    References
    ----------
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    Brown, Lawrence D.; Cai, T. Tony; DasGupta, Anirban (2001). "Interval
        Estimation for a Binomial Proportion",
        Statistical Science 16 (2): 101–133. doi:10.1214/ss/1009213286.
        TODO: Is this the correct one ?

    '''

    pd_index = getattr(count, 'index', None)
    if pd_index is not None and callable(pd_index):
        # this rules out lists, lists have an index method
        pd_index = None
    count = np.asarray(count)
    nobs = np.asarray(nobs)

    q_ = count * 1. / nobs
    alpha_2 = 0.5 * alpha

    if method == 'normal':
        std_ = np.sqrt(q_ * (1 - q_) / nobs)
        dist = stats.norm.isf(alpha / 2.) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist

    elif method == 'binom_test':
        # inverting the binomial test
        def func(qi):
            return stats.binom_test(q_ * nobs, nobs, p=qi) - alpha
        if count == 0:
            ci_low = 0
        else:
            ci_low = optimize.brentq(func, float_info.min, q_)
        if count == nobs:
            ci_upp = 1
        else:
            ci_upp = optimize.brentq(func, q_, 1. - float_info.epsilon)

    elif method == 'beta':
        ci_low = stats.beta.ppf(alpha_2, count, nobs - count + 1)
        ci_upp = stats.beta.isf(alpha_2, count + 1, nobs - count)

        if np.ndim(ci_low) > 0:
            ci_low[q_ == 0] = 0
            ci_upp[q_ == 1] = 1
        else:
            ci_low = ci_low if (q_ != 0) else 0
            ci_upp = ci_upp if (q_ != 1) else 1

    elif method == 'agresti_coull':
        crit = stats.norm.isf(alpha / 2.)
        nobs_c = nobs + crit**2
        q_c = (count + crit**2 / 2.) / nobs_c
        std_c = np.sqrt(q_c * (1. - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist

    elif method == 'wilson':
        crit = stats.norm.isf(alpha / 2.)
        crit2 = crit**2
        denom = 1 + crit2 / nobs
        center = (q_ + crit2 / (2 * nobs)) / denom
        dist = crit * np.sqrt(q_ * (1. - q_) / nobs + crit2 / (4. * nobs**2))
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist

    # method adjusted to be more forgiving of misspellings or incorrect option name
    elif method[:4] == 'jeff':
        ci_low, ci_upp = stats.beta.interval(1 - alpha, count + 0.5,
                                             nobs - count + 0.5)

    else:
        raise NotImplementedError('method "%s" is not available' % method)

    if method in ['normal', 'agresti_coull']:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if pd_index is not None and np.ndim(ci_low) > 0:
        import pandas as pd
        if np.ndim(ci_low) == 1:
            ci_low = pd.Series(ci_low, index=pd_index)
            ci_upp = pd.Series(ci_upp, index=pd_index)
        if np.ndim(ci_low) == 2:
            ci_low = pd.DataFrame(ci_low, index=pd_index)
            ci_upp = pd.DataFrame(ci_upp, index=pd_index)

    return ci_low, ci_upp
