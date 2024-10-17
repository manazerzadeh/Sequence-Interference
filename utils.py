import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from scipy import stats
import matplotlib.cm as cm
import seaborn as sns
from typing import List
import pingouin as pg

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM


from natsort import index_natsorted

path = "./SI1/SequenceInterference"

path_misc = "./SI1_miscs/"

g_sequences = {}
digit_change = {}
g_sequences[0] = ['13524232514' ,'35421252143',  '51423252413', '14325242135'] #Group 1 sequences
g_sequences[1] = ['51423252413', '14325242135', '13524232514' ,'35421252143'] #Group 2 sequences

seq_length = len(g_sequences[0][0])


digit_change = [4, 6, 8]

fingers = ['1', '2', '3', '4', '5'] #mapping of fingers to numbers


iti = 3000   #Inter trial interval
execTime = 10000 # msecs for each trial maximum
precueTime = 1500 # msecs for planning before movement 
hand = 2 #left or right hand




def read_dat_file(path : str):
    column_names = pd.read_csv(path, delimiter='\t', usecols=lambda column: not column.startswith("Unnamed")).columns
    dtype_dict = {col: int for col in column_names}
    dtype_dict['timeThreshold'] = float
    dtype_dict['timeThresholdSuper'] = float

    data = pd.read_csv(path, delimiter= '\t', dtype = dtype_dict, usecols=lambda column: not column.startswith("Unnamed"))
    data['seq'] = data['seq'].astype('str')
    data['cue'] = data['cue'].astype('str')
    return data 




def read_dat_files_subjs_list(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    return [read_dat_file(path + "_" + str(sub) + ".dat") for sub in subjs_list]



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['isError'] == 0) & (subj['timingError'] == 0)]



def remove_error_trials_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isTrialError'] == 0) & (subj_press['timingError'] == 0)]


def remove_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isPressError']) == 0]


def remove_next_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    error_incremented = subj_press[subj_press['isPressError'] == 1].copy()
    error_incremented['N'] = (error_incremented['N'] + 1)

    subj_press = subj_press.merge(error_incremented[['BN','TN', 'SubNum', 'N']], on = ['BN','TN', 'SubNum', 'N'], how= 'left', indicator=True)
    subj_press = subj_press[subj_press['_merge'] == 'left_only']
    return subj_press



def remove_remaining_next_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:
    error_rows = subj_press[subj_press['isPressError'] == 1]

    # Find the max N for each group where isPressError is 1
    max_n_for_error = error_rows.groupby(['BN','TN','SubNum'])['N'].min().reset_index()


    # Merge this information back to the original df to find the max N for each group in the original df
    press_with_max_n = subj_press.merge(max_n_for_error, on=['BN', 'TN', 'SubNum'], how='left', suffixes=('', '_max')).fillna(np.inf)

    # Filter out rows where N is more than the max N in the error rows
    press_filtered = press_with_max_n[press_with_max_n['N'] <= press_with_max_n['N_max']].drop(columns=['N_max'])

    return press_filtered





def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'pressTime'+str(i+1)
        col2 = 'pressTime'+str(i+2)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    subj['IPI0'] = subj['RT']


def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'cue', 'windowSize', 'digitChangePos', 'isError', 'timingError', 'isCross', 'crossTime'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)

    

    
    return subj_melted


def finger_melt_presses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'cue', 'windowSize', 'digitChangePos', 'isError', 'timingError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('press') and not _.startswith('pressTime')],
                    var_name='Press_Number', 
                    value_name='Press_Value')
    

    subj_melted['N'] = subj_melted['Press_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'cue', 'windowSize', 'digitChangePos', 'isError', 'timingError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('response')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = subj_melted['Response_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_presses = finger_melt_presses(subj)
    melt_responses = finger_melt_responses(subj)
    merged_df = melt_IPIs.merge(melt_presses, on = ['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'cue', 'windowSize',
                                               'digitChangePos', 'isError', 'timingError', 'N'])\
                                               .merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'cue', 'windowSize',
                                               'digitChangePos', 'isError', 'timingError', 'N'] )

    return add_press_error(merged_df)


def add_press_error(merged_df):
    merged_df['isPressError'] = ~(merged_df['Press_Value'] == merged_df['Response_Value'])
    return merged_df


def is_trained_seq(row: pd.Series):
    """
    Determine if the sequence was in the trained sequence group
    """

    return row['seq'] in g_sequences[row['group']][:2]

def is_untrained_seq(row: pd.Series):
    """
    Determine if the sequence was in the untrained sequence group
    """

    return row['seq'] in g_sequences[row['group']][2:]


def is_rand_seq(row: pd.Series):
    """
    Determine if the sequence was in the rand sequence group
    """

    return row['seq'] not in g_sequences[row['group']]


def is_digit_changed(row: pd.Series):
    """
    Determines if a digit change happened to that trial/press comparing to train
    """
    return (row['seq'] != row['cue']) & row['is_trained_seq']


def correct_error_trial_IPI(row: pd.Series):
    """
    Maps IPIs of error trials to infinity to account for speed/accuracy tradeoff
    """

    if row['isTrialError']:
        return np.inf
    else:
        return row['IPI_Value']
    


def correct_error_presses(row: pd.Series):
    """
    Maps IPIs of error presses to infinity to account for speed/accuracy tradeoff
    """
    if row['isPressError']:
        return np.inf
    else:
        return row['IPI_Value']
    

def correct_error_trial(row: pd.Series):
    """
    Maps MT and ET of error trials to infinity to account for speed/accuracy tradeoff
    """

    if row['isError']:
        return np.inf
    else:
        return row['norm_MT']
    

def check_window_around_change_press(row: pd.Series):
    """
    Determines if a press is around a certain window around change position
    """
    return (row['IPI_Number'] in ['IPI' + str(x) for x in range(row['digitChangePos'] -2 , row['digitChangePos'] + 3)])
    


def finger_melt_Forces(subjs_force: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each Finger Force in the whole experiment adding two columns, "Force_Number" determining the order of Force
    and "Force_Value" determining the time of Force
    """

    
    subj_force_melted = pd.melt(subjs_force, 
                    id_vars=['state', 'timeReal', 'time', 'BN', 'TN', 'SubNum', 'group', 'hand', 'isTrain', 'seq', 'cue', 'windowSize', 
                             'digitChangePos', 'isError', 'timingError', 'isCross', 'crossTime', 'norm_MT'], 
                    value_vars =  [_ for _ in subjs_force.columns if _.startswith('force')],
                    var_name='Force_Number', 
                    value_name='Force_Value')
    
    return subj_force_melted
