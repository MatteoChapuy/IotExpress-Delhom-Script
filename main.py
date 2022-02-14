import sys
import numpy as np
import pandas as pd
import json
import time


def get_timestamp(df):
    # >> check IoTExpress format <<
    df['timestamp'] = pd.to_datetime(df['timestamp'], format=' %d/%m/%Y %H:%M').astype('datetime64[s]')
    return df


def get_wind_velocity_class(df):
    df['wind_velocity_class'] = pd.cut(df['wind_velocity_mean'],
                              bins=[0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                                        10.5, 11.5, 12.5, 100.0],
                              include_lowest=True,
                              labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'excluded'])
    return df


def get_median_ambient(df):
    Lamb_med = []
    Vamb_mean = []
    Namb = []
    vk_ = np.arange(1, 13)
    for k, vk in enumerate(vk_):
        ambient = df.loc[(df['chronogram'] == 1) & (df['wind_velocity_class'] == vk)]
        Namb.append(len(ambient))
        Lamb_med.append(round(ambient['L50eq'].median(), 2))
        Vamb_mean.append(round(ambient['wind_velocity_mean'].mean(), 2))
    return Lamb_med, Vamb_mean, Namb


def get_median_residual(df):
    Lres_med = []
    Vres_mean = []
    Nres = []
    vk_ = np.arange(1, 13)
    for k, vk in enumerate(vk_):
        residual = df.loc[(df['chronogram'] == 0) & (df['wind_velocity_class'] == vk)]
        Nres.append(len(residual))
        Lres_med.append(round(residual['L50eq'].median(), 2))
        Vres_mean.append(round(residual['wind_velocity_mean'].mean(), 2))
    return Lres_med, Vres_mean, Nres


def get_centered_median(L_med, v_mean, n):
    vk_ = np.arange(1, 13)
    L_med_cent = []
    for k, vk in enumerate(vk_):
        try:
            if v_mean[k] >= vk:
                if n[k+1] >= 10:
                    t = (vk-v_mean[k])/(v_mean[k+1]-v_mean[k])
                    L_med_cent.append(round((1-t)*L_med[k] + t*L_med[k+1], 2))
                elif n[k+1] < 10:
                    L_med_cent.append(L_med[k])
            elif v_mean[k] < vk:
                if n[k-1] >= 10:
                    t = (vk-v_mean[k])/(v_mean[k-1]-v_mean[k])
                    L_med_cent.append(round((1 - t) * L_med[k] + t * L_med[k + 1], 2))
                elif n[k-1] < 10:
                    L_med_cent.append(L_med[k])
            else:
                L_med_cent.append(np.nan)
        except IndexError as e:
            print(e, k)
    return L_med_cent


def check_ambient_criteria(Lamb_med_cent, Namb):
    vk_ = np.arange(1, 13)
    crit_amb = []
    for k, vk in enumerate(vk_):
        if Lamb_med_cent[k] > 35.0 and Namb[k] > 10:
            crit_amb.append(True)
        else:
            crit_amb.append(False)
    return crit_amb


def check_residual_criteria(Lres_med_cent):
    vk_ = np.arange(1, 13)
    crit_res = []
    for k, vk in enumerate(vk_):
        if Lres_med_cent[k] > 10:
            crit_res.append(True)
        else:
            crit_res.append(False)
    return crit_res


def get_emergence(df):
    L_emerg = []
    for k in np.arange(1,13):
        if (df['Camb'][k-1]) and (df['Cres'][k-1]):
            L_emerg.append(df['Lamb_med_cent'][k-1] - df['Lres_med_cent'][k-1])
        else:
            L_emerg.append(np.nan)
    return L_emerg


if __name__ == '__main__':

    ### Format Input ###
    ts = time.time()
    rawConfiguration = sys.stdin.readline()
    configuration = json.loads(rawConfiguration)
    arguments = configuration['arguments']

    rawDataStream = sys.stdin.readline()
    dataStream = json.loads(rawDataStream)
    inputHeaders = {val: idx for idx, val in enumerate(
        dataStream['data']['object']['data']['Headers'])}
    inputData = dataStream['data']['object']['data']['Data']
    inputDuration = dataStream['duration']


    ### Compute data ### to do Matteo
    # function specific treatement
    ###
    input_df = {snValue: [] for snValue in inputData}

    for snValue in inputData:
        for row in inputData[snValue]:
            df = get_timestamp(input_df)
            df = get_wind_velocity_class(df)
            Lres_med, Vres_mean, Nres = get_median_residual(df)
            Lamb_med, Vamb_mean, Namb = get_median_ambient(df)
            Lres_med_cent = get_centered_median(Lres_med, Vres_mean, Nres)
            Lamb_med_cent = get_centered_median(Lamb_med, Vamb_mean, Namb)
            crit_amb = check_ambient_criteria(Lamb_med_cent, Namb)
            crit_res = check_residual_criteria(Lres_med_cent)
            output = {'Vk': np.arange(1, 13),
                    'Namb': Namb,
                    'Vamb_mean': Vamb_mean,
                    'Lamb_med': Lamb_med,
                    'Lamb_med_cent': Lamb_med_cent,
                    'Nres': Nres,
                    'Vres_mean': Vres_mean,
                    'Lres_med': Lres_med,
                    'Lres_med_cent': Lres_med_cent,
                    'Camb': crit_amb,
                    'Cres': crit_res}
            input_df[snValue].append([row[inputHeaders['tss']],
                                row[inputHeaders['tse']], output['Namb']])


    ### Generate output ###
    output = {
        'data': {
            'object': {
                'data': {
                    # TODO : define ouptput names
                    'Headers': ["tss", "tse", "mediane_bruit_ambiant_3"],
                    'Data': input_df
                }
            }
        },
        'duration': inputDuration,
        'scriptDuration': int((time.time()-ts)*1000),
        'input': dataStream
    }
    print(json.dumps(output))

    


    """with open(sys.argv[1], 'r+') as f:
        data = json.load(f)*/
    input_df = pd.DataFrame(data)"""
    
    """df_output = pd.DataFrame(output)
    df_output['Lemergence'] = get_emergence(df_output)
    print(df_output)
    df_output.to_json(path_or_buf='output_data/output_data.json', orient='index', indent=4)"""







