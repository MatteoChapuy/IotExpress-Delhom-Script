import sys
import numpy as np
import pandas as pd
import json
import time
pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
### Inputs ###
#
# configuration : {
#   arguments : { [name:string] : string}
# }
#
# dataStream : {
#   data : {
#       object: {
#           data : {
#               Headers : ["tss" , "tse", "input", "param1", ..."paramN"],
#               Data : { [snValue: number] : [
#                               [<tss_1>, <tse_1>, <input_1>, <param1_1>, ...<paramN_1>],
#                               [<tss_2>, <tse_2>, <input_2>, <param1_2>, ...<paramN_2>],
#                               ...
#                               [<tss_n>, <tse_n>, <input_n>, <param1_n>, ...<paramN_n>],
#                           ]
#               },
#               filterParameters : {
#                   startDateUTC: number;
#                   endDateUTC: number;
#                   timeFrameAggregation: string | number;
#                   filters_SNS: SQLFilter[];
#                   filters_PARAMETERS: SQLFilter[];
#                   filters_CONTEXTUALS: number[];
#                   userTimeZone: string;
#               },
#           }
#       }
#   },
#   duration: number (SQL request duration in ms)
# }
#
# Note: input and params can be recieved in any order.
#       A dict "inputHeaders" is created to map input and params with their positions
####


### Output ###
# output : {
#   data : {
#       object: {
#           data : {
#               Headers : ["tss" , "tse", <name_of_first_output1>, ...<name_of_first_outputn>],
#               Data : { [snValue: number] : number[][] }
#           }
#       }
#   },
#   duration: number,
#   scriptDuration: number
# }
###


def get_timestamp(df):
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', origin='unix')
    df['hour'] = df['date'].dt.hour
    return df


def get_wind_velocity_class(df):
    df['wind_velocity_class'] = pd.cut(df['wind_velocity_mean'],
                              bins=[0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                                        10.5, 11.5, 12.5, 100.0],
                              include_lowest=True,
                              labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'excluded'])
    return df


def filter_wind_direction(df, dir_min, dir_max):
    return df[(df.wind_direction >= dir_min) & (df.wind_direction <= dir_max)]


def filter_hour(df, t1, t2):
    return df[(df.hour >= t1) & (df.hour <= t2)]


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

    ### Load Inputs dev Delhom ###
    # ts = time.time()
    # with open('input_data/arguments.json', 'r') as f:
    #     arguments = json.load(f)
    # with open('input_data/input.json', 'r') as f:
    #     dataStream = json.load(f)
    # inputHeaders = {val: idx for idx, val in enumerate(
    #     dataStream['data']['object']['data']['Headers'])}
    # inputData = dataStream['data']['object']['data']['Data']
    # inputDuration = dataStream['duration']



    # function specific treatement
    output_dict = {snValue: [] for snValue in inputData}

    header_tab = dict(zip(inputHeaders, ['timestamp', 'tse', 'chronogram', 'wind_direction', 'wind_velocity_mean', 'L50eq']))
    for snValue in inputData:
        df = pd.DataFrame(inputData[snValue], columns=header_tab.values())
        df = df.drop(['tse'], axis=1)
        df = get_wind_velocity_class(df)
        df = get_timestamp(df)
        df = filter_hour(df, int(arguments["Heure_min"]), int(arguments["Heure_max"]))
        df = filter_wind_direction(df, int(arguments["Direction_min"]), int(arguments["Direction_max"]))
        Lres_med, Vres_mean, Nres = get_median_residual(df)
        Lamb_med, Vamb_mean, Namb = get_median_ambient(df)
        Lres_med_cent = get_centered_median(Lres_med, Vres_mean, Nres)
        Lamb_med_cent = get_centered_median(Lamb_med, Vamb_mean, Namb)
        crit_amb = check_ambient_criteria(Lamb_med_cent, Namb)
        crit_res = check_residual_criteria(Lres_med_cent)
        output_sn = {'Vk': np.arange(1, 13),
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
        df_output_sn = pd.DataFrame(output_sn)
        df_output_sn['Lemergence'] = get_emergence(df_output_sn)
        sublist = [[dataStream['data']['object']['filterParameters']['startDateUTC']],
                   [dataStream['data']['object']['filterParameters']['endDateUTC']],
                   df_output_sn['Namb'].fillna(""),
                   df_output_sn['Vamb_mean'].fillna(""),
                   df_output_sn['Lamb_med_cent'].fillna(""),
                   df_output_sn['Nres'].fillna(""),
                   df_output_sn['Vres_mean'].fillna(""),
                   df_output_sn['Lres_med_cent'].fillna(""),
                   df_output_sn['Lemergence'].fillna("")]
        output_dict[snValue] = [[value for s in sublist for value in s]]

    ### Generate output ###
    output = {
        'data': {
            'object': {
                'data': {
                      # TODO : define ouptput names
                    'Headers': ["tss",
                                "tse",
                                "Namb_1",
                                "Namb_2",
                                "Namb_3",
                                "Namb_4",
                                "Namb_5",
                                "Namb_6",
                                "Namb_7",
                                "Namb_8",
                                "Namb_9",
                                "Namb_10",
                                "Namb_11",
                                "Namb_12",
                                "Wamb_mean_1",
                                "Wamb_mean_2",
                                "Wamb_mean_3",
                                "Wamb_mean_4",
                                "Wamb_mean_5",
                                "Wamb_mean_6",
                                "Wamb_mean_7",
                                "Wamb_mean_8",
                                "Wamb_mean_9",
                                "Wamb_mean_10",
                                "Wamb_mean_11",
                                "Wamb_mean_12",
                                "Lamb_med_cent_1",
                                "Lamb_med_cent_2",
                                "Lamb_med_cent_3",
                                "Lamb_med_cent_4",
                                "Lamb_med_cent_5",
                                "Lamb_med_cent_6",
                                "Lamb_med_cent_7",
                                "Lamb_med_cent_8",
                                "Lamb_med_cent_9",
                                "Lamb_med_cent_10",
                                "Lamb_med_cent_11",
                                "Lamb_med_cent_12",
                                "Nres_1",
                                "Nres_2",
                                "Nres_3",
                                "Nres_4",
                                "Nres_5",
                                "Nres_6",
                                "Nres_7",
                                "Nres_8",
                                "Nres_9",
                                "Nres_10",
                                "Nres_11",
                                "Nres_12",
                                "Wres_1",
                                "Wres_2",
                                "Wres_3",
                                "Wres_4",
                                "Wres_5",
                                "Wres_6",
                                "Wres_7",
                                "Wres_8",
                                "Wres_9",
                                "Wres_10",
                                "Wres_11",
                                "Wres_12",
                                "Lres_cent_med_1",
                                "Lres_cent_med_2",
                                "Lres_cent_med_3",
                                "Lres_cent_med_4",
                                "Lres_cent_med_5",
                                "Lres_cent_med_6",
                                "Lres_cent_med_7",
                                "Lres_cent_med_8",
                                "Lres_cent_med_9",
                                "Lres_cent_med_10",
                                "Lres_cent_med_11",
                                "Lres_cent_med_12",
                                "Lemergence_1",
                                "Lemergence_2",
                                "Lemergence_3",
                                "Lemergence_4",
                                "Lemergence_5",
                                "Lemergence_6",
                                "Lemergence_7",
                                "Lemergence_8",
                                "Lemergence_9",
                                "Lemergence_10",
                                "Lemergence_11",
                                "Lemergence_12",
                                ],
                    'Data': output_dict
                }
            }
        },
        'duration': inputDuration,
        'scriptDuration': int((time.time()-ts)*1000),
        'input': dataStream
    }
    print(json.dumps(output))





