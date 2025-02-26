#--------------------------------------------------------------------------------------------
#--- Util for housekeeping a large number of JADE simulations.
#--- Useful for parameter space explorations.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from time import time
from search_info import create_df


def format_time(t):
    
    '''
    Returns a conveniently formatted time
    Input: 't' (float) duration to be formatted, in seconds
    Output: formatted time (str) in the form 'd-hh:mm:ss' or ms if t < 1
    '''

    if t < 1:
        ms = t*1000.
        if ms < 1: 
            f = '< 1 ms'
        else:
            f = str(int(ms)) + ' ms'
    else:
        t = int(round(t))
        s = t%60
        m = t//60
        h = m//60
        m = m%60
        d = h//24
        h = h%24
        f = '{:02d}:{:02d}:{:02d}'.format(h, m, s)
        if d > 0: f = str(d) + '-' + f
    
    return f


def finished(log_path):
    
    '''
    Determines whether a simulation is finished or not, and if the planet was tidally disrupted
    Input: 'log_path' (str) path to the simulation log
    Output: 
    ... finished flag (int), 0 if not finished and 1 otherwise
    ... disruption flag (int), 0 if not disrupted and 1 otherwise
    ... launched flag (int), 0 if simulation not launched and 1 otherwise
    '''
    
    f_flag, t_flag, l_flag = np.nan, np.nan, 0

    if not os.path.exists(log_path):
        return f_flag, t_flag, l_flag

    f_flag, t_flag, l_flag = 0, 0, 1
    
    t_text = '+ Planet tidally disrupted at'
    f_text = '+ Simulation successfully performed.'
    
    for line in open(log_path, 'r'):
        if t_text in line:
            t_flag = 1
            break
        if f_text in line:
            f_flag = 1
            break
    
    return f_flag, t_flag, l_flag


def finished_df(df, log_folder_path, sim_name):
    
    '''
    Adds 'f_flag', 't_flag', and 'l_flag' columns to the simulation dataframe
    Input: 
    ... 'df' (DataFrame)
    ... 'log_folder_path' (string) path to the log folder
    ... 'sim_name' (string) name of the exploration
    Output: updated 'df' (DataFrame)
    '''
    
    df['f_flag'], df['t_flag'], df['l_flag'] = zip(*df['sim_id'].map(lambda n: \
    finished('{}/{}_{:05d}.out'.format(log_folder_path, sim_name, n))))
    
    return df


def error_log(log_path):
    
    '''
    Returns the error type from the error log file
    Input: 'log_path' (str) path to the simulation error log
    Output:
    ... radius retrieval error flag (int), 0 if no error and 1 otherwise
    ... time limit error flag (int), 0 if no error and 1 otherwise
    ... dynamical integrator error flag (int), 0 if no error and 1 otherwise
    ... out-of-memory error flag (int), 0 if no error and 1 otherwise
    ... other error flag (int), 0 if no error and 1 otherwise
    '''
    
    r_err, t_err, d_err, m_err, o_err = np.nan, np.nan, np.nan, np.nan, np.nan
    
    if not os.path.exists(log_path):
        return r_err, t_err, d_err, m_err, o_err

    r_err, t_err, d_err, m_err, o_err = 0, 0, 0, 0, 0
    
    r_text = 'Radius retrieval did not converge.'
    t_text = 'TIME LIMIT'
    d_text = 'Dynamical integration failed. Step size probably becomes too small.'
    m_text = 'Some of your processes may have been killed by the cgroup out-of-memory handler.'
    
    with open(log_path, 'r') as log_file:
        text = log_file.read().rstrip()
        if r_text in text:
            r_err = 1
        elif t_text in text:
            t_err = 1
        elif d_text in text:
            d_err = 1
        elif m_text in text:
            m_err = 1
        elif len(text) > 0:
            o_err = 1
    
    return r_err, t_err, d_err, m_err, o_err


def error_sim(sim_id, f_flag, t_flag, log_folder_path, sim_name):
    
    '''
    Returns the error type of a simulation
    Input:
    ... 'sim_id' (int) id of the simulation
    ... 'f_flag' (int) finished flag
    ... 't_flag' (int) tidal disruption flag
    ... 'log_folder_path' (string) path to the log folder
    ... 'sim_name' (string) name of the exploration
    Output:
    ... radius retrieval error flag (int), 0 if no error and 1 otherwise
    ... time limit error flag (int), 0 if no error and 1 otherwise
    ... dynamical integrator error flag (int), 0 if no error and 1 otherwise
    ... out-of-memory error flag (int), 0 if no error and 1 otherwise
    ... other error flag (int), 0 if no error and 1 otherwise
    '''

    if np.isnan(f_flag):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    sim_id = int(sim_id)
    f_flag = int(f_flag)
    t_flag = int(t_flag)
    flag = np.max([f_flag, t_flag])
    
    if flag == 1:
        r_err, t_err, d_err, m_err, o_err = 0, 0, 0, 0, 0
    else:
        log_path = '{}/{}_{:05d}.err'.format(log_folder_path, sim_name, sim_id)
        r_err, t_err, d_err, m_err, o_err = error_log(log_path)
        
    return r_err, t_err, d_err, m_err, o_err


def error_df(df, log_folder_path, sim_name):
    
    '''
    Adds error flag columns to the simulation dataframe
    Input: 
    ... 'df' (DataFrame)
    ... 'log_folder_path' (string) path to the log folder
    ... 'sim_name' (string) name of the exploration
    Output: updated 'df' (DataFrame)
    '''
    
    df['r_err'], df['t_err'], df['d_err'], df['m_err'], \
    df['o_err'] = zip(*df.apply(lambda x: error_sim(x['sim_id'], x['f_flag'], x['t_flag'], 
                      log_folder_path, sim_name), axis=1))
    
    return df


def ongoing_sim(l_flag, f_flag, t_flag, r_err, t_err, d_err, m_err, o_err):

    '''
    Determines whether a simulation is still ongoing
    Input: all of the flags
    Output: ongoing flag (int), 0 if not ongoing and 1 otherwise
    '''

    if np.isnan(f_flag):
        return np.nan

    l_flag = int(l_flag)
    f_flag = int(f_flag)
    t_flag = int(t_flag)
    r_err = int(r_err)
    t_err = int(t_err)
    d_err = int(d_err)
    m_err = int(m_err)
    o_err = int(o_err)
    flag = np.max([1 - l_flag, f_flag, t_flag])
    err = np.max([r_err, t_err, d_err, m_err, o_err])

    if flag == 1 or err == 1:
        o_flag = 0
    else:
        o_flag = 1

    return o_flag


def ongoing_df(df):

    '''
    Adds an ongoing flag column to the simulation dataframe
    Input: 'df' (DataFrame)
    Output: updated 'df' (DataFrame)
    '''

    df['o_flag'] = df.apply(lambda x: ongoing_sim(x['l_flag'], x['f_flag'], x['t_flag'], x['r_err'], x['t_err'], 
                                                  x['d_err'], x['m_err'], x['o_err']), axis=1)

    return df


def compute_logr(df):

    '''
    Adds a log_r column to the simulation dataframe
    Input: 'df' (DataFrame)
    Output: updated 'df' (DataFrame)
    '''

    df['log_r'] = np.log(df['pert_sma']**3/df['Mpert'])

    return df


def extract_sim_name(info_path):

    '''
    Extracts the name of the exploration from the info file
    Input: 'info_path' (string) path to the info file
    Output: name of the exploration (string)
    '''

    with open(info_path, 'r') as info_file:
        line = info_file.readline()
        sim_name = line[31:-5]

    return sim_name


def construct_df(info_path, log_folder_path, update_df=None, verbose=False):

    '''
    Constructs the simulation dataframe, with all relevant columns
    Input:
    ... 'info_path' (string) path to the info file
    ... 'log_folder_path' (string) path to the log folder
    ... 'update_df' (string) path to a dataframe to be updated (optional)
    ... 'verbose' (bool) verbosity
    Output: simulation dataframe
    '''

    sim_name = extract_sim_name(info_path)

    if verbose:
        print('=========================================')
        print(' Exploration name: {}'.format(sim_name))
        print('=========================================')

    sim_df = create_df(info_path)

    sim_df['e_max']   = np.repeat(np.nan, len(sim_df))
    sim_df['n_kozai'] = np.repeat(np.nan, len(sim_df))
    sim_df['l_kozai'] = np.repeat(np.nan, len(sim_df))
    sim_df['t_trans'] = np.repeat(np.nan, len(sim_df))

    if verbose:
        print('   + Number of simulations: {}'.format(len(sim_df)))

    sim_df = compute_logr(sim_df)

    if verbose:
        print('   + Computing various flags...', end='')
        ti = time()

    sim_df = finished_df(sim_df, log_folder_path, sim_name)
    sim_df = error_df(sim_df, log_folder_path, sim_name)
    sim_df = ongoing_df(sim_df)

    if update_df is not None:
        sim_df_old = pd.read_csv(update_df)
        sim_df = sim_df.set_index('sim_id')
        sim_df = sim_df.combine_first(sim_df_old.set_index('sim_id'))
        sim_df.update(sim_df_old.set_index('sim_id')[['l_flag']])
        sim_df = sim_df.reset_index()

    sim_df.name = sim_name

    if verbose:
        et = format_time(time() - ti)
        print(' Elapsed time: {}'.format(et))
        print('=========================================')

        f_sims = list(sim_df[sim_df['f_flag'] == 1].sim_id)
        print('   + Finished simulations: {}'.format(len(f_sims)))
        print('-----------------------------------------')
        print(*f_sims, sep=', ')

        print('=========================================')

        t_sims = list(sim_df[sim_df['t_flag'] == 1].sim_id)
        print('   + Tidally disrupted simulations: {}'.format(len(t_sims)))
        print('-----------------------------------------')
        print(*t_sims, sep=', ')

        print('=========================================')

        o_sims = list(sim_df[sim_df['o_flag'] == 1].sim_id)
        print('   + Ongoing simulations: {}'.format(len(o_sims)))
        print('-----------------------------------------')
        print(*o_sims, sep=', ')

        print('=========================================')

        nl_sims = list(sim_df[sim_df['l_flag'] == 0].sim_id)
        print('   + Non-launched simulations: {}'.format(len(nl_sims)))
        print('-----------------------------------------')
        print(*nl_sims, sep=', ')

        print('=========================================')

        re_sims = list(sim_df[sim_df['r_err'] == 1].sim_id)
        print('   + Radius retrieval error flag: {}'.format(len(re_sims)))
        print('-----------------------------------------')
        print(*re_sims, sep=', ')

        print('=========================================')

        te_sims = list(sim_df[sim_df['t_err'] == 1].sim_id)
        print('   + Time limit error flag: {}'.format(len(te_sims)))
        print('-----------------------------------------')
        print(*te_sims, sep=', ')

        print('=========================================')

        de_sims = list(sim_df[sim_df['d_err'] == 1].sim_id)
        print('   + Dynamical integrator error flag: {}'.format(len(de_sims)))
        print('-----------------------------------------')
        print(*de_sims, sep=', ')

        print('=========================================')

        me_sims = list(sim_df[sim_df['m_err'] == 1].sim_id)
        print('   + Out-of-memory error flag: {}'.format(len(me_sims)))
        print('-----------------------------------------')
        print(*me_sims, sep=', ')

        print('=========================================')

        oe_sims = list(sim_df[sim_df['o_err'] == 1].sim_id)
        print('   + Other error flag: {}'.format(len(oe_sims)))
        print('-----------------------------------------')
        print(*oe_sims, sep=', ')

        print('=========================================')

    return sim_df


def get_finished(df, include_r_err=False):

    '''
    Retrieves the ids of the simulations that have finished
    Input: 
    ... 'df' (DataFrame)
    ... 'include_r_err' (bool) include radius retrieval error simulations
    Output: ids of the simulations (list of int)
    '''

    f_sims = list(df[df['f_flag'] == 1].sim_id)
    t_sims = list(df[df['t_flag'] == 1].sim_id)
    if include_r_err:
        re_sims = list(df[df['r_err'] == 1].sim_id)
    else:
        re_sims = []
    finished_id = sorted(f_sims + t_sims + re_sims)

    return finished_id


def get_to_launch(df):

    '''
    Retrieves the ids of the simulations that need to be launched
    Input: 'df' (DataFrame)
    Output: ids of the simulations (list of int)
    '''

    nl_sims = list(df[df['l_flag'] == 0].sim_id)
    re_sims = list(df[df['r_err'] == 1].sim_id)
    te_sims = list(df[df['t_err'] == 1].sim_id)
    de_sims = list(df[df['d_err'] == 1].sim_id)
    me_sims = list(df[df['m_err'] == 1].sim_id)
    oe_sims = list(df[df['o_err'] == 1].sim_id)
    to_launch_id = sorted(re_sims + te_sims + de_sims + me_sims + oe_sims + nl_sims)

    return to_launch_id


def write_dl(server_name, sim_name, finished_id, downloaded_id):

    '''
    Writes a bash line to download simulation results
    Input:
    ... 'server_name' (string) name of the server (`yggdrasil` or `baobab`)
    ... 'sim_name' (string) name of the exploration
    ... 'finished_id' (list of int) ids of the finished simulations
    ... 'downloaded_id' (list of int) ids of the already downloaded simulations
    Output: the bash line (string)
    '''

    if server_name == 'baobab':
        login_id = 2
    elif server_name == 'yggdrasil':
        login_id = 1
    else:
        raise SystemExit('Invalid server name. Must be `yggdrasil` or `baobab`.')
    to_download_id = [sim for sim in finished_id if sim not in downloaded_id]
    to_download_id = sorted(to_download_id)
    bash_line = 'scp -r oattia@login{}.{}.hpc.unige.ch:~/JADE/saved_data/{}/\{{'.format(login_id, server_name, sim_name)
    for sim in to_download_id: 
        bash_line += '{}_{:05d},'.format(sim_name, sim)
    bash_line = bash_line[:-1] + '\} ./'

    return bash_line


def write_launch(sim_name, to_launch_id):

    '''
    Writes a bash line to launch simulations
    Input:
    ... 'sim_name' (string) name of the exploration
    ... 'to_launch_id' (list of int) ids of the simulations
    Output: the bash line (string)
    '''

    bash_line = 'declare -a StringArray=('
    to_launch_id = sorted(to_launch_id)
    for sim in to_launch_id: 
        bash_line += '"{}/{}_{:05d}.bash" '.format(sim_name, sim_name, sim)
    bash_line = bash_line[:-1] + ')'
    
    return bash_line


def kozai_npz(sim_name, npz_folder):
    
    '''
    Determines important Kozai-related quantities from npz files
    Input: 
    ... 'sim_name' (string) name of the exploration
    ... 'npz_folder' (str) path to the npz folder
    Output:
    ... Maximum reached eccentricity (float)
    ... No Kozai flag (int), 1 if no Kozai happened and 0 otherwise
    ... Long Kozai flag (int), 1 if Kozai didn't end yet and 0 otherwise
    ... Transition timescale (float) in Gyr
    '''
    
    KOZAI_E_THRESH = 0.05
    KOZAI_DA_THRESH = 0.05

    e_max, n_kozai, l_kozai, t_trans = np.nan, np.nan, np.nan, np.nan
    
    if not os.path.isdir(npz_folder):
        return e_max, n_kozai, l_kozai, t_trans
    
    _npz = os.listdir(npz_folder)
    npz = sorted([npz_folder + f for f in _npz \
                  if len(f) == len(sim_name) + 14 and f.startswith(sim_name) and f.endswith('.npz')])
    
    if len(npz) == 0:
        return e_max, n_kozai, l_kozai, t_trans
    
    _e_max = 0.
    a_all = []
    t_all = []
    
    for npz_path in npz:
        try:
            npz_file = np.load(npz_path, mmap_mode='r', allow_pickle=True)
        except:
            return e_max, n_kozai, l_kozai, t_trans
        _e_max = np.max([np.max(npz_file['e']), _e_max])
        a_all += list(npz_file['a'])
        t_all += list(npz_file['t'])
        
    e_max = _e_max
        
    if e_max < KOZAI_E_THRESH:
        n_kozai, l_kozai = 1, 0
    else:
        da = np.abs(np.max(a_all) - np.min(a_all))
        if da < KOZAI_DA_THRESH:
            n_kozai, l_kozai = 0, 1
        else:
            dadt = np.nan_to_num(np.abs(np.diff(a_all, append=a_all[-1])/np.diff(t_all, append=t_all[-1])))
            t_trans = t_all[np.argmax(dadt)]*1e-9
            n_kozai, l_kozai = 0, 0
    
    return e_max, n_kozai, l_kozai, t_trans


def kozai_sim(output_folder, sim_name, sim_id, f_flag, e_max, n_kozai, l_kozai, t_trans, verbose=False):
    
    '''
    Determines important Kozai-related quantities for a simulation
    Input: 
    ... 'output_folder' (str) path to the global output folder
    ... 'sim_name' (string) name of the exploration
    ... 'sim_id' (int) id of the simulation
    ... 'f_flag' (int) finished flag
    Output:
    ... Maximum reached eccentricity (float)
    ... No Kozai flag (int), 1 if no Kozai happened and 0 otherwise
    ... Long Kozai flag (int), 1 if Kozai didn't end yet and 0 otherwise
    ... Transition timescale (float) in Gyr
    '''
    
    sim_id = int(sim_id)
    
    if f_flag == 0:
        e_max, n_kozai, l_kozai, t_trans = np.nan, np.nan, np.nan, np.nan
    else:
        if not (np.isnan(e_max) or np.isnan(n_kozai) or np.isnan(l_kozai) or np.isnan(t_trans)):
            return e_max, n_kozai, l_kozai, t_trans
        if verbose:
            ti = time()
        npz_folder = '{}/{}_{:05d}/'.format(output_folder, sim_name, sim_id)
        e_max, n_kozai, l_kozai, t_trans = kozai_npz(sim_name, npz_folder)       
        if verbose: 
            et = format_time(time() - ti)
            print('   + Successfully done for simulation #{:05d}. Elapsed time: {}'.format(sim_id, et))
        
    return e_max, n_kozai, l_kozai, t_trans


def kozai_df(df, output_folder, verbose=False):
    
    '''
    Adds Kozai-related columns to the simulation dataframe
    Input: 
    ... 'df' (DataFrame)
    ... 'output_folder' (str) path to the global output folder
    Output: updated 'df' (DataFrame)
    '''
    
    if verbose:
        print('=========================================')
        print(' Computing Kozai-related quantities for exploration {}'.format(df.name))
        print(' {} finished simulations found'.format(len(df[df['f_flag'] == 1])))
        print('=========================================')
        ti = time()

    df['e_max'], df['n_kozai'], df['l_kozai'], \
    df['t_trans'] = zip(*df.apply(lambda x: kozai_sim(output_folder, df.name, x['sim_id'], x['f_flag'], 
                                                      x['e_max'], x['n_kozai'], x['l_kozai'], x['t_trans'], verbose=verbose), axis=1))

    if verbose:
        et = format_time(time() - ti)
        print('=========================================')
        print(' Total elapsed time: {}'.format(et))
        print('=========================================')
    
    return df