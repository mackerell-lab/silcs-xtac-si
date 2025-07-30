#!/usr/bin/env python
#
# Erik Nordquist, enord
# big script to loop over lots of protac data and calculate/plot correlations
# options to specify what data to use, etc

import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.special import logsumexp
from scipy.stats import zscore
import numpy as np
import argparse
import os,csv,ast
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut
from pprint import pprint
import itertools, random
from collections import defaultdict

# https://arxiv.org/abs/2107.02270 Petroff, Accessible Color Sequences for Data Visualization
color_list = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
phi = 1.618

RT = 0.6
Beta = 1/RT
M_nM=10**(-9)
N_Dihe_States = 4

# $SILCSBIODIR/utils/python/protac/calc_protac_properties.py
rdkit_base_names = ['CrippenClogP', 'CrippenMR', 'exactmw', 'tpsa']
integer_descriptor_names = ['NumRotatableBonds', 'NumHeavyAtoms', 'lipinskiHBD', 'lipinskiHBA']
rdkit_base_names += integer_descriptor_names
rdkit_split_names = [prefix+name for name in rdkit_base_names for prefix in ['Target ','Ligase ','Linker ']]
rdkit_all_names = rdkit_base_names + rdkit_split_names
rdkit_linker_names = [prefix+name for name in rdkit_base_names for prefix in ['Linker ']]

def read_pair_list(filename):
  with open(filename, newline='\n') as file:
    array = [tuple(line.strip().split()) for line in file]
  return array

def read_protacs_data(pair_file):
  activity_array = []
  prop_array = []
  for ligase, target in read_pair_list(pair_file):
    init_index = 2 # 0:'TARGET-PROTAC' properties dump, 1:'binding-data' data dump, 2-N:'data[0-9]'
    try:
      Nsheets = len(pd.read_excel('./protac-db/'+target+'/'+target+'-'+ligase+'.xlsx', sheet_name=None))
      for sheet_index in range(init_index, Nsheets):
        df = pd.read_excel('./protac-db/'+target+'/'+target+'-'+ligase+'.xlsx', sheet_name=sheet_index)
        df['Target'] = target
        df['Ligase'] = ligase
        activity_array.append(df)

      props = pd.read_csv('./protac-db/'+target+'/'+ligase+'/protac_properties.csv')
      props['Target'] = target
      props['Ligase'] = ligase
      prop_array.append(props)

    except FileNotFoundError: # file doesn't exist b/c that combo doesn't have data
      continue

  activity = pd.concat(activity_array)
  properties = pd.concat(prop_array)

  # need to enforce that these are all strings to avoid merging issues - the protacs I've added look like p* or c*
  activity['Compound ID']   = activity['Compound ID'].astype(str)
  properties['Compound ID'] = properties['Compound ID'].astype(str)
  merged = pd.merge(activity, properties, on=['Compound ID', 'Target', 'Ligase'])

  return merged, activity, properties

def rmse_nan(x, y):
  x = np.array(x)
  y = np.array(y)
  nan = np.logical_or(np.isnan(x), np.isnan(y))
  return np.sqrt(np.mean((x[~nan]-y[~nan])**2))

def plot_regression(x, y, ax, c='k'):
  x = np.array(x)
  y = np.array(y)
  nan = np.logical_or(np.isnan(x), np.isnan(y))
  try:
    m, b, r, _, _ = scipy.stats.linregress(x[~nan], y[~nan])
  except ValueError:
    m,b,r=0,1,0
  y_bar = m*x[~nan]+b
  ax.plot(x[~nan], y_bar, color=c, linewidth=1)

def zscore_rmse_nan(x, y):
  x = np.array(x)
  y = np.array(y)
  nan = np.logical_or(np.isnan(x), np.isnan(y))
  x_zscore = zscore(x[~nan])
  y_zscore = zscore(y[~nan])
  return rmse_nan(x_zscore,y_zscore)
  
# prepare arrays of labels in pairs, the full activity label, e.g. 'IC50' and 'IC50 (nM, Y assay in X cells)'
def get_mini_labels(act, act_type='all'): # todo: get this act_type working for cells/not cells
  activity_columns = []
  mini_labels = []
  for name, col in act.items():
    size = col.dropna(axis=0, how='any').shape[0]
    for minlbl in activity_labels:
      if name.startswith(minlbl) and size > 2:
        activity_columns.append(name)
        mini_labels.append(minlbl)
  return activity_columns, mini_labels

def parse_array_string(s):
  try:
    # Remove brackets and newlines, then split and convert
    s_clean = s.replace('[', '').replace(']', '').replace('\n', ' ')
    return np.fromstring(s_clean, sep=' ')
  except Exception:
    return np.nan  # or return s if you want to debug it

def get_merged_lgfe_props_activities(data, agg_log):
  merge = pd.DataFrame()
  for ligase,target in read_pair_list(pair_file):
    try:
      file=f"ppi-{ligase}-{target}/{agg_log}"
      _, ext = os.path.splitext(file)
      if os.path.exists(file):
        if   ext == '.pkl': agg = pd.read_pickle(file)
        elif ext == '.csv': 
          agg = pd.read_csv(file, engine='python', quoting=csv.QUOTE_MINIMAL)
          for col in ['LGFE','Ligase LGFE','Target LGFE','Linker LGFE']: 
            if isinstance(agg[col].iloc[0], str) and agg[col].iloc[0].strip().startswith('['):
              agg[col] = agg[col].apply(parse_array_string)  
          print(agg['LGFE'][0][0]+1.0)
        agg['Target'] = target
        agg['Ligase'] = ligase
        merge = pd.concat([agg, merge])
    except Exception as e:  print(f"Error processing file {file}: {e}")
  data = pd.merge(data, merge, on=['Compound ID', 'Target', 'Ligase'])
  return data

def get_ppips():
  ppips = pd.DataFrame()
  for ligase,target in read_pair_list(pair_file):
    file=f"ppi-{ligase}-{target}/ppips_clusters_sorted.log"
    if os.path.exists(file):
      new_df = pd.read_csv(file, sep=r"\s+", skiprows=1)
      ppips = pd.concat([ppips, new_df])
  return ppips

def boltzmann_average(energies):
  weights = np.exp(-Beta * energies)
  return np.sum(energies * weights)/np.sum(weights)

# Calculate Pearson and Spearman Correlation Coefs, omitting nan
def get_pearson_spearman_nan(x, y):
  x = np.array(x)
  y = np.array(y)
  nan = np.logical_or(np.isnan(x), np.isnan(y))
  pearsonr, pr = scipy.stats.pearsonr(x[~nan], y[~nan])
  spearmanr, ps = scipy.stats.spearmanr(x, y, nan_policy='omit')
  return pearsonr, pr, spearmanr, ps

def percent_correct(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  c=0
  total=0
  n=len(y_true)
  if n < 2: return np.nan
  for i in range(n):
    for j in range(i+1, n):
      total+=1
      if ((y_true[i] < y_true[j]) and (y_pred[i] < y_pred[j])) or \
         ((y_true[i] > y_true[j]) and (y_pred[i] > y_pred[j])):
        c += 1
  return c/total

def findCorrelation(corr, threshold=0.9, verbose=False):
  """
  Takes a df correlation matrix and returns the columns which should be removed.
  NOTE: modified assuming first columns are preferred, and recognizing that average correlation is noisy, small differences aren't reliable
  Python implementation of R function `findCorrelation()`.
  https://www.rdocumentation.org/packages/caret/topics/findCorrelation
  https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R
  https://stackoverflow.com/questions/44889508/remove-highly-correlated-columns-from-a-pandas-dataframe/75379515#75379515
  """
  corr = corr.abs()
  avg = corr.mean()
  x = corr.loc[(*[avg.sort_values(ascending=False).index]*2,)]
  difference_threshold = 0.1
  if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any(): x = x.astype(float)
  x.values[(*[np.arange(len(x))]*2,)] = np.nan
  deletecol = []
  for ix, i in enumerate(x.columns[:-1]):
    for j in x.columns[ix+1:]:
      if x.loc[i, j] > threshold:
        if x[i].mean() > x[j].mean() and abs(x[i].mean()-x[j].mean()) > difference_threshold:
          deletecol.append(i)
          if verbose: print(f'Dropping {i}, keeping {j}')
          x.loc[i] = x[i] = np.nan
        else:
          deletecol.append(j)
          if verbose: print(f'Dropping {j}, keeping {i}')
          x.loc[j] = x[j] = np.nan
  return deletecol

# Calculate PROTAC score functions
def calc_scores(merge, ppips):
  def trim(x):
    return x[:args.topn]
  merge['LGFE'] = merge['LGFE'].apply(trim)
  merge['Target LGFE'] = merge['Target LGFE'].apply(trim)
  merge['Ligase LGFE'] = merge['Ligase LGFE'].apply(trim)
  merge['Linker LGFE'] = merge['Linker LGFE'].apply(trim)
  # make the name more convenient
  merge['Ref_Target_LGFE'] = merge['Ref Target LGFE']
  merge['Ref_Ligase_LGFE'] = merge['Ref Ligase LGFE']

  merge['LGFE_min'] = merge['LGFE'].apply(np.min)
  merge['LGFE_argmin'] = merge['LGFE'].apply(np.argmin) # use to extract corresponding PGFE, Warhead LGFE, etc.
  merge['LGFE_boltz'] = merge['LGFE'].apply(boltzmann_average)
  merge['LGFE_min_PGFE'] = merge['LGFE'].apply(lambda x: x[0])

  # Extraction of features based on the argmin of LGFE (total)
  merge['Target_LGFE_min'] = merge.apply(lambda row: row['Target LGFE'][row['LGFE_argmin']], axis=1)
  merge['Ligase_LGFE_min'] = merge.apply(lambda row: row['Ligase LGFE'][row['LGFE_argmin']], axis=1)
  merge['Linker_LGFE_min'] = merge.apply(lambda row: row['Linker LGFE'][row['LGFE_argmin']], axis=1)

  merge['LWCOMDist_LGFEmin']  = merge.apply(lambda row: row['Ligase_COM_Dist'][row['LGFE_argmin']], axis=1)
  merge['TWCOMDist_LGFEmin']  = merge.apply(lambda row: row['Target_COM_Dist'][row['LGFE_argmin']], axis=1)
  merge['Restr_RMSD_LGFEmin'] = merge.apply(lambda row: row['Restr_RMSD'][row['LGFE_argmin']], axis=1)

  merge['Ligase_LGFE/COMDist'] = merge['Ligase_LGFE_min'] / merge['LWCOMDist_LGFEmin']
  merge['Target_LGFE/COMDist'] = merge['Target_LGFE_min'] / merge['TWCOMDist_LGFEmin']

  ## minimum of just target+ligase warhead LGFEs (exclude linker contribution in assessment of minimum LGFE)
  #merge['T+L LGFE'] = merge.apply(lambda row: row['Target LGFE']+merge['Ligase LGFE'], axis=1)
  #merge['T+L_LGFE_min'] = merge['T+L LGFE'].apply(np.min)
  #merge['T+L_LGFE_argmin'] = merge['T+L LGFE'].apply(np.argmin)

  # Delta_Warheads (Final - Initial, PPI - Reference)
  merge['Delta_Target'] = merge['Target_LGFE_min'] - merge['Ref_Target_LGFE'] 
  merge['Delta_Ligase'] = merge['Ligase_LGFE_min'] - merge['Ref_Ligase_LGFE'] 

  # Adding the PGFE free-energy-like score
  merge['LGFE+PGFE'] = merge['LGFE'].apply(lambda x: x+ppips['PGFE'][:args.topn].to_numpy())
  merge['LGFE+PGFE_min'] = merge['LGFE+PGFE'].apply(np.min)
  merge['LGFE+PGFE_argmin'] = merge['LGFE+PGFE'].apply(np.argmin)
  merge['LGFE+PGFE_boltz'] = merge['LGFE+PGFE'].apply(boltzmann_average)

  # Extraction of features based on the argmin of LGFE+PGFE
  merge['Target_LGFE+PGFE_min'] = merge.apply(lambda row: row['Target LGFE'][row['LGFE+PGFE_argmin']], axis=1)
  merge['Ligase_LGFE+PGFE_min'] = merge.apply(lambda row: row['Ligase LGFE'][row['LGFE+PGFE_argmin']], axis=1)
  merge['Linker_LGFE+PGFE_min'] = merge.apply(lambda row: row['Linker LGFE'][row['LGFE+PGFE_argmin']], axis=1)

  # Extraction of features based on the argmin of Restraint loci RMSD 
  merge['Restr_RMSD_argmin'] = merge['Restr_RMSD'].apply(np.argmin)
  merge['Target_Restr_RMSD_min'] = merge.apply(lambda row: row['Target LGFE'][row['Restr_RMSD_argmin']], axis=1)
  merge['Ligase_Restr_RMSD_min'] = merge.apply(lambda row: row['Ligase LGFE'][row['Restr_RMSD_argmin']], axis=1)
  merge['Linker_Restr_RMSD_min'] = merge.apply(lambda row: row['Linker LGFE'][row['Restr_RMSD_argmin']], axis=1)
  
  # Can't do overall Boltzmann average of separate terms: have to get each from argmin so that linker + target + ligase = lgfe_total
  #merge['Target_LGFE_boltz'] = merge['Target LGFE'].apply(boltzmann_average)
  #merge['Ligase_LGFE_boltz'] = merge['Ligase LGFE'].apply(boltzmann_average)
  #merge['Linker_LGFE_boltz'] = merge['Linker LGFE'].apply(boltzmann_average)

  merge['Warheads_LGFE_boltz'] = (merge['Target LGFE'] + merge['Ligase LGFE']).apply(boltzmann_average)
  merge['LGFE_linker/nrot_boltz'] = (merge['Target LGFE'] + merge['Ligase LGFE'] + merge['Linker LGFE']/merge['Linker NumRotatableBonds']).apply(boltzmann_average)

  # Specific physically-motivated correction terms
  merge['RTlnZdihe'] = RT*merge['NumRotatableBonds']*np.log(N_Dihe_States) # RTln(Nstates**Nrot) = RT*Nrot*ln(Nstates)
  merge['Linker RTlnZdihe'] = RT*merge['Linker NumRotatableBonds']*np.log(N_Dihe_States) # RTln(Nstates**Nrot) = RT*Nrot*ln(Nstates)
  merge['RTlnP'] = RT * np.log(10) * merge['Linker CrippenClogP'] 

  merge['logZdihe'] = merge['NumRotatableBonds']*np.log10(N_Dihe_States)
  merge['Linker logZdihe'] = RT*merge['Linker NumRotatableBonds']*np.log10(N_Dihe_States)
  
  return merge

def run_loocv(data, X_labels, y_label, alpha=1.0):
  data = data.copy()
  data = data[~np.isnan(data[y_label])]
  data.loc[:, y_label] = RT * np.log(data[y_label] * M_nM)

  X = data[X_labels].values
  y = data[y_label].values
  loo = LeaveOneOut()

  preds = []
  trues = []
  weights = []

  for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=alpha, solver='lsqr', fit_intercept=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    preds.append(y_pred[0])
    trues.append(y_test[0])
    weights.append(model.coef_)

  # Metrics
  r, p_r, s, p_s = get_pearson_spearman_nan(trues, preds)
  rmse = rmse_nan(trues, preds)
  PC = percent_correct(trues, preds)

  weights = np.vstack(weights)
  mean_weights = np.mean(weights, axis=0)
  stderr_weights = np.std(weights, axis=0, ddof=1) / np.sqrt(len(weights))

  return {'r':r,'s':s,'PC':PC,'y_pred':preds,'y_true':trues,'n':len(trues),'rmse':rmse, 'mean_weights':mean_weights,'stderr_weights':stderr_weights}

# looks at all loo-cv results across data sets
# also computes general model derived by weighted-average of weights 
def analyze_loocv_results(results, X_labels, data):
  def apply_rename(x):
    return sys_dict[x]
  df = pd.DataFrame(results)
  df['System'] = df['pair'].apply(apply_rename)
  df.set_index('System',inplace=True)
  new_cols = {}
  mean_weights,n_cumulative = 0,0
  for idx, row in df.iterrows():
    mean_weights += row['mean_weights']*row['n']
    n_cumulative += row['n']
    for i, (mean, stderr) in enumerate(zip(row['mean_weights'], row['stderr_weights'])):
      new_cols.setdefault(f'{i}', []).append(f'{mean:.2f} ± {stderr:.2f}')
  
  new_df = pd.DataFrame(new_cols, index=df.index)
  new_df.columns = X_labels
  weights = mean_weights/n_cumulative

  new_df.loc['Mean'] = [f"{w:.2f}" for w in weights]

  new_df.to_csv(f'loocv_rfe_results/Nlabels_{len(X_labels)}.csv')

  fig, (axp, axl) = plt.subplots(1, 2, figsize=(8,4.35))
  axl.axis('off')
  trues,preds=[],[]
  sys_index=0
  n_cumulative=0
  r_weight=0
  rmse_weight=0
  shift = 0
  fak1_count=0
  
  for (ligase, target) in read_pair_list(pair_file):
    #print(ligase, target)
    s_data = data[(data['Ligase']==ligase)&(data['Target']==target)].copy().dropna(how='all',axis=1)
    y_labels = [col for col in keep_y_labels if col in s_data.columns]
 
    for y_label in y_labels:
      s_data.loc[:, y_label] = RT * np.log(s_data[y_label] * M_nM)

      not_nan = ~np.isnan(s_data[y_label])
      X = s_data[not_nan][X_labels].values
      y = s_data[not_nan][y_label].values
      if len(y) > 3:
        #print(X,y)

        # general model:
        y_pred = X @ weights + shift
        r, p_r, s, p_s = get_pearson_spearman_nan(y,y_pred)
        # x= pred, y= true
        m = 'o' if sys_index < 10 else '^'
        label = sys_dict[f'{ligase}-{target}']+ f' (r = {r:.2f})'
        # for only fak1 there are multiples
        if y_label == 'DC50 (nM, Degradation of FAK in PC3 cells after 24 h treatment)':
          label = sys_dict[f'{ligase}-{target}']+r'$^{(a)}$' +f' (r = {r:.2f})'
        if y_label == 'DC50 (nM, Degradation of FAK in A549 cells after 24 h treatment)':
          label = sys_dict[f'{ligase}-{target}']+r'$^{(b)}$' +f' (r = {r:.2f})'
        
        axp.scatter(y_pred, y, c=color_list[sys_index%10],label=label, marker=m)
        #plot_regression(y_pred, y, axp, c=color_list[sys_index%10])
        trues.append(y)
        preds.append(y_pred)
        r_weight    += r*len(y)
        rmse_weight += rmse_nan(y, y_pred)*len(y)
        n_cumulative+=len(y)
        sys_index+=1

  axp.set_xlabel('Predicted (kcal/mol)', fontweight='bold')
  axp.set_ylabel(r'RT$\bf{\cdot}$lnDC$\bf{_{50}}$ (kcal/mol)', fontweight='bold')
  axp.grid()
  handles,labels=axp.get_legend_handles_labels()
  axl.legend(handles,labels,fontsize=13, loc='center left')

  d = np.concatenate((np.concatenate(trues), np.concatenate(preds)), axis=0)
  y_min, y_max = d.min()-1, d.max()+1
  limits=np.arange(y_min,y_max+0.5,0.5)
  axp.plot(limits,limits,color='k', linestyle='--', linewidth=1)
  axp.fill_between(limits, limits-1, limits+1, alpha=0.2, color='gray')
  axp.set_ylim([y_min, y_max])
  axp.set_xlim([y_min, y_max])

  plt.tight_layout()
  r_weight/=n_cumulative
  rmse_weight/=n_cumulative
  print(f"Aggregated Model: r = {r_weight:.2f} RMSE = {rmse_weight:.2f}")
  #print(new_df)
  plt.tight_layout()
  if not args.q and not args.randomize:
    plt.savefig(f"figs/general_model/Nmetrics_{len(X_labels)}.svg")
    plt.show()
  plt.close()
  return r_weight, rmse_weight

# Perform LeaveOneOut CV, LOOCV
# plot
def plot_loocv(data, X_labels, y_label, alpha=1.0):
  data = data.copy()
  data = data[~np.isnan(data[y_label])]
  data.loc[:, y_label] = RT * np.log(data[y_label] * M_nM)

  X = data[X_labels].values
  y = data[y_label].values
  loo = LeaveOneOut()

  preds = []
  trues = []
  weights = []

  for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=alpha, solver='lsqr', fit_intercept=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    preds.append(y_pred[0])
    trues.append(y_test[0])
    weights.append(model.coef_)

  # Metrics
  r, p_r, s, p_s = get_pearson_spearman_nan(trues, preds)
  PC = percent_correct(trues, preds)

  print(f"Pearson r = {r:.3f}, Spearman = {s:.3f}, Percent Correct = {PC:.2f}")

  # Plotting
  #fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
  fig, ax0 = plt.subplots(1, 1)

  # this is for plot boundaries
  y_min, y_max = np.inf, -np.inf
  if np.min(trues) < y_min: y_min = np.min(trues)
  if np.max(trues) > y_max: y_max = np.max(trues)
  if np.min(preds)  < y_min: y_min = np.min(preds)
  if np.max(preds)  > y_max: y_max = np.max(preds)
  # Diagonal x=y and +/- 1 kcal/mol
  limits=np.arange(y_min-0.5,y_max+1.0,0.5)
  ax0.plot(limits,limits,color='k', linestyle='--', linewidth=1)
  ax0.fill_between(limits,limits-1, limits+1, alpha=0.2, color='gray')

  ax0.scatter(trues, preds, c='k')

  ax0.set_ylabel(r'RT$\bf{\cdot}$lnDC$\bf{_{50}}$ (kcal/mol)', fontweight='bold', fontsize=16)
  ax0.set_xlabel('Predicted (kcal/mol)', fontweight='bold', fontsize=16)
  #ax0.set_title(f'LOOCV: r = {r:.2f}, PC = {PC:.2f}')
  ax0.set_ylim([y_min-0.2,y_max+0.2])
  ax0.set_xlim([y_min-0.2,y_max+0.2])
  ax0.grid()

  weights = np.vstack(weights)
  mean_weights = np.mean(weights, axis=0)
  stderr_weights = np.std(weights, axis=0, ddof=1) / np.sqrt(len(weights))

  #ax1.bar(range(len(X_labels)), mean_weights, yerr=stderr_weights, color='maroon')
  #ax1.set_xticks(range(len(X_labels)))
  #ax1.set_xticklabels(X_labels, rotation=45, ha='right')
  #ax1.set_ylabel('Average weight')
  #ax1.set_title('Feature Weights (mean ± SE)')
  #ax1.grid()

  plt.tight_layout()
  plt.show()
  return fig

def rfe_loocv(data, X_labels, y_label, alpha=1.0, verbose=True):
  results = []

  data = data.copy()
  data = data[~np.isnan(data[y_label])]
  data.loc[:, y_label] = RT * np.log(data[y_label] * M_nM)

  remaining_labels = X_labels.copy()

  while len(remaining_labels) >= 1:
    X = data[remaining_labels].values
    y = data[y_label].values
    loo = LeaveOneOut()

    preds = []
    trues = []
    weights = []

    for train_idx, test_idx in loo.split(X):
      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]

      model = Ridge(alpha=alpha, solver='lsqr', fit_intercept=False)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      preds.append(y_pred[0])
      trues.append(y_test[0])
      weights.append(model.coef_)

    r, _, s, _ = get_pearson_spearman_nan(trues, preds)
    PC = percent_correct(trues, preds)
    weights = np.vstack(weights)
    mean_abs_weights = np.mean(np.abs(weights), axis=0)

    results.append({
      'features': remaining_labels.copy(),
      'r': r,
      'PC': PC,
      'weights': np.mean(weights, axis=0),
      'n': len(trues)
    })

    if verbose:
      if len(remaining_labels) > 5:
        print(f"n_features={len(remaining_labels)} | r={r:.3f}, PC={PC:.2f}")
      else:
        print(f"n_features={len(remaining_labels)} | r={r:.3f}, PC={PC:.2f}", remaining_labels[:5])

    # Stop if 1 feature left
    if len(remaining_labels) == 1:
      break

    # Eliminate least important feature
    worst_idx = np.argmin(mean_abs_weights)
    del remaining_labels[worst_idx]

  return results

def plot_rfe_results(rfe_results):
  n_feats = [len(r['features']) for r in rfe_results]
  r_vals = [r['r'] for r in rfe_results]
  PC_vals = [r['PC'] for r in rfe_results]

  fig, ax = plt.subplots(figsize=(8,5))
  ax.plot(n_feats, r_vals, 'o-k', label='Pearson r')
  ax.plot(n_feats, PC_vals, '^-', color='maroon', label='Percent Correct')
  ax.set_xlabel('Number of Features', fontweight='bold')
  ax.set_ylabel('Performance', fontweight='bold')
  ax.invert_xaxis()
  ax.legend()
  ax.grid()
  plt.tight_layout()
  plt.show()

# consider taking into account peformance like low RMSE or high r
def consensus_features(meta_rfe_results, n_keep=10, threshold=50):
  from collections import defaultdict
  # meta_rfe_results: dict mapping keys to {'features': [...], 'n': int}
  # weight_threshold: minimum total weight (n) a feature must accumulate to be included
  feature_weights = defaultdict(float)
  for result in meta_rfe_results.values():
    features = result['features']
    weight = result['n']
    for f in features:
      feature_weights[f] += weight  # accumulate weighted vote

  # Sort features by total weight (descending)
  # Take top n_keep features
  sorted_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
  consensus = [f for f, _ in sorted_features[:n_keep]]

  ## Apply threshold
  #consensus = [f for f, w in feature_weights.items() if w >= threshold]
  return consensus

def fill_repeats(series):
  counts = series.value_counts()
  repeats = counts[counts > args.Nsame].index
  return series.where(~series.isin(repeats), np.nan)

def filter_activity(df, threshold, my_filter):
  # my_filter is a list of lists,e.g. [['IC50','DC50'],['cell','complex'],...]
  # this will generate any label which has a combination of an entry from each list
  # Order of these drops is critical!!

  # 1: Drop rows not matching supplied filter, e.g. ['ternary','complex']
  activity_columns, _ = get_mini_labels(df)
  filtered_activity_cols = [name for name in activity_columns if all(any(lbl.lower() in name.lower() for lbl in sublist) for sublist in my_filter)]
  print(f'\nFilter by',my_filter,end='\n\n')
  df = df.dropna(subset=filtered_activity_cols, how='all')

  # 2: Drop columns which still present and multiple activities per PROTAC
  activity_columns, _ = get_mini_labels(df)
  drop_cols = [col for col in activity_columns if col not in filtered_activity_cols]
  df = df.drop(columns=drop_cols)

  # 3: Fill in values with more than args.Nsame identical values with nan
  activity_columns, _ = get_mini_labels(df)
  for (doi, target, ligase), group in df.groupby(['Article DOI', 'Target', 'Ligase']):
    df.loc[(df['Target'] == target) & (df['Ligase'] == ligase) & (df['Article DOI'] == doi), activity_columns] = group[activity_columns].apply(lambda col: fill_repeats(col))
  
  # 4: Drop Correlated activity columns
  activity_columns, _ = get_mini_labels(df)
  drop_cols = []
  for (doi, target, ligase), group in df.groupby(['Article DOI', 'Target', 'Ligase']):
    corr_df = group[activity_columns].dropna(axis=1, how='all').corr()
    drop_cols += findCorrelation(corr_df, threshold=threshold) 
  df = df.drop(columns=drop_cols)

  keep_y_labels,_ = get_mini_labels(df)
  return df, keep_y_labels

def drop_correlated_X(data, X_labels, threshold):
  corr_df = data[X_labels].corr()
  drop_cols = findCorrelation(corr_df, threshold, verbose=True)
  print(f'\nDropping (correlation above {threshold} to another):', drop_cols, end='\n\n')
  keep_X_labels = [col for col in X_labels if col not in drop_cols]
  return data.drop(columns=drop_cols), keep_X_labels

def shuffle_row(row):
  row = np.random.permutation(row.values)
  return row

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plot correlation of SILCS-PROTAC scores and Expt')
  parser.add_argument('-n', '--topn',           type=int,   default=20,                            help='Top N ppi complexes')
  parser.add_argument(      '--agg_log',                    default='aggregated_smc_log.pkl',      help='Aggregated SILCS-MC log file')
  parser.add_argument('-t', '--target',         type=str,   default=None,                          help='Name of target')
  parser.add_argument('-l', '--ligase',         type=str,   default=None,                          help='Name of ligase')
  parser.add_argument(      '--hide_act_frac',              action='store_true',                   help='Flag to hide %% Degraded and Dmax activities')
  parser.add_argument('-f', '--threshold',      type=float, default=0.8,                           help='Threshold for correlation to drop columns (both activity and X)')
  parser.add_argument('-c', '--consensus',      type=int,                                          help='Threshold for consensus to pick general features')
  parser.add_argument('-pf','--pairfile',       type=str,   default='pair_list_dc50.txt',   help='Text file containing list of ligase target pairs')
  parser.add_argument('-filter',                type=int,   default=5,                             help='Index of the filter type, e.g. -fhelp for options')
  parser.add_argument('-Nsame',                 type=int,   default=5,                             help='Number of identical entries in an activity column to replace with nan')
  parser.add_argument('-alpha',                 type=float, default=1,                             help='Strength of Ridge Regularization, smaller is less regularization, default: %(default)s')
  parser.add_argument('-fhelp',                             action='store_true',                   help='Print the filter options')
  parser.add_argument('-metricIndex',           type=int,   default=0,                             help='0:rPearson,1:rSpearman,2:RMSE; default: %(default)s')
  parser.add_argument('-metricTol',             type=float, default=1e-4,                          help='Tolerance for accepting improved model, default: %(default)s')
  parser.add_argument('-rfe',                               action='store_true',                   help='Perform Recursive Feature Elimination training; default: %(default)s')
  parser.add_argument('-q',                                 action='store_true',                   help='Quiet the plot')
  parser.add_argument('-min_data_limit',        type=int,   default=2,                             help='Before plot/fit, ensure there are at least this many data per subset')
  parser.add_argument('--randomize',                        action='store_true',                   help='Randomly shuffle X data prior to any LOO CV fitting')
  args = parser.parse_args()

  # initialize some stuff
  nonactivity_columns = ['index', 'Compound ID', 'Article DOI', 'Target', 'Ligase']
  activity_conc = ['DC50','IC50','EC50','GI50','Ki','Kd']
  activity_frac = ['Dmax','Percent','Activity'] # Activity used in WDR5 Western Blotting in one paper: True=100, False=0
  activity_labels = activity_conc + activity_frac

  sys_dict = { 'vhl-bcl2': 'VHL-BCL2', 'crbn-bclxl': 'CRBN-BCL-xL', 'vhl-bclxl': 'VHL-BCL-xL', 'crbn-brd4bd1': 'CRBN-BRD4BD1',
    'crbn-brd4bd2': 'CRBN-BRD4BD2', 'crbn-cdk9': 'CRBN-CDK9', 'mdm2-egfr-L858R-T790M': 'MDM2-EGFR', 'vhl-egfr-L858R-T790M': 'VHL-EGFR', 'vhl-fak1': 'VHL-FAK1',
    'crbn-hdac8': 'CRBN-HDAC8', 'crbn-smarca2': 'CRBN-SMARCA2', 'vhl-smarca2': 'VHL-SMARCA2', 'vhl-smarca4': 'VHL-SMARCA4', 'vhl-wdr5': 'VHL-WDR5' }
    #'mdm2-egfr-L858R-T790M': r'MDM2-EGFR$^{\text{L858R,T790M}}$', 'vhl-egfr-L858R-T790M': r'VHL-EGFR$^{\text{L858R,T790M}}$'

  ## Filters for activity data
  ## Elements must be a list of lists!
  ## [OR] AND [OR] AND [OR]
  #filter_dict = { 0:[activity_conc,['ternary','complex']], 1:[activity_conc,['cell','incuba']], 2:[activity_conc,['cell','incuba','complex','ternary']], 3:[['Ki','Kd']], 4:[['IC50'],['cell']], 5:[['DC50']], 6:[['EC50'],['cell']], 7:[['GI50'],['cell']], 8:[['DC50']], 9:[['cell']], 10:[activity_frac+activity_conc], 11:[['molm']],
  #              }
  #if args.fhelp:
  #  pprint(filter_dict)
  #  exit()

  ## Read in data, calculate scores
  ## Read in sar data for all protacs NOTE: not read from written merged_data.csv but gathered on the spot
  ##pair_file = 'protac-db/pair_list_master.txt'
  pair_file = args.pairfile
  #data, activity_df,_ = read_protacs_data(pair_file)

  ### SETUP
  ## Get LGFE arrays, protac properties, and activity (expt) data
  ## Read in the PPI logs
  ## Calculate various scores, like boltzmann averages, etc
  #ppips = get_ppips()
  #data = get_merged_lgfe_props_activities(data, args.agg_log)
  #data = calc_scores(data, ppips)
  ## ----------------------------------------------

  ### Clean X 
  ## Drop correlated X columns
  ## want to filter X on largest data possible, although the specific y fitted to may be smaller and thus subset of X may be correlated
  #all_X_labels = ['Ligase_LGFE_min', 'Target_LGFE_min','Linker_LGFE_min', 'Ref_Target_LGFE', 'Ref_Ligase_LGFE', 'CrippenClogP','NumRotatableBonds', 'LWCOMDist_LGFEmin', 'TWCOMDist_LGFEmin', 'Restr_RMSD_LGFEmin']
  ##init_X_labels = ['Ligase_LGFE_min','Target_LGFE_min','Linker_LGFE_min', 'Ref_Target_LGFE', 'Ref_Ligase_LGFE','Delta_Target','Delta_Ligase', 'NumRotatableBonds', 'CrippenClogP', 'Restr_RMSD_LGFEmin','Ligase_LGFE/COMDist','Target_LGFE/COMDist']
  ##init_X_labels = ['Linker_LGFE_min', 'Delta_Target','Delta_Ligase', 'NumRotatableBonds', 'CrippenClogP', 'Restr_RMSD_LGFEmin','Ligase_LGFE/COMDist','Target_LGFE/COMDist']

  #core = ['Ligase_LGFE_min','Target_LGFE_min','Linker_LGFE_min']
  #others = ['Delta_Target', 'Delta_Ligase','CrippenClogP','NumRotatableBonds','Restr_RMSD_LGFEmin', 'TWCOMDist_LGFEmin','LWCOMDist_LGFEmin']
  ##others = ['CrippenClogP','NumRotatableBonds', 'Delta_Target','Delta_Ligase']
  #init_X_labels = core + others

  #data, keep_X_labels  = drop_correlated_X(data, init_X_labels, args.threshold)

  #
  ##-------------------------------------------------------------

  ### Clean Y 
  ## Filter based on activity type, then drop correlated columns
  #my_filter = filter_dict[args.filter]
  #data, keep_y_labels = filter_activity(data, args.threshold, my_filter)

  ##print('Data used:')
  ##print(data[['Ligase','Target','Article DOI']].drop_duplicates())
  ##print()

  #if len(keep_y_labels) < 1: # keep_y_labels may not be correct?
  #  print(f"No more activity labels at this filter and threshold.")
  #  exit(1)
  ##-------------------------------------------------------------
  
  # read in data
  keep_y_labels = ['DC50 (nM, Degradation of BCL2 in 293T cells after 16 h treatment)', 'DC50 (nM, Degradation of BCL-xL in MOLT-4 cells after 16 h treatment)', 'DC50 (nM,Degradation of BCL-xL in MOLT-4 cells after 16 h treatment)', 'DC50 (nM, Degradation of BCL-xL in 293T cells after 16 h treatment)', 'DC50 (nM, Degradation of BRD4 BD1 assessed by EGFP/mCherry reporter assay, 5 hr)', 'DC50 (nM, Degradation of BRD4 BD2 assessed by EGFP/mCherry reporter assay, 5 hr)', 'DC50 (nM, Degradation of CDK9 in MOLM-13 cells)', 'DC50 (nM, Degradation of EGFR L858R/T790M in H1975 cells after 16h treatment)', 'DC50 (nM, Degradation of FAK in PC3 cells after 24 h treatment)', 'DC50 (nM, Degradation of FAK in A549 cells after 24 h treatment)', 'DC50 (nM, Degradation of HDAC8 in MDA-MB-231 cells after 24 h treatment)', 'DC50 (nM, Degradation of HDAC8 in A549 cells)', 'DC50 (nM, Degradation of HDAC8 in HCT-116 cells after 10 h treatment)', 'DC50 (nM, Degradation of SMARCA2 HiBiT in HT1080 cells after 6 h treatment)', 'DC50 (nM, Degradation of SMARCA2 in A549 cells)', 'DC50 (nM, Degradation of SMARCA4 in A549 cells)', 'DC50 (nM, Degradation of WDR5-HiBiT in MV4-11 (WDR5-HiBiT) cells after 24 h treatment)']
  my_filter = [['DC50']]
  core = ['Ligase_LGFE_min','Target_LGFE_min','Linker_LGFE_min']
  others = ['Delta_Target', 'Delta_Ligase','CrippenClogP','NumRotatableBonds','Restr_RMSD_LGFEmin', 'TWCOMDist_LGFEmin','LWCOMDist_LGFEmin']
  #others = ['CrippenClogP','NumRotatableBonds', 'Delta_Target','Delta_Ligase']
  keep_X_labels = core + others


  data = pd.read_pickle('protac_data.pkl')
  if args.randomize: data[keep_X_labels] = data[keep_X_labels].apply(shuffle_row, axis=0)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Single system mode: if target/ligase pair supplied
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if bool(args.target and args.ligase):
    if (args.ligase, args.target) not in read_pair_list(pair_file): 
      print(f"\n\tERROR: Pair ({args.target}, {args.ligase}) not found in {pair_file}.", end='\n\n')
      exit(1)

    single_system = data[(data['Ligase']==args.ligase)&(data['Target']==args.target)].copy().dropna(how='all',axis=1)
    keep_y_labels = [col for col in keep_y_labels if col in single_system.columns]

    #keep_X_labels = ['Ligase_LGFE_min', 'Target_LGFE_min','Linker_LGFE_min','CrippenClogP', 'NumRotatableBonds']
    #keep_X_labels = ['Ligase_LGFE_min','Target_LGFE_min','Linker_LGFE_min', 'Ref_Target_LGFE', 'Ref_Ligase_LGFE','Delta_Target','Delta_Ligase', 'NumRotatableBonds', 'CrippenClogP', 'Restr_RMSD_LGFEmin','Ligase_LGFE/COMDist','Target_LGFE/COMDist']

    if len(keep_y_labels) < 1:
      print(f"\n\tERROR: Pair ({args.target}, {args.ligase}) has no y data with this filter {my_filter} and threshold {args.threshold}.", end='\n\n')
      exit(1)
    print(single_system[keep_y_labels].columns)
    print(single_system[['Compound ID']+keep_X_labels+keep_y_labels])
    #single_system[['Compound ID']+keep_X_labels+keep_y_labels].to_csv(f'ppi-{args.ligase}-{args.target}/data.csv')

    # Now do the iteration
    activity_columns, mini_labels = get_mini_labels(single_system, my_filter)
    # Loop over available activities
    for act_ndx in range(len(activity_columns)):
      y_label = activity_columns[act_ndx]
      if sum(single_system[y_label].notna()) > 3:
        print(y_label)
        print(single_system[single_system[y_label].notna()]['Article DOI'].iloc[0])
        
        if args.rfe: ## RFE with LOOCV
          rfe_results = rfe_loocv(single_system, keep_X_labels, y_label, alpha=args.alpha, verbose=True)
          best_result = max(rfe_results, key=lambda d: d['r'] if d['r'] is not None else -np.inf)
          print(best_result['features'])
          plot_rfe_results(rfe_results)
          plot_loocv(single_system, best_result['features'], y_label, alpha=args.alpha)
        else: # leave-one-out CV
          plot_loocv(single_system, keep_X_labels, y_label, alpha=args.alpha)
  
    exit()
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif args.rfe:
    all_rfe_results = {}
    for (ligase, target) in read_pair_list(pair_file):
      #print(ligase, target)
      single_system = data[(data['Ligase']==ligase)&(data['Target']==target)].copy().dropna(how='all',axis=1)
      y_labels = [col for col in keep_y_labels if col in single_system.columns]
      #keep_X_labels = ['Ligase_LGFE_min','Target_LGFE_min','Linker_LGFE_min', 'Ref_Target_LGFE', 'Ref_Ligase_LGFE','Delta_Target','Delta_Ligase', 'NumRotatableBonds', 'CrippenClogP', 'Restr_RMSD_LGFEmin','Ligase_LGFE/COMDist','Target_LGFE/COMDist']
      if len(y_labels) < 1:
        print(f"\n\tERROR: Pair ({target}, {ligase}) has no y data with this filter {my_filter} and threshold {args.threshold}.", end='\n\n')
        continue
      #print(single_system[y_labels].columns)
      #print(single_system[['Compound ID']+keep_X_labels+y_labels])

      # Now do the iteration
      activity_columns, mini_labels = get_mini_labels(single_system, my_filter)
      # Loop over available activities
      for act_ndx in range(len(activity_columns)):
        y_label = activity_columns[act_ndx]
        if sum(single_system[y_label].notna()) > 3:
          rfe_results = rfe_loocv(single_system, keep_X_labels, y_label, alpha=args.alpha, verbose=False)
          best_result = max(rfe_results, key=lambda d: d['r'] if d['r'] is not None else -np.inf)
          best_features = best_result['features']
          all_rfe_results[(ligase, target, y_label)] = best_result
          #plot_rfe_results(rfe_results)

    # Perform RFE, and at each iteration (incrementally reduce N features), perform LOO-CV
    threshold_results=[]
    for n_keep in np.arange(len(keep_X_labels),0,-1) if not args.consensus else [args.consensus]:
      c_feat = consensus_features(all_rfe_results, n_keep=n_keep)
      if len(c_feat) < 1: 
        break

      all_loocv_preds = []
      for (ligase, target) in read_pair_list(pair_file):
        #print(ligase, target)
        single_system = data[(data['Ligase']==ligase)&(data['Target']==target)].copy().dropna(how='all',axis=1)
        y_labels = [col for col in keep_y_labels if col in single_system.columns]
        if len(y_labels) < 1:
          print(f"\n\tERROR: Pair ({target}, {ligase}) has no y data with this filter {my_filter} and threshold {args.threshold}.", end='\n\n')
          continue
        #print(single_system[y_labels].columns)
        #print(single_system[['Compound ID']+c_feat+y_labels])

        # Now do the iteration
        activity_columns, mini_labels = get_mini_labels(single_system, my_filter)
        # Loop over available activities
        for act_ndx in range(len(activity_columns)):
          y_label = activity_columns[act_ndx]
          if sum(single_system[y_label].notna()) > 3:
            results = run_loocv(single_system, c_feat, y_label, alpha=args.alpha)
            #print(results)
            results['pair'] = f"{ligase}-{target}"
            results['y_label'] = y_label
            all_loocv_preds.append(results)

      print()
      # take consensus features,
      # make plot and compute weighted r as well as overall r
      fig, (ax0,ax_leg) = plt.subplots(1,2, figsize=(8,4.35))
      ax_leg.axis('off')
      legend_text,legend_data=[],[]
      n_sum,r_weighted,rmse_weighted=0,0,0
      pool_true,pool_pred=[],[]
      for i,entry in enumerate(all_loocv_preds):
        m = 'o' if i < 10 else '^'

        label = f"{sys_dict[entry['pair']]} (r = {entry['r']:.2f})"
        # for only fak1 there are multiples
        if entry['y_label'] == 'DC50 (nM, Degradation of FAK in PC3 cells after 24 h treatment)':
          label = sys_dict[entry['pair']] + r'$^{(a)}$' + f'(r = {entry['r']:.2f})'
        if entry['y_label'] == 'DC50 (nM, Degradation of FAK in A549 cells after 24 h treatment)':
          label = sys_dict[entry['pair']] + r'$^{(b)}$' + f'(r = {entry['r']:.2f})'
 
        l=ax0.scatter(entry['y_pred'], entry['y_true'], alpha=0.8, label=label,color=color_list[i%10], marker=m)
        #plot_regression(entry['y_pred'], entry['y_true'], ax0, c=color_list[i%10])
        pool_true.extend(entry['y_true'])
        pool_pred.extend(entry['y_pred'])
        legend_text.append(label)
        legend_data.append(l)
        r_weighted += entry['r']*entry['n']
        rmse_weighted += entry['rmse']*entry['n']
        n_sum += entry['n']
      r_weighted /= n_sum
      rmse_weighted /= n_sum
      pool_r,_,pool_s,_=get_pearson_spearman_nan(pool_true,pool_pred)
      pool_rmse = rmse_nan(pool_true, pool_pred)
      print(f'N_feat: {len(c_feat)}: {c_feat}')
      print(f'r_weight = {r_weighted:.2f} RMSE = {rmse_weighted:.2f}')
      #print(f'pool_r = {pool_r:.2f} pool_rmse = {pool_rmse:.2f}')

      ax_leg.legend(legend_data,legend_text,fontsize=13, loc='center left')
      ax0.grid()
      y_min, y_max = min(pool_true+pool_pred)-1, max(pool_true+pool_pred)+1
      limits=np.arange(y_min, y_max+0.5,0.5)

      ax0.set_ylim([y_min,y_max])
      ax0.set_xlim([y_min,y_max])
      ax0.set_ylabel(r'RT$\bf{\cdot}$lnDC$\bf{_{50}}$ (kcal/mol)', fontweight='bold')
      ax0.set_xlabel('Predicted (kcal/mol)', fontweight='bold')

      # Diagonal x=y and +/- 1 kcal/mol
      ax0.plot(limits,limits,color='k', linestyle='--', linewidth=1.0)
      ax0.fill_between(limits,limits-1, limits+1, alpha=0.2, color='gray')

      plt.tight_layout()
      if not args.q and not args.randomize: 
        plt.savefig(f"figs/loocv-rfe/Nmetrics_{len(c_feat)}.svg")
        plt.show()
      plt.close()

      # these are r and rmse of general model
      agg_r, agg_rmse = analyze_loocv_results(all_loocv_preds, c_feat, data)

      threshold_results.append({
      'N_feat': len(c_feat),
      'wait_r': r_weighted,
      'wait_rmse': rmse_weighted,
      'wait_r_over_rmse': r_weighted/rmse_weighted,
      'pool_rmse':pool_rmse,
      'pool_r':pool_r,
      'pool_r_over_rmse': pool_r/pool_rmse,
      'agg_r_weight': agg_r,
      'agg_rmse_weight':agg_rmse
      })
      # end of threshold loop
     

    # plot optimize curves over all thresholds
    # Convert to arrays for plotting
    n_feats = [d['N_feat'] for d in threshold_results]
    wait_r = [d['wait_r'] for d in threshold_results]
    wait_rmse = [d['wait_rmse'] for d in threshold_results]
    agg_r = [d['agg_r_weight'] for d in threshold_results]
    agg_rmse = [d['agg_rmse_weight'] for d in threshold_results]

    # Plot
    #fig, ax1 = plt.subplots(1, 1)
    #ax1b = ax1.twinx() # twinx (new y)
    fig, (ax1,ax1b) = plt.subplots(2, 1, sharex=True)
    
    # Subplot 1: r/rmse and N_feat vs. threshold
    #ax1.plot(n_feats, pool_r, '-o', color=color_list[2], label='pool r')
    ax1.plot(n_feats, wait_r, '-o', color=color_list[0], label='LOO-CV')
    ax1.plot(n_feats, agg_r, '-o', color=color_list[1], label='General model')
    #ax1.set_xlabel('Number of Consensus Metrics', fontweight='bold')
    ax1.set_ylabel('r', color='k', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.axhline(0, zorder=0, color='k')
    ax1.grid()
    ax1.annotate('A',xy=(-0.1,0.9),xycoords='axes fraction', fontsize=14, fontweight='bold')
    ax1.set_ylim([-0.15,0.55])
    ax1.legend()

    #ax1b.plot(n_feats, pool_rmse, '-o', color=color_list[3], label='pool RMSE')
    ax1b.plot(n_feats, wait_rmse, '--o', color=color_list[0], label='RMSE')
    ax1b.plot(n_feats, agg_rmse, '--o', color=color_list[1], label='RMSE')
    ax1b.set_ylabel('RMSE (kcal/mol)', fontweight='bold')
    ax1b.set_xlabel('Number of Consensus Metrics', fontweight='bold')
    ax1b.grid()
    ax1b.annotate('B',xy=(-0.1,0.9),xycoords='axes fraction', fontsize=14, fontweight='bold')
    ax1b.set_ylim([-0.1,5.1])

    plt.tight_layout()
    if not args.randomize: plt.savefig(f'figs/overall_goodnessoffit.svg')
    plt.show()
