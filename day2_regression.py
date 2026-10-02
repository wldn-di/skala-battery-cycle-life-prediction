"""Small, reproducible Batch1 development and one-shot Batch2 regression test."""
from pathlib import Path
from project_paths import SCRATCH_PATH
import argparse
import hashlib
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
RESULTS, FIGURES, MODELS = ROOT/'results', ROOT/'figures', ROOT/'models'
SEED, TARGET_MAPE = 42, 9.1
CORE = ['log_var_delta_Q']
EXPANDED = CORE + ['mean_QD','std_QD','mean_IR','mean_Tavg','mean_Tmax',
                   'mean_chargetime','delta_QD_100_10']
CHARGING = ['C1','C2','switch_pct']
FEATURE_SETS = {'A_Core':CORE,'B_Expanded':EXPANDED,'C_Charging_ablation':EXPANDED+CHARGING}
FAMILIES = ['Ridge','RandomForest','GradientBoosting']

def now(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
def metrics(y,pred):
    assert np.isfinite(pred).all() and (pred>0).all()
    return {'MAPE_percent':100*mean_absolute_percentage_error(y,pred),
            'RMSE_cycles':float(np.sqrt(mean_squared_error(y,pred))), 'R2':float(r2_score(y,pred))}

def model_and_grid(family):
    if family == 'Ridge':
        estimator = Ridge()
        grid = {'regressor__model__alpha':[.1,1.,10.,100.]}
    elif family == 'RandomForest':
        estimator = RandomForestRegressor(n_estimators=150,random_state=SEED,n_jobs=1)
        grid = {'regressor__model__max_depth':[3,None],
                'regressor__model__min_samples_leaf':[2,4]}
    else:
        estimator = GradientBoostingRegressor(n_estimators=150,min_samples_leaf=3,random_state=SEED)
        grid = {'regressor__model__max_depth':[1,2],
                'regressor__model__learning_rate':[.03,.1]}
    steps = [('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if family == 'Ridge': steps.append(('scaler',StandardScaler()))
    steps.append(('model',estimator))
    # Fixed, reversible target transform; scoring always uses original cycle units.
    model = TransformedTargetRegressor(regressor=Pipeline(steps),func=np.log,inverse_func=np.exp)
    return model,grid

def search(model,grid,X,y):
    cv = KFold(n_splits=3,shuffle=True,random_state=SEED)
    result = GridSearchCV(model,grid,scoring='neg_mean_absolute_percentage_error',
                          cv=cv,n_jobs=1,refit=True,error_score='raise')
    return result.fit(X,y)

def development():
    for path in [RESULTS,FIGURES,MODELS]: path.mkdir(exist_ok=True)
    if (RESULTS/'batch2_test_receipt.json').exists():
        raise RuntimeError('Test already started: development locked.')
    audit = json.loads((RESULTS/'day1_audit.json').read_text())
    assert audit['passed'] and audit['day1_outputs_unchanged']
    # Development only reads the Batch1 subset for scoring and selection.
    all_rows = pd.read_csv(RESULTS/'feature_table.csv')
    b1 = all_rows[(all_rows.batch=='Batch1') & all_rows.target_available].copy().reset_index(drop=True)
    assert len(b1)==46 and b1.cell_id.is_unique
    train_idx,valid_idx = train_test_split(np.arange(len(b1)),test_size=.25,random_state=SEED)
    train,valid = b1.iloc[train_idx].copy(),b1.iloc[valid_idx].copy()
    assignments = pd.concat([train.assign(split='Train'),valid.assign(split='HoldoutValid')])
    assignments[['batch','cell_id','split','cycle_life']].to_csv(RESULTS/'batch1_split.csv',index=False)
    assert set(train.cell_id).isdisjoint(valid.cell_id)
    outer_splits = list(RepeatedKFold(n_splits=5,n_repeats=2,random_state=SEED).split(train))
    plan = {'created_at':now(),'task':'Regression','prediction_cycle':100,'summary_window':[2,100],
        'random_state':SEED,'train_n':len(train),'valid_n':len(valid),'train_ids':train.cell_id.tolist(),
        'valid_ids':valid.cell_id.tolist(),'outer_cv':'RepeatedKFold(5 splits, 2 repeats)',
        'inner_tuning_cv':'KFold(3 splits, shuffled seed42)', 'target_transform':'natural log, inverse exp',
        'official_metric':'MAPE on original cycle units, percent', 'feature_sets':FEATURE_SETS,
        'model_families':FAMILIES,'test_batch':'Batch2','no_model_evaluation_batch':'Batch3',
        'selection_rule':'Minimum Valid MAPE. Within 0.5 percentage points, prefer fewer features/simpler family '
         'only if nested CV mean is within 1 pp and CV SD within 2 pp of minimum-Valid candidate.',
        'final_refit':'Train34 only; do not refit on HoldoutValid so Valid and Test assess the same fitted model',
        'test_limitations':'Batch2 was inspected in DAY1; external one-shot model evaluation, not an untouched blind dataset',
        'feature_table_sha256':digest(RESULTS/'feature_table.csv'), 'sklearn_version':sklearn.__version__}
    write_json(RESULTS/'development_plan.json',plan)
    rows,fold_records,oof_records,valid_records = [],[],[],[]
    candidates = {}
    for set_name,features in FEATURE_SETS.items():
        for family in FAMILIES:
            name = f'{set_name}__{family}'
            model,grid = model_and_grid(family)
            cv_metrics = []
            for fold,(fit_idx,eval_idx) in enumerate(outer_splits):
                fit = train.iloc[fit_idx];evaluation = train.iloc[eval_idx]
                tuned = search(clone(model),grid,fit[features],fit.cycle_life)
                pred = tuned.predict(evaluation[features])
                score = metrics(evaluation.cycle_life,pred)
                cv_metrics.append(score)
                fold_records.append({'candidate':name,'outer_fold':fold+1,'fit_n':len(fit),
                    'evaluate_n':len(evaluation),**score,'best_params':json.dumps(tuned.best_params_)})
                for cid,target,prediction in zip(evaluation.cell_id,evaluation.cycle_life,pred):
                    oof_records.append({'candidate':name,'outer_fold':fold+1,'cell_id':cid,
                                        'actual':target,'predicted':prediction})
                if fold in [0,4,9]: print(f'{name}: nested CV fold {fold+1}/10',flush=True)
            tuned = search(clone(model),grid,train[features],train.cycle_life)
            pred_valid = tuned.predict(valid[features])
            valid_metrics = metrics(valid.cycle_life,pred_valid)
            scores = pd.DataFrame(cv_metrics)
            row = {'candidate':name,'feature_set':set_name,'model':family,'n_features':len(features),
                'Train_CV_MAPE_percent':scores.MAPE_percent.mean(),
                'Train_CV_MAPE_SD_pp':scores.MAPE_percent.std(ddof=1),
                'Train_CV_RMSE_cycles':scores.RMSE_cycles.mean(),'Train_CV_R2':scores.R2.mean(),
                'Valid_MAPE_percent':valid_metrics['MAPE_percent'],
                'Valid_RMSE_cycles':valid_metrics['RMSE_cycles'],'Valid_R2':valid_metrics['R2'],
                'Train_Valid_Gap_pp':valid_metrics['MAPE_percent']-scores.MAPE_percent.mean(),
                'best_params':json.dumps(tuned.best_params_),'feature_names':json.dumps(features)}
            rows.append(row);candidates[name] = tuned.best_estimator_
            for cid,target,prediction in zip(valid.cell_id,valid.cycle_life,pred_valid):
                valid_records.append({'candidate':name,'cell_id':cid,'actual':target,'predicted':prediction,
                                      'APE_percent':100*abs(target-prediction)/target})
            print(f'{name}: CV={row["Train_CV_MAPE_percent"]:.3f}%, Valid={row["Valid_MAPE_percent"]:.3f}%',flush=True)
            pd.DataFrame(rows).to_csv(RESULTS/'development_comparison.csv',index=False)
    comparison = pd.DataFrame(rows)
    pd.DataFrame(fold_records).to_csv(RESULTS/'nested_cv_folds.csv',index=False)
    pd.DataFrame(oof_records).to_csv(RESULTS/'train_cv_predictions.csv',index=False)
    pd.DataFrame(valid_records).to_csv(RESULTS/'holdout_predictions_all_candidates.csv',index=False)
    best = comparison.loc[comparison.Valid_MAPE_percent.idxmin()]
    near = comparison[(comparison.Valid_MAPE_percent <= best.Valid_MAPE_percent+.5)
        & (comparison.Train_CV_MAPE_percent <= best.Train_CV_MAPE_percent+1.)
        & (comparison.Train_CV_MAPE_SD_pp <= best.Train_CV_MAPE_SD_pp+2.)].copy()
    near['complexity'] = near.model.map({'Ridge':0,'RandomForest':1,'GradientBoosting':2})
    selected = near.sort_values(['n_features','complexity','Valid_MAPE_percent']).iloc[0]
    selected_name = selected.candidate
    selected_features = FEATURE_SETS[selected.feature_set]
    model_path = MODELS/'final_regression.joblib'
    joblib.dump(candidates[selected_name],model_path)
    selected_valid = pd.DataFrame(valid_records).query('candidate == @selected_name').copy()
    selected_valid.to_csv(RESULTS/'valid_predictions.csv',index=False)
    freeze = {'frozen_at':now(),'candidate':selected_name,'model':selected.model,
        'feature_set':selected.feature_set,'features':selected_features,'best_params':json.loads(selected.best_params),
        'minimum_valid_candidate':best.candidate,'selection_rule':plan['selection_rule'],
        'selection_reason':f'Chosen before Batch2 evaluation: Valid {selected.Valid_MAPE_percent:.4f}%, '
          f'nested CV {selected.Train_CV_MAPE_percent:.4f}% ± {selected.Train_CV_MAPE_SD_pp:.4f} pp; '
          f'{len(selected_features)} features. Near-tie simplicity rule applied within fixed tolerances.',
        'train_ids':train.cell_id.tolist(),'valid_ids':valid.cell_id.tolist(),'training_n':len(train),
        'model_sha256':digest(model_path),'feature_table_sha256':digest(RESULTS/'feature_table.csv'),
        'development_comparison_sha256':digest(RESULTS/'development_comparison.csv'),
        'script_sha256':digest(Path(__file__)),'target_transform':'log(y), exp inverse',
        'test_batch':'Batch2','Batch3_evaluated':False,'paper_reference_target_percent':TARGET_MAPE}
    write_json(RESULTS/'model_freeze.json',freeze)
    comparison['selected'] = comparison.candidate==selected_name
    comparison.to_csv(RESULTS/'development_comparison.csv',index=False)
    # Hash final comparison after writing selection annotations.
    freeze['development_comparison_sha256'] = digest(RESULTS/'development_comparison.csv')
    write_json(RESULTS/'model_freeze.json',freeze)
    schema = {'model_features':selected_features,'candidate_features':EXPANDED,'charging_ablation':CHARGING,
        'not_model_inputs':['batch','cell_id','source_cell_index','charging_policy','cycle_life','target_available',
                            'feature_min_cycle','feature_max_cycle'],
        'excluded_future_columns':['n_recorded_cycles','last_recorded_cycle','first_valid_QD','last_valid_QD','endpoint_change_pct'],
        'summary_feature_window':'original Cycle2–100; 99 cycles; nonpositive/nonfinite values missing',
        'delta_feature_definition':'log10(population variance of Qdlin100-Qdlin10), ddof=0',
        'delta_QD_definition':'summary QDischarge Cycle100 minus Cycle10',
        'std_QD_ddof':1,'target':'provided cycle_life','fit_scope':'Batch1 Train34 and training-only CV folds'}
    write_json(RESULTS/'feature_schema.json',schema)
    fig,ax = plt.subplots(figsize=(13,6))
    order = comparison.sort_values('Valid_MAPE_percent')
    x = np.arange(len(order))
    ax.bar(x-.18,order.Train_CV_MAPE_percent,width=.36,yerr=order.Train_CV_MAPE_SD_pp,
           label='Train nested CV mean ± SD',color='#5a89b5',capsize=3)
    ax.bar(x+.18,order.Valid_MAPE_percent,width=.36,label='Batch1 Holdout Valid',color='#e6a04b')
    ax.axhline(TARGET_MAPE,color='red',ls='--',label='9.1% reference (different study protocol)')
    ax.set_xticks(x,order.candidate,rotation=45,ha='right',fontsize=8)
    ax.set(ylabel='MAPE (%)',title='Batch1 only: feature sets and model families')
    ax.legend();fig.tight_layout();fig.savefig(FIGURES/'day2_development_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    ax.scatter(selected_valid.actual,selected_valid.predicted,color='#377eb8')
    lo=min(selected_valid.actual.min(),selected_valid.predicted.min());hi=max(selected_valid.actual.max(),selected_valid.predicted.max())
    ax.plot([lo,hi],[lo,hi],ls='--',color='black')
    ax.set(xlabel='Actual provided life (cycles)',ylabel='Predicted life (cycles)',
           title=f'Batch1 Holdout Valid: {selected_name}\nMAPE={selected.Valid_MAPE_percent:.3f}% (n=12)')
    fig.tight_layout();fig.savefig(FIGURES/'day2_valid_predictions.png',dpi=150);plt.close(fig)
    decisions = [
        {'step':1,'decision':'Retain DAY1 definitions','reason':'Raw recomputation matches saved output; common 99-cycle window disclosed.'},
        {'step':2,'decision':'Regression / log target / MAPE','reason':'Positive lifetime and relative-error objective; reversible fixed transform, original-unit scoring.'},
        {'step':3,'decision':'Fixed Batch1 Train34/Valid12 split, seed42','reason':'Independent Holdout; no split shopping or Batch2/3-driven choices.'},
        {'step':4,'decision':'Three families and three predefined feature sets','reason':'Core interpretability, Expanded comparison, charging-only ablation; no individual feature search on external batches.'},
        {'step':5,'decision':'Nested Train CV','reason':'Hyperparameters chosen within inner folds; outer scores are not fitted training error.'},
        {'step':6,'decision':selected_name,'reason':freeze['selection_reason']},
        {'step':7,'decision':'Freeze fitted Train34 model before Test','reason':'Same fitted model for Holdout and external Test; no fitting on Valid or Batch2.'},
    ]
    pd.DataFrame(decisions).to_csv(RESULTS/'decision_log.csv',index=False)
    print('\nMODEL FROZEN:',selected_name,freeze['selection_reason'],flush=True)
    return freeze

def verify_frozen_inputs(freeze):
    for path,key in [(MODELS/'final_regression.joblib','model_sha256'),
                     (RESULTS/'feature_table.csv','feature_table_sha256'),
                     (RESULTS/'development_comparison.csv','development_comparison_sha256')]:
        if digest(path)!=freeze[key]:
            raise RuntimeError(f'Frozen artifact changed: {path.name}')
    receipt_path=RESULTS/'batch2_test_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['freeze_sha256']!=digest(RESULTS/'model_freeze.json'):
            raise RuntimeError('Freeze no longer matches the Test receipt.')

def final_test():
    freeze = json.loads((RESULTS/'model_freeze.json').read_text())
    receipt_path = RESULTS/'batch2_test_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') == 'completed':
            verify_frozen_inputs(freeze)
            print('Batch2 already evaluated once. Returning saved metrics without prediction or refit.')
            return pd.read_csv(RESULTS/'model_performance.csv')
        raise RuntimeError('An incomplete one-shot test receipt exists; do not automatically retest.')
    verify_frozen_inputs(freeze)
    if digest(Path(__file__)) != freeze['script_sha256']:
        raise RuntimeError('Frozen training script differs; do not start a new official Test.')
    comparison = pd.read_csv(RESULTS/'development_comparison.csv')
    selected = comparison[comparison.candidate==freeze['candidate']].iloc[0]
    table = pd.read_csv(RESULTS/'feature_table.csv')
    test = table[(table.batch=='Batch2') & table.target_available].copy()
    assert len(test)==39 and test.cell_id.is_unique
    assert set(test.cell_id).isdisjoint(freeze['train_ids']+freeze['valid_ids'])
    receipt = {'started_at':now(),'status':'started','test_batch':'Batch2','test_n':len(test),
        'prediction_calls':0,'model_sha256':freeze['model_sha256'],'freeze_sha256':digest(RESULTS/'model_freeze.json')}
    with receipt_path.open('x') as f: json.dump(receipt,f,indent=2)
    model = joblib.load(MODELS/'final_regression.joblib')
    # The only Batch2 model prediction call in the project.
    predictions = model.predict(test[freeze['features']])
    score = metrics(test.cycle_life,predictions)
    result = test[['batch','cell_id','source_cell_index','charging_policy','cycle_life']].copy()
    result['predicted_cycle_life']=predictions
    result['signed_error_cycles']=predictions-result.cycle_life
    result['absolute_error_cycles']=abs(result.signed_error_cycles)
    result['APE_percent']=100*result.absolute_error_cycles/result.cycle_life
    result.to_csv(RESULTS/'test_predictions.csv',index=False)
    performance = pd.DataFrame([{'model':freeze['model'],'feature_set':freeze['feature_set'],
        'features':json.dumps(freeze['features']),'Train_n':freeze['training_n'],'Valid_n':12,'Test_n':39,
        'Train_Batch1_CV_MAPE_percent':selected.Train_CV_MAPE_percent,
        'Train_CV_MAPE_SD_pp':selected.Train_CV_MAPE_SD_pp,'Valid_Batch1_MAPE_percent':selected.Valid_MAPE_percent,
        'Test_Batch2_MAPE_percent':score['MAPE_percent'],
        'Train_Valid_Gap_pp':selected.Valid_MAPE_percent-selected.Train_CV_MAPE_percent,
        'Valid_Test_Gap_pp':score['MAPE_percent']-selected.Valid_MAPE_percent,
        'Target_Test_Gap_pp':score['MAPE_percent']-TARGET_MAPE,'Paper_Target_MAPE_percent':TARGET_MAPE,
        'Train_CV_RMSE_cycles':selected.Train_CV_RMSE_cycles,'Valid_RMSE_cycles':selected.Valid_RMSE_cycles,
        'Test_RMSE_cycles':score['RMSE_cycles'],'Train_CV_R2':selected.Train_CV_R2,
        'Valid_R2':selected.Valid_R2,'Test_R2':score['R2']}])
    performance.to_csv(RESULTS/'model_performance.csv',index=False)
    receipt.update({'status':'completed','completed_at':now(),'prediction_calls':1,'metrics':score})
    write_json(receipt_path,receipt)
    print('\nFINAL BATCH2 TEST:',score,flush=True)
    return performance

def reporting():
    freeze=json.loads((RESULTS/'model_freeze.json').read_text())
    receipt=json.loads((RESULTS/'batch2_test_receipt.json').read_text())
    if receipt['status']!='completed' or receipt['prediction_calls']!=1:
        raise RuntimeError('Report requires a completed one-shot Test.')
    verify_frozen_inputs(freeze)
    report_results=ROOT/'reproduced'/'report'/'results'
    report_figures=ROOT/'reproduced'/'report'/'figures'
    report_results.mkdir(parents=True,exist_ok=True)
    report_figures.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(RESULTS/'feature_table.csv')
    train=table[table.cell_id.isin(freeze['train_ids'])]
    valid=pd.read_csv(RESULTS/'valid_predictions.csv')
    test=pd.read_csv(RESULTS/'test_predictions.csv')
    performance=pd.read_csv(RESULTS/'model_performance.csv')
    # Error analysis uses saved predictions. No second Batch2 predict call.
    error=test.merge(table[['cell_id']+freeze['features']],on='cell_id',validate='one_to_one')
    missing_names=[];outside_names=[]
    for _,row in error.iterrows():
        missing=[];outside=[]
        for feature in freeze['features']:
            x=row[feature]
            if pd.isna(x): missing.append(feature)
            elif x<train[feature].min() or x>train[feature].max(): outside.append(feature)
        missing_names.append(','.join(missing));outside_names.append(','.join(outside))
    error['missing_model_features']=missing_names
    error['outside_train_range_features']=outside_names
    error['policy_seen_in_train']=error.charging_policy.isin(train.charging_policy)
    error.sort_values('APE_percent',ascending=False).to_csv(report_results/'cell_error_analysis.csv',index=False)
    error.sort_values('APE_percent',ascending=False).head(10).to_csv(report_results/'worst_cells.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,part,title,pred_column in [(axes[0],valid,'Batch1 Holdout Valid','predicted'),
                                     (axes[1],test,'Batch2 Final Test','predicted_cycle_life')]:
        actual=part.actual if 'actual' in part else part.cycle_life
        pred=part[pred_column]
        ax.scatter(actual,pred,alpha=.75)
        lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max())
        ax.plot([lo,hi],[lo,hi],color='black',ls='--')
        ax.set(xlabel='Actual provided life (cycles)',ylabel='Predicted life (cycles)',title=title)
    fig.suptitle(f'Frozen model: {freeze["candidate"]}');fig.tight_layout()
    fig.savefig(report_figures/'day2_valid_test_predictions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].scatter(test.cycle_life,test.signed_error_cycles,alpha=.75)
    axes[0].axhline(0,color='black',ls='--');axes[0].set(xlabel='Actual life (cycles)',
        ylabel='Predicted - actual (cycles)',title='Batch2 signed errors')
    worst=error.nlargest(10,'APE_percent').sort_values('APE_percent')
    axes[1].barh(worst.cell_id,worst.APE_percent,color='#e6a04b')
    axes[1].set(xlabel='Absolute percentage error (%)',title='Worst Batch2 cells (post-test analysis)')
    fig.tight_layout();fig.savefig(report_figures/'day2_error_analysis.png',dpi=150);plt.close(fig)
    # Reuse the frozen explanation; reporting never unpickles or evaluates a model.
    explanation=pd.read_csv(RESULTS/'model_explanation.csv')
    explanation.to_csv(report_results/'model_explanation.csv',index=False)
    baseline=performance.iloc[0]
    audit=json.loads((RESULTS/'day1_audit.json').read_text())
    historical_sources=[]
    for name,dig in audit['source_hashes'].items():
        source=SCRATCH_PATH if name=='scratch_original' else ROOT/name
        status=('MISSING' if not source.is_file() else
                'MATCH' if digest(source)==dig else 'CHANGED_SINCE_HISTORICAL_AUDIT')
        historical_sources.append({'source':name,'status':status})
    pd.DataFrame(historical_sources).to_csv(report_results/'historical_source_status.csv',index=False)
    consistency=[
        {'item':'Frozen model and feature inputs unchanged','passed':True,'evidence':'Current SHA256 matches freeze; historical source status reported separately'},
        {'item':'Early 100-cycle boundary','passed':True,'evidence':'Raw audit confirms Cycle2–100 + Q10/Q100 only'},
        {'item':'One row per recorded battery cell','passed':table.cell_id.is_unique,'evidence':f'{len(table)} raw records, 129 target-available'},
        {'item':'Batch1-only model development','passed':all(cid.startswith('Batch1_') for cid in freeze['train_ids']+freeze['valid_ids']),
         'evidence':'Fixed split; all CV/tuning/selection confined to Batch1'},
        {'item':'Preprocessing fit inside training folds','passed':True,'evidence':'Imputer/scaler in Pipeline, inner tuning inside outer CV'},
        {'item':'Train performance is CV mean','passed':True,'evidence':'10 nested outer-fold metrics, no fitted training performance'},
        {'item':'Holdout not used for fitting','passed':set(freeze['train_ids']).isdisjoint(freeze['valid_ids']),
         'evidence':'Artifact fit on Train34; Holdout12 used for final candidate selection only'},
        {'item':'Core delta feature retained','passed':'log_var_delta_Q' in freeze['features'],'evidence':freeze['feature_set']},
        {'item':'Batch2 once after freeze','passed':receipt['prediction_calls']==1,'evidence':receipt['freeze_sha256']},
        {'item':'Batch3 model evaluation absent','passed':not freeze['Batch3_evaluated'],'evidence':'No Batch3 prediction/metrics code path'},
        {'item':'Metric and gap direction','passed':True,'evidence':'MAPE percent; Valid-CV, Test-Valid, Test-9.1; gaps in percentage points'},
    ]
    assert all(bool(row['passed']) for row in consistency)
    pd.DataFrame(consistency).to_csv(report_results/'consistency_audit.csv',index=False)
    pd.DataFrame([{'gap':'Train–Valid','formula':'Valid MAPE - Train CV mean MAPE','unit':'percentage points','positive_means':'Valid worse than Train CV'},
        {'gap':'Valid–Test','formula':'Test MAPE - Valid MAPE','unit':'percentage points','positive_means':'External Test worse than Valid'},
        {'gap':'Target–Test','formula':'Test MAPE - 9.1','unit':'percentage points','positive_means':'Test above 9.1% reference'}]).to_csv(report_results/'gap_definitions.csv',index=False)
    decision=pd.read_csv(RESULTS/'decision_log.csv')
    if not (decision.step==8).any():
        decision=pd.concat([decision,pd.DataFrame([{'step':8,'decision':'One-shot external test complete',
            'reason':f'Batch2 MAPE={baseline.Test_Batch2_MAPE_percent:.4f}%; no subsequent tuning or retest.'}])],ignore_index=True)
    decision.to_csv(report_results/'decision_log.csv',index=False)
    worst_row=error.loc[error.APE_percent.idxmax()]
    readme=f'''# SKALA DS Mini Project — Cycle Life Regression

기존 DAY1 결과와 동일한 작업 폴더·실행 환경을 이어 사용했다. 원본 Scratch·MAT·DAY1 결과는 변경하지 않았다.

## DAY1 검증 및 설계

Batch1 Cycle1은 46개 모두 summary 0·Qdlin empty다. Scratch는 이 0을 포함하지만 DAY1은
Batch 공통 구간을 Cycle2–100으로 정했다. 이는 초기 100사이클 안의 99개 측정 사이클이며,
Batch2·3 Cycle1의 결함을 뜻하지 않는다. 이 정의를 유지한다. raw 재계산 오차는 최대 약 1.42e-14였다.
ΔQ=Q100−Q10, log_var_delta_Q=log10(var(ΔQ)), ddof=0, 공통 1,000점 전압축이다.
수명 결측 10개는 master table에 유지하고 지도학습에서만 제외한다.

스토리: Batch별 제공 수명 분포 차이 → Batch Shift 가능성 → Batch1 개발/Batch2 외부 평가.
ΔQ 로그 분산은 사용자 확정 Core Feature다. 기타 Feature와 충전 ablation의 채택은 Batch1 결과로만 결정했다.
Batch2·3의 DAY1 상관은 추가 Feature 선택이나 모델 튜닝에 사용하지 않았다.

## 데이터 및 분할

Feature Table: 139행, 유효 target 129행(B1 46/B2 39/B3 44). cell ID는 파일의 기록 ID다.
물리 셀 중복·연속 기록의 확정 식별이나 논문 정제 레이블 재구성은 수행하지 않았다.
Batch1 Train 34 / Holdout Valid 12, random_state=42. Batch2 Test 39. Batch3 모델 평가 없음.
Train 성능은 반복 5-fold 2회(10 outer folds)의 nested CV 평균이다. 각 outer Train의
3-fold inner CV에서만 튜닝했다. MAPE를 원래 사이클 단위에서 계산했다.
Holdout은 모델·Feature 조합 선택에 사용했으므로 최종 성능의 독립 평가가 아니다.

## 비교 및 최종 선택

Ridge / RandomForest / GradientBoosting, A Core(1) / B Expanded(8) / C charging ablation(11).
후보 모두 log target 후 exp로 복원해 양의 수명을 예측한다. 공식 MAPE는 복원한 예측으로 계산한다.
Imputer는 Train fold의 중앙값, Ridge scaler도 Train fold에서만 fit한다. 원본 8개 Feature에
미래 입력은 없다. 전체 수명곡선·최종 사이클·기록 길이·정책 평균 수명은 입력에서 제외했다.

최종: **{freeze['candidate']}**, Feature: {', '.join(freeze['features'])}.
선택 근거: {freeze['selection_reason']}
최소 Valid MAPE 우선이며 0.5%p 이내는 CV 평균·SD 허용범위 안에서 단순한 구성을 우선했다.
최종 모델은 Train34에만 fit해 Valid와 Test가 같은 fitted model을 평가한다. 전체 Batch1 재학습은 하지 않았다.

## 실제 성능

| Metric | Train: Batch1 nested CV 평균 | Valid: Batch1 Holdout | Test: Batch2 |
|---|---:|---:|---:|
| MAPE (%) | {baseline.Train_Batch1_CV_MAPE_percent:.4f} | {baseline.Valid_Batch1_MAPE_percent:.4f} | {baseline.Test_Batch2_MAPE_percent:.4f} |
| RMSE (cycles) | {baseline.Train_CV_RMSE_cycles:.4f} | {baseline.Valid_RMSE_cycles:.4f} | {baseline.Test_RMSE_cycles:.4f} |
| R² | {baseline.Train_CV_R2:.4f} | {baseline.Valid_R2:.4f} | {baseline.Test_R2:.4f} |

Train CV MAPE SD: {baseline.Train_CV_MAPE_SD_pp:.4f}%p. 이는 fold 간 산포이며 신뢰구간이 아니다.
Train–Valid Gap = Valid − Train CV = {baseline.Train_Valid_Gap_pp:+.4f}%p.
Valid–Test Gap = Test − Valid = {baseline.Valid_Test_Gap_pp:+.4f}%p.
Target–Test Gap = Test − 9.1 = {baseline.Target_Test_Gap_pp:+.4f}%p.
양수는 뒤의 평가 오차가 더 크다는 뜻이다. 9.1%는 사용자 지정 논문 참고 목표이며,
raw 파일 구성·정제·분할이 같다고 검증하지 않았으므로 논문 성능의 직접 재현·우열로 해석하지 않는다.

## Error Analysis 및 ESS/PdM 해석

최대 APE 셀: {worst_row.cell_id}, 실제 {worst_row.cycle_life:.0f}, 예측 {worst_row.predicted_cycle_life:.2f},
APE {worst_row.APE_percent:.2f}%, signed error {worst_row.signed_error_cycles:+.2f} cycles.
학습 범위 밖 Feature: {worst_row.outside_train_range_features or '없음'}.
이는 관측 사실이며 오류의 인과적 원인이라고 확정하지 않는다. 상세 기록은 worst_cells.csv와 cell_error_analysis.csv다.

ESS/PdM에서 수명 과대예측은 점검·교체 시기를 늦추는 위험, 과소예측은 조기 교체·추가 점검 비용으로
이어질 수 있다. 따라서 평균 MAPE와 함께 signed error 및 큰 오차 셀을 검토해야 한다.
이 결과는 실험 셀의 cycle_life 예측이며 현장 ESS 시스템의 잔여수명(RUL), calendar aging,
운영 환경·시스템 고장까지 검증한 결과는 아니다. 운영 임계값이나 안전 마진은 여기서 확정하지 않는다.

## 평가 무결성과 한계

Batch2는 이미 DAY1에서 확인했으므로 완전한 untouched blind test가 아니다. 모델을 freeze한 후
predict 한 번만 실행했다. 이후 Error Analysis는 저장된 예측을 재사용하며 재튜닝·재평가하지 않았다.
Batch1 표본이 46개로 작고 Holdout도 12개이므로 선택 및 성능의 불확실성을 함께 고려해야 한다.
동일 Train에서 candidate를 선택한 뒤 그 candidate의 CV 값을 보고했으므로 모델 가족·Feature Set 선택
단계 전체의 선택 편향까지 nested CV가 제거한 것은 아니다. 독립 외부 성능은 Batch2 표에 한 번 보고했다.

## 파일 및 재실행

- 30-ESSHealth-DAY1.ipynb / .html: 기존 EDA (그대로 보존)
- 30-ESSHealth-DAY2.ipynb / .html: 실행 및 저장된 성능을 설명하는 DAY2 notebook
- audit_day1.py: raw와 기존 Feature 검증, DAY1 결과를 변경하지 않음
- day2_regression.py: develop / final-test / report 단계
- results/feature_table.csv: master 후보 Feature·target·metadata
- results/development_comparison.csv / nested_cv_folds.csv: Batch1 비교·CV
- results/model_freeze.json / batch2_test_receipt.json: 확정 시각·해시·one-shot test 기록
- results/model_performance.csv: 공식 성능 및 Gap
- results/decision_log.csv / consistency_audit.csv: 의사결정·일관성 검사
- models/final_regression.joblib: 전처리·target 역변환 포함 fitted model
- figures/day2_*.png: 개발·검증·외부 평가·오차 그림

기존 분석 실행 환경: Python 3.12. 분석 단계에서는 새 환경을 생성하지 않았다.
기존 환경에 scikit-learn·joblib만 추가했고 LightGBM/XGBoost는 설치하지 않았다.
이미 Test가 완료된 상태에서 develop은 차단되며 final-test는 저장된 표만 반환한다.
report는 저장된 예측으로 출력만 재생성한다. 따라서 평가를 반복하면서 성능을 맞추지 않는다.
'''
    (ROOT/'reproduced'/'report'/'README.md').write_text(readme)
    print('Reports written to reproduced/report; submission files preserved; saved predictions reused.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['develop','final-test','report'],required=True)
    stage=parser.parse_args().stage
    result={'develop':development,'final-test':final_test,'report':reporting}[stage]()
    if isinstance(result,pd.DataFrame):
        print(result.to_string(index=False))
