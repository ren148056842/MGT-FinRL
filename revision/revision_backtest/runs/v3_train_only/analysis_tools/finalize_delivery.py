"""Postprocessing only. Run after the sequential main pipeline has exited."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,pathlib,json,shutil,zipfile
ROOT=pathlib.Path(r'C:\Users\perfe\Desktop\2\xiu\revision_backtest')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import pandas as pd
from v3_audit import RUN,sha,save_csv,save_json,integrity
from v3_engine import records,context
from v3_reports import report,package,summary
from v2_reports import write_table
from verify_completed_artifacts import verify

def finalize():
 context();integrity()
 if not verify():raise RuntimeError('Completed artifact verification failed; no delivery replacement')
 # Report is read-only with respect to success artifacts; verify again after it.
 counts=report()
 if not verify():raise RuntimeError('Post-report hash/accounting verification failed')
 r=RUN/'results';cost=pd.read_csv(r/'cost_sensitivity.csv')
 main=cost[cost.variant.isin(['Vanilla','A0'])]
 ablation=cost[(cost.algorithm=='ppo')&cost.variant.isin(['A0','A1','A2','A6','A7','A8','A9'])&cost.fee.isin(['F0','F1'])]
 write_table(r/'main_only_cost_sensitivity.csv',main)
 write_table(r/'main_only_mean_sd.csv',summary(main))
 write_table(r/'fixed_ppo_F0_F1_only.csv',ablation)
 write_table(r/'fixed_ppo_F0_F1_mean_sd.csv',summary(ablation))
 priority=main[(main.phase==1)&(main.fee=='F1')]
 compact=[]
 for key,g in priority.groupby(['sector','algorithm','variant']):
  row=dict(zip(['sector','algorithm','variant'],key));row['n']=len(g)
  for col in ['AR','SR','MDD','delta_AR_vs_F0_pp','delta_SR_vs_F0','delta_MDD_vs_F0_pp','cumulative_cost','mean_gross_turnover']:
   row[col+'_mean']=g[col].mean();row[col+'_sd']=g[col].std(ddof=1)
  row.update(accounting_scope='unadjusted_price_cost_sensitivity',economic_return_validated=False)
  compact.append(row)
 write_table(r/'priority_F1_cost_effects.csv',pd.DataFrame(compact))
 last={x['task_id']:x for x in records()};failed=[x for x in last.values() if x['status'] in ['failed','blocked']]
 expected=dict(actual_real_data_train=96,actual_real_data_eval=492,nonrl_fit=4,nonrl_eval=56,real_api_short_run=1,regression_tests=1)
 delivery=dict(actual=counts,expected_if_all_inputs_remain_valid=expected,complete=all(counts.get(k)==v for k,v in expected.items()) and not failed,failed_or_blocked=failed,finalizer_sha256=sha(__file__),scope='unadjusted_price_cost_sensitivity; no total-return claim',extra_event_tests='Two synthetic entitlement/timing cases, not RL training',formal_main_train_cap=96)
 save_json(RUN/'delivery_verification.json',delivery)
 lines=['# 可据实使用的审稿回复素材','', '实际计算次数见 delivery_verification.json；未完成任务不得写为完成。', '', '作者澄清后的实现：MGT-derived observations are used during policy training in 2022. Validation and testing use observed market data for all policies. The comparison therefore evaluates the effect of forecast-augmented training on subsequent portfolio performance.', '', '2023仅用于固定检查点选择，2024是此前已查看的固定历史评估期。训练、验证、测试的账本和奖励价格均来自真实行情；验证与测试的外生观测及完整预热在各表示之间相同。', '', '## R1.2 / R2.7 / R4 / R5：费用实证范围', '', '固定F1训练策略在F0—F6下分别驱动完整自融资账本。下列为F1相对F0的逐种子变化均值；AR/MDD用百分点，SR用绝对差。']
 for row in compact:
  lines.append('- {sector} {algorithm} {variant}，n={n}：ΔAR={delta_AR_vs_F0_pp_mean:.4f}个百分点，ΔSR={delta_SR_vs_F0_mean:.4f}，ΔMDD={delta_MDD_vs_F0_pp_mean:.4f}个百分点。'.format(**row))
 lines+=['', '该分析基于作者确认的Tushare不复权价和固定分数份额。没有完整公司行动资料，除权原始价格跳变不能表述为股东经济损失；结果不称total return，也不能将现实约束要求全部勾为fully_resolved。费用后仓位、实际买卖额及复利路径随费用变化，AR/SR/MDD不要求每项严格单调。', '', '## R1.3 / R2.6：独立验证与固定PPO', '', '以2023 raw F1净Sharpe选择预设20/40/60/80/100%检查点，平分取最早。固定PPO的A0/A1/A2/A6/A7/A8/A9在相同容量、预算和RL种子下比较；规范的F0/F1结果在fixed_ppo_F0_F1_only和mean_sd表。该消融检验训练增强输入差异，不是测试时实时删除图的干预。', '', '## R1.4 / R2.5：条件配对区间', '', '同日期重采样索引共用于两策略及三个RL种子；逐种子计算指标差，再对种子平均。主block10/B2000，block5/20为敏感性；报告95%区间而无p值。它们条件于固定merge、股票池、时期和已训练策略，不是预测器和RL共同重复训练。见paired_bootstrap.csv。', '', '## R5.4：同训练输入的非RL对照', '', 'Ridge固定alpha=1、Top3，分别以2022 Vanilla/A0展平状态拟合，共四次小型非RL拟合。训练标签为真实close[t+2]/close[t+1]-1且不跨入2023；StandardScaler只fit对应训练输入；测试统一raw状态。无测试期预测Top3要求。', '', '## 仍未解决', '', '完整公司行动经济收益、完整市场执行约束、未知增强器/图生成截止和定义、预测初始化、跨股票注意力、完整预测器多种子重训、奖励网格及新股票/新时期外推均不能由本轮代替。来源未知不冒充已证污染，也不宣称无泄漏。具体条目见BLOCKED_ITEMS与REVIEWER_STATUS。']
 (RUN/'REVIEWER_RESPONSE_MATERIAL_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 package()
 src=ROOT/'results_to_send_v3.zip'
 with zipfile.ZipFile(src) as z:
  assert z.testzip() is None
  assert not any('/prepared/' in n or 'local_event_cache' in n or n.endswith(('.h5','.npy','.npz','.joblib')) for n in z.namelist())
 out=pathlib.Path(r'C:\Users\perfe\Documents\Codex\2026-10-24\c-users-perfe-desktop-2-xiu\outputs');out.mkdir(exist_ok=True)
 dest=out/src.name;shutil.copy2(src,dest);assert sha(src)==sha(dest)
 save_json(out/'results_to_send_v3_delivery.json',dict(**delivery,zip_sha256=sha(dest),zip_bytes=dest.stat().st_size))
 print(json.dumps(dict(delivered=str(dest),complete=delivery['complete'],actual=counts),ensure_ascii=False),flush=True)
if __name__=='__main__':finalize()
