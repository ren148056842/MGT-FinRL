"""V3 training-only augmentation runner. Does not alter v1/v2 code or result directories."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,sys,subprocess,time,shutil
import torch
torch.set_num_threads(1)
from v3_audit import ROOT,RUN,PYTHON,audit,integrity,save_json,load_json
from v3_data import prepare
from v3_engine import train,evaluate,nonrl,record,plan
from v3_reports import report,package

def tests():
 p=subprocess.run([sys.executable,'-m','unittest','-v','test_backtest','test_v3'],cwd=ROOT,capture_output=True,text=True,errors='replace');(RUN/'regression_tests.log').write_text(p.stdout+p.stderr,encoding='utf-8');record('v3_regression','completed' if p.returncode==0 else 'failed',kind='regression_tests')
 if p.returncode:raise RuntimeError('V3 regressions failed; see regression_tests.log')
 print('Necessary v3 regressions passed; immediately starting real training.',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['audit','prepare','test','train','evaluate','nonrl','report','all']);ap.add_argument('--phase',default='all',choices=['all','1','2','3']);ap.add_argument('--sector',choices=['bank','healthcare']);ap.add_argument('--algorithm',choices=['a2c','ppo','sac','td3','ddpg']);ap.add_argument('--seed',type=int,choices=[1,2,3]);a=ap.parse_args()
 if os.path.normcase(os.path.realpath(sys.executable))!=os.path.normcase(os.path.realpath(PYTHON)):raise RuntimeError('Use required finrl-meta interpreter')
 RUN.mkdir(exist_ok=True,parents=True)
 if a.command in ['audit','all']:audit()
 if a.command in ['prepare','all']:prepare();plan()
 if a.command in ['test','all']:tests()
 if a.command=='train':train(a.phase,a.sector,a.algorithm,a.seed)
 if a.command=='evaluate':evaluate(a.phase)
 if a.command=='nonrl':nonrl()
 if a.command=='all':
  train('1');report();package('results_priority_v3.zip');save_json(RUN/'priority_completion_counts.json',load_json(RUN/'completion_counts.json'))
  nonrl();train('2');report();package();train('3')
 if a.command in ['report','all']:report();integrity();package()
if __name__=='__main__':main()
