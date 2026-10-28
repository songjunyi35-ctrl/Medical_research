"""Register audit history and keep study-head metadata pointing to highest Scan."""
import argparse
from scripts.segresnet_auto_inference12h_20261028.register_candidate_v1 import main as register
from scripts.segresnet_auto_inference12h_20261028.common import *

def main(candidate):
 register(candidate);path=ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=read(path)
 own=[r for r in board['test270_tuned_native_rank_by_scan'] if r['method'].startswith('segresnet_w32_e171_auto12h_')]
 best=max(own,key=lambda r:(r['scan_dice'],r['subject_dice']))
 board['latest_segresnet_auto12h_interim_20261028']=dict(method=best['method'],scan_dice=best['scan_dice'],subject_dice=best['subject_dice'],point_hits=best['point_hits'],test_tuned=True,research_continues=True,result_path=best['result_path'])
 write(path,board)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args();main(a.candidate)
