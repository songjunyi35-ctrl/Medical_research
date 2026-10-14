"""Finalist identity clarification; same three-branch, 19-recipe budget."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.complete_t1ce_10h_v2
from scripts import complete_t1ce_10h as original
from scripts.select_t1ce_10h_finalists import select
from scripts.t1ce_10h_common import *
original.select=select

if __name__=='__main__':
 try:original.main()
 except BaseException:write(R/f'COMPLETION_V3_FATAL_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
