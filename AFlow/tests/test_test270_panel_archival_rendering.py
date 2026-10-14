import copy
import json
from pathlib import Path
import pytest
from scripts.archive_verified_test270_panel import parse_existing_table,render_leaderboard


HEADER='| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |\n|---:|---|---:|---:|---:|---:|---|\n'


def fixture():
    rows=[{'rank':1,'method':'a','scan_dice':.7,'subject_dice':.6,'locator_forwards':1},{'rank':2,'method':'b','scan_dice':.6,'subject_dice':.5,'locator_forwards':8}]
    text='# fixture\n\n更新时间：old UTC\n\nUNCHANGED BEFORE\n\n## Reused internal test270：全自动direct locator\n\n'+HEADER+'| **1** | **A** | **`0.700000000`** | **`0.600000000`** | `0.8` | `1×` | 历史最高 |\n| 2 | B | `0.600000000` | `0.500000000` | `0.7` | `8×` | old |\n\n关键增益：\n\nUNCHANGED AFTER\n'
    return text,rows


def test_insert_sort_keeps_old_rows_values_and_other_sections():
    text,old=fixture();ranked=copy.deepcopy(old)
    new={'rank':1,'method':'new','display_name':'C','scan_dice':.8,'subject_dice':.7,'point_hits':243,'point_hit':.9,'locator_forwards':1}
    ranked.insert(0,new)
    for i,row in enumerate(ranked,1):row['rank']=i
    updated=render_leaderboard(text,old,ranked,{'new'},'archive/result.md')
    parsed=parse_existing_table(updated,ranked)
    assert set(parsed)=={'a','b','new'} and 'UNCHANGED BEFORE' in updated and 'UNCHANGED AFTER' in updated
    assert parsed['a'][6].startswith('历史状态：') and '0.800000' in updated
    assert old[0]['rank']==1 and old[0]['scan_dice']==.7


def test_missing_rows_or_wrong_metrics_refused():
    text,rows=fixture()
    with pytest.raises(ValueError):parse_existing_table(text,rows[:1])
    rows[0]['scan_dice']=.8
    with pytest.raises(ValueError):parse_existing_table(text,rows)


def test_current_canonical_table_is_parseable_without_mutation():
    root=Path(__file__).resolve().parents[1]
    value=json.loads((root/'AFLOW_BRATS_LEADERBOARD.json').read_text())
    text=(root/'AFLOW_BRATS_LEADERBOARD.md').read_text()
    mapped=parse_existing_table(text,value['test270_direct_locator_rank_by_scan'])
    assert len(mapped)==len(value['test270_direct_locator_rank_by_scan'])
