"""Record the user's worker scope correction without touching training files."""
import datetime
import json

from .governance import ROOT, RECORD, atomic_json


def replace_block(name, body):
    path = ROOT / name
    text = path.read_text()
    start = '<!-- RACE_V2_STAGE_20261029 -->'
    end = '<!-- END_RACE_V2_STAGE_20261029 -->'
    assert text.count(start) == text.count(end) == 1
    first = text.index(start) + len(start)
    last = text.index(end)
    path.write_text(text[:first] + '\n' + body + '\n' + text[last:])


def main():
    links = ('[首批108](archive/research_records/2026-10-29_race_v2_stage/INITIAL_STAGE_RESULT.md) · '
             '[推理创新方案](archive/research_records/2026-10-29_race_v2_stage/INFERENCE_INNOVATION_PLAN.md) · '
             '[新机制108结果](archive/research_records/2026-10-29_race_v2_stage/INFERENCE_INNOVATION108_RESULT.md) · '
             '[实时状态](archive/research_records/2026-10-29_race_v2_stage/STAGE_STATUS.json) · '
             '[清理审计](archive/research_records/2026-10-29_race_v2_stage/cache_cleanup_20261029/RESULT.md)')
    scope = ('用户最新明确：1154259 GPU0+1保持正在执行的80新增轮／9760更新RACE-v2双卡训练，'
             '进程、源码冻结、数据、优化器、学习率和预算均不改；1154121 GPU0做RACE单模型推理创新，'
             'GPU1做纯SegResNet双模型推理创新。第二训练候选只做过CPU评审，未实施，现已取消。')
    inference = ('首批108单模equal8 Scan0.768011881，双模固定0.75融合0.769554565，'
                 '分折融合0.768177869，稳定优势未确认。推理侧继续两项已登记新机制：'
                 'A局部跨镜像相关分歧融合16配置，B配对镜像协方差收缩融合12配置；'
                 '只读复用108视图，新增网络前向0，部署单模8／双模16前向。')
    inference += ('新机制已完成：A最高Scan0.768314657／Subject0.758473906／Point98，'
                  '对equal8差+0.000302776；B最高Scan0.769835511／Subject0.759471253／Point99，'
                  '对旧固定融合差+0.000280946，两者患者配对区间均跨0，未确认稳定收益。')
    bounds = ('保持单T1ce、原972／108／270与原生1mm完整裁剪评分；108选择后最终最多两个候选270，'
              '旧RACE270视图经完整SHA审计后直接复用，新模型生成自身预测。保留test-tuned历史，'
              '单模型与纯SegResNet融合分别报告，禁GT路由、额外模态和nnU-Net。尚无本轮新270分数，目标0.8待实测。')
    storage = ('按用户授权删除1134份旧缓存，实际释放2.900753GiB，清理后空闲约23.19GiB；'
               '构建源码／索引／SHA、193份模型和当前worker缓存均保留，训练未中断。'
               '大数组仅worker/tmp，持久空闲至少20GiB，总新增上限0.75GiB。'
               '资源仅按原保守24小时截止执行，条件96小时不预支，不重启旧协调器。')
    replace_block('AGENTS.md', '## 2026-10-29：双卡训练保持原执行，另一worker推进推理创新\n\n' +
                  scope + '\n\n' + inference + '\n\n' + bounds + '\n\n' + storage + '\n\n' + links)
    replace_block('AFLOW_BRATS_HANDOFF_CURRENT.md', '## 2026-10-29：训练与推理创新两worker并行\n\n' +
                  scope + '\n\n' + inference + '\n\n' + bounds + '\n\n' + storage + '\n\n' + links)
    replace_block('AFLOW_BRATS_RESULTS.md', '## 2026-10-29：原80轮微调继续，新增两路有界推理创新\n\n' +
                  '**问题→** 单T1ce SegResNet目标270 Scan>0.8000；旧RACE完整推理0.789263。\n' +
                  '**做法→** ' + scope + '\n' + '**结果→** ' + inference + '\n' +
                  '**决定→** ' + bounds + '\n\n' + storage + '\n\n' + links)
    card = ('## 2026-10-29：参照收益校准与局部推理依赖建模｜训练／108研究并行\n\n'
            '**一句话：** 学习何时参考有益，检验镜像之间及模型之间的局部相关信息。\n'
            '**出发点：** 旧RACE完整推理270 Scan0.789263，首批融合稳定收益未确认。\n'
            '**具体做法：** 原80轮参照收益微调保持不变；另一worker研究16项单模局部相关分歧与12项双模配对协方差收缩。\n'
            '**结果：** 新机制108单模0.768314657／双模0.769835511，较各自旧最强参照差+0.000303／+0.000281，区间均跨0，无本轮新270评分。\n'
            '**是否支持直觉：** 首批等权镜像有收益；融合分折区间跨0，新局部机制不能把扰动代理当真实错误。\n'
            '**代价/下一步：** 新解码网络0，部署8／16前向；最终模型和配方只在108选，最多两个270候选。\n\n' + links)
    replace_block('AFLOW_BRATS_METHOD_CATALOG.md', card)
    replace_block('AFLOW_BRATS_LEADERBOARD.md', '原80轮双卡训练继续，另一worker两卡推进单模／双模推理创新。'
                  '本轮尚无新270榜单成绩。' + '\n\n' + links)
    path = ROOT / 'AFLOW_BRATS_LEADERBOARD.json'
    board = json.loads(path.read_text())
    board['active_research_stage_20261029'].update(
        status='training_unchanged_parallel_inference_innovation', new_test270_results=False,
        inference_new_configuration_limits={'single': 16, 'fusion': 12},
        training_scope='existing80_new_epochs_unchanged_no_second_training_candidate')
    atomic_json(path, board)
    atomic_json(RECORD / 'SCOPE_DOCUMENT_UPDATE.json', {
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'state': 'updated', 'training_files_or_processes_modified': False,
        'new270_scores': False, 'new_neural_forwards': 0,
        'scope_source': 'INFERENCE_INNOVATION_SCOPE_20261029.json'})


if __name__ == '__main__':
    main()
