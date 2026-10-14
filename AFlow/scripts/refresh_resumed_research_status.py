"""Backed-up, idempotent mechanical status blocks derived from study artifacts."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
STUDY=ROOT/'archive/research_records/2026-09-10_small_rotation'


def main():
    status='已预注册，工程验证中，尚未正式训练或读取候选开发结果'
    if (STUDY/'TRAINING_STARTED.json').exists():status='正式训练中；无候选开发结果'
    if (STUDY/'TRAINING_DONE.json').exists():status='四条训练已完成，待独立重放与开发读出'
    if (STUDY/'RESULT.json').exists():
        r=json.loads((STUDY/'RESULT.json').read_text());status=f"开发已归档：{r['status']}，入选层={r['selected_arm']}"
    if (STUDY/'TRAINING_FAILURE.json').exists():status='训练工程/预算失败，见TRAINING_FAILURE.json；不按部分结果选择'
    text='\n'.join(['## 2026-09-10 第二个八小时续研：小角度旋转', '',
         '用户重新授权13:39:55—21:39:55 UTC，旧轮次到期不代表本轮停止。当前：'+status+'。', '',
         '**一句话：** 训练时轻微转动整个输入，检验模型是否能减少对固定朝向的依赖。',
         '**出发点：** native开发Scan0.746762/0.744177；患者模型八镜像虽提高Dice，seed12 Point168降至160。整体旋转不同于此前局部弹性形变，不复活旧负结果。',
         '**具体做法：** 仅额外20%病例三轴各±15°输入网格旋转；图像三线性、硬标签最近邻、零填充。两个采样层×两个固定种子、168轮，对照/原RNG/模型/损失/阈值/argmax定位/1×部署不变。',
         '**结果：** '+status+'；所有指标仅在对应队列内解释，不把开发分数当超榜。',
         '**是否支持直觉：** 以冻结双seed Dice/Point/患者簇CI联合门决定；患者采样层还须胜自身对照。旧点共识收据含无效JSON，旧中心点收据只有哈希；撤回其已独立验证的解读，保留两项负结果。',
         '**代价/下一步：** 预计双卡4–5h、8–10GPUh，<400MiB模型/恢复点，无新像素缓存；部署1×。先合成/fit压力检查/GPU smoke，再冻结训练。', '',
         '[七问](archive/research_records/2026-09-10_small_rotation/IDEA.md)、[预注册](archive/research_records/2026-09-10_small_rotation/PREREG.md)、[新窗口](archive/research_records/2026-09-10_resumed_eight_hour/SESSION.md)。',
         '尚无新test/outer/confirmation；旧test270/119 overall Scan0.757269662（8×），单次前向Scan0.752003176。以下所有旧阶段记录是历史，不覆盖本块。'])
    start='<!-- aflow-resumed-20260910:start -->';end='<!-- aflow-resumed-20260910:end -->'
    changes={}
    for name in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_METHOD_CATALOG.md']:
        path=ROOT/name;old=path.read_bytes();s=old.decode()
        backup=REC/(name+'.BEFORE_RESUME')
        if not backup.exists():
            with backup.open('xb') as f:f.write(old)
        block=start+'\n'+text+'\n'+end
        if start in s:
            a=s.index(start);b=s.index(end,a)+len(end);new=s[:a]+block+s[b:]
        else:new=block+'\n\n'+s
        path.write_text(new)
        changes[name]={'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()}
    output=REC/('STATUS_REFRESH_'+datetime.now(timezone.utc).strftime('%H%M%S%f')+'.json')
    with output.open('x') as f:json.dump({'created_utc':datetime.now(timezone.utc).isoformat(),'status':status,'changes':changes},f,indent=2)
    print(status)


if __name__=='__main__':main()
