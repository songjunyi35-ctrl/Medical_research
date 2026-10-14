#!/usr/bin/env python3
"""Mechanically archive the completed Flip8 result in canonical docs/leaderboards."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
METHOD = "full972_random_init_voxel_kd_epoch168_flip8_tta"


def write_text_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label}; found {text.count(old)}")
    return text.replace(old, new, 1)


def replace_line_prefix(text: str, prefix: str, replacement: str, label: str) -> str:
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}; found {len(matches)}")
    suffix = "\n" if lines[matches[0]].endswith("\n") else ""
    lines[matches[0]] = replacement + suffix
    return "".join(lines)


def update_leaderboard_json() -> None:
    path = ROOT / "AFLOW_BRATS_LEADERBOARD.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["test270_direct_locator_rank_by_scan"]
    if any(row.get("method") == METHOD for row in rows):
        raise RuntimeError("Flip8 leaderboard row already exists")
    row = {
        "rank": 0,
        "method": METHOD,
        "scan_dice": 0.7572696616942132,
        "subject_dice": 0.7412535853511771,
        "point_hit": 0.9814814814814815,
        "point_hits": 265,
        "point_denominator": 270,
        "locator_forwards": 8,
        "cohort": {"scans": 270, "subjects": 119},
        "checkpoint_sha256": "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b",
        "resolution": 128,
        "threshold": 0.0,
        "tta": True,
        "tta_formula": "all 8 combinations of spatial-axis reflection; inverse-reflect each raw logit; ordered float32 equal mean",
        "sam": False,
        "postprocessing": False,
        "scan_dice_95ci": [0.739088565033895, 0.7741517265513539],
        "subject_dice_95ci": [0.7141092440426474, 0.7661524792215904],
        "paired_vs_identity": {
            "scan_dice": 0.0055101244813866085,
            "subject_dice": 0.003926478225101276,
            "point_hits": 3,
            "scan_dice_95ci": [0.0027509939153161317, 0.008022352442559269],
            "subject_dice_95ci": [-0.00037875292107922, 0.007353958139034295],
        },
        "evidence_status": "two_fresh_inner_replications_all_gates_passed; user_authorized_post_test_development; reused_internal_test; not_pristine_blind; not_external; not_test_tuned; no_test_retuning_or_retest_allowed",
        "report_path": "workspace/BraTS/sammed3d_t1ce_wt/flip8_tta_stability_20260904/FLIP8_TEST270.json",
        "report_sha256": "29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad",
        "status": "highest_reused_internal_test_scan_subject_point_estimates; paired_scan_ci_vs_identity_positive; paired_subject_ci_crosses_zero; 8x_inference",
    }
    rows.append(row)
    rows.sort(key=lambda item: float(item["scan_dice"]), reverse=True)
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    value["updated_utc"] = "2026-09-04"
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def update_leaderboard_markdown() -> None:
    path = ROOT / "AFLOW_BRATS_LEADERBOARD.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "更新时间：2026-09-02 UTC", "更新时间：2026-09-04 UTC", "leaderboard date")
    text = replace_once(
        text,
        "- `1×/2×`表示locator forward数量或近似体素成本，不包含NIfTI预处理。",
        "- `1×/2×/8×`表示locator forward数量或近似体素成本，不包含NIfTI预处理。",
        "cost legend",
    )
    header = "| 排名 | 方法 | Scan Dice | Subject Dice | Point hit | 成本 | 状态 |\n|---:|---|---:|---:|---:|---:|---|\n"
    section = text.index("## Reused internal test270：全自动direct locator")
    table_start = text.index(header, section) + len(header)
    table_end = text.index("\n\n关键增益：", table_start)
    old_rows = text[table_start:table_end]
    if old_rows.count("\n") != 7 or "GCSAM fit864" not in old_rows or "true-128³ plain，epoch99" not in old_rows:
        raise RuntimeError("unexpected existing test270 Markdown table")
    new_rows = "\n".join(
        (
            "| **1** | **Full972 random-init voxel-KD，epoch168，固定Flip8 TTA** | **`0.757269662`** | **`0.741253585`** | **`0.981481`** | `8×` | 两个fresh inner全门后唯一一次post-test/reused评测；三项点估计第1；非盲测/外部确认 |",
            "| 2 | GCSAM fit864，train-inner-selected epoch184 | `0.752003176` | `0.737266420` | `0.933333` | `1×` | 用户授权failed-gate探索性；1× Scan点估计第1；非Full972 final，CI跨0且Point低 |",
            "| 3 | Full972 random-init voxel-KD，epoch168 | `0.751759537` | `0.737327107` | `0.970370` | `1×` | 用户授权failed-gate探索性；1× Subject/Point最高 |",
            "| 4 | 固定logit ensemble：0.75 history168 + 0.25 EMA168 | `0.745054064` | `0.727344609` | `0.955556` | `2×` | 旧总体最高 |",
            "| 5 | 2025/2026 voxel-KD single，global epoch200 | `0.742976353` | `0.725178533` | `0.955556` | `1×` | 旧最高单模型 |",
            "| 6 | historical true-128³ plain，epoch168 | `0.741203949` | `0.723179486` | `0.929630` | `1×` | 旧单模型记录 |",
            "| 7 | EMA168 single | `0.740696468` | `0.724904822` | `0.948148` | `1×` | Dice持平、point更高 |",
            "| 8 | true-160³ full756，epoch200 | `0.737060297` | `0.719728630` | `0.874074` | `~1.95×` | 负结果；160³标签网格 |",
            "| 9 | true-128³ plain，epoch99 | `0.725820068` | `0.709412503` | `0.900000` | `1×` | 2026-08-17记录 |",
        )
    )
    text = text[:table_start] + new_rows + text[table_end:]
    text = replace_once(
        text,
        "关键增益：\n\n",
        "关键增益：\n\n- Flip8 − same-checkpoint identity：scan `+0.005510124`、subject `+0.003926478`、Point `+3/270`；paired 95% CI为`[+0.002751,+0.008022]`/`[-0.000379,+0.007354]`。Scan区间整体为正，Subject区间跨0。相对此前GCSAM Scan榜首，三项点估计为`+0.005266486/+0.003987165/+13 points`，但未做冻结的二者配对检验，不能称相对GCSAM显著。绝对CI按pretest冻结seed20260817从保存的per-case行离线修正；raw helper的seed偏差已保留归档。\n\n",
        "key gains insertion",
    )
    text = replace_once(
        text,
        "报告入口：\n\n",
        "报告入口：\n\n- Flip8：[报告](workspace/BraTS/sammed3d_t1ce_wt/flip8_tta_stability_20260904/FLIP8_TEST270.json)，[归档](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)，[修正记录](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/BOOTSTRAP_CORRECTION.json)\n\n",
        "report insertion",
    )
    text = replace_once(
        text,
        "- 最高内部test Scan点估计：GCSAM fit864 inner-selected epoch184（1×）；但只领先Full972 KD `0.000244`且paired CI跨0，Subject略低、Point少10，证据为用户授权failed-gate探索性test，且不是Full972 final。\n- 最高内部test Subject/Point点估计：Full972 random-init voxel-KD epoch168（1×）。",
        "- 最高内部test总体点估计：Full972 random-init KD168固定Flip8（8×），Scan/Subject/Point=`0.757270/0.741254/265/270`；相对same-checkpoint identity的Scan paired CI整体为正，Subject CI跨0。证据是两fresh inner全门后的唯一一次post-test/reused评测，不是盲测或外部确认。\n- 最高内部test 1× Scan点估计仍为GCSAM fit864 epoch184；最高1× Subject/Point仍为Full972 random-init KD168 identity。",
        "decision summary",
    )
    write_text_atomic(path, text)


def update_method_catalog() -> None:
    path = ROOT / "AFLOW_BRATS_METHOD_CATALOG.md"
    text = path.read_text(encoding="utf-8")
    text = replace_line_prefix(
        text,
        "| Direct locator固定Flip8 TTA |",
        "| Direct locator固定Flip8 TTA | 同一模型把MRI按8种镜像看完，翻回后平均 | ✅ 完成；reused test270三项点估计第1 | fresh A/B平均Scan/Subject `+0.006828/+0.007132`；test=`0.757270/0.741254/265` |",
        "catalog table row",
    )
    start = text.index("## 全自动Direct Locator固定Flip8测试时增强")
    end = text.index("## UCSD患者隔离外部确认", start)
    card = """## 全自动Direct Locator固定Flip8测试时增强 ✅ 完成；reused test270三项点估计第1

**一句话：** 同一模型把MRI按三轴全部8种镜像方向都看一遍，把答案翻回原方向后平均。

**状态：** 两个fresh inner复制全门通过；随后唯一一次reused internal test270四个成功门全过。只能称post-test/non-blind/non-external点估计记录，禁止test调参或重测。

**出发问题：** Full972 KD168旧test Scan=`0.751760`，距GCSAM榜首`0.752003`仅`0.000244`，而Subject/Point更高；训练随机覆盖全部三轴翻转但部署只看identity，可能留下方向偶然误差。

**具体做法：** A/B各用新subject-disjoint 864 fit/108 inner训练一个标准Full972-style KD base到固定epoch168，两个checkpoint全部完成并冻结后才访问inner。同一checkpoint依次看完整8种reflection，把raw logits逆翻回原方向并按固定顺序float32等权平均，阈值0、Point=argmax。

**改变/固定因素：** 唯一变化是推理1×→固定8×。数据、模型、checkpoint、teacher、loss、采样、增强、AdamW、epoch168、阈值0、无Segment Anything Model/连通域/后处理全部固定；没有搜索reflection子集、权重或概率平均。

**预期机制：** 训练翻转没有让卷积网络严格等变时，8种等价视图的逆变换平均会消除方向敏感噪声。旧TTA只在GT-centered refiner支线，未否定全自动direct locator。

**成功/停止门：** 每个fresh inner均须Scan/Subject严格提高、Point最多少1、两项CI下界≥`-0.003`；平均Scan须超过`0.000243638`。全部通过后只允许一次固定test；test须Scan超过GCSAM与identity、Subject不低于identity、Point最多少1。

**观察结果：** A Flip8−Identity Scan/Subject/Point=`+0.006668/+0.006930/-1`，paired CI=`[+0.003718,+0.009444]/[+0.003622,+0.010338]`；B=`+0.006988/+0.007333/+3`，CI=`[+0.004026,+0.010414]/[+0.003229,+0.011789]`。两复制平均=`+0.006828/+0.007132`。Test270 identity完整summary/per-case exact；Flip8/identity=`0.757269662/0.741253585/265` vs `0.751759537/0.737327107/262`，即`+0.005510/+0.003926/+3`。Scan paired CI=`[+0.002751,+0.008022]`整体为正，Subject CI=`[-0.000379,+0.007354]`跨0。

**直觉是否支持：** 支持。两fresh inner与test三处Scan/Subject同向，test Point增加3；方向平均消除了可复现误差。但只对identity的Scan增益有正配对CI，Subject与相对GCSAM不能称显著。

**代价与下一决定：** Stage A约6.783 GPU-hours/双A100墙钟3.404h、1,728 forwards；test 2,160 forwards。部署仍4.70M参数/同一checkpoint，峰值近1×但计算延迟约8×；workspace55MB、无新cache。raw绝对CI helper seed与pretest不符，已保留raw并从保存per-case行离线修正，不影响点估计/配对门。冻结当前公式，绝不test调参/重测；下一步只做独立外部确认。证据见[结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.json)、[raw test](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json)、[Stage A](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/STAGE_A_INNER.json)、[pretest](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/TEST270_PREACCESS.json)与[修正记录](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/BOOTSTRAP_CORRECTION.json)。

"""
    text = text[:start] + card + text[end:]
    write_text_atomic(path, text)


def update_handoff() -> None:
    path = ROOT / "AFLOW_BRATS_HANDOFF_CURRENT.md"
    text = path.read_text(encoding="utf-8")
    text = replace_line_prefix(
        text,
        "- **Direct locator固定Flip8 TTA Stage A训练中：**",
        "- **Direct locator固定Flip8 TTA已完成并刷新reused test270全面点估计：** 唯一变化是同一Full972 KD168 checkpoint从identity 1×改为全部8种三轴reflection、逆翻后float32等权raw-logit平均；阈值0、Point=argmax不变。A/B fresh inner的Scan/Subject平均提升=`+0.006828/+0.007132`，四个CI下界均为正，全部门通过。唯一一次test270先验证identity完整summary/per-case exact，再得到Flip8 Scan/Subject/Point=`0.757269662/0.741253585/265/270`，相对identity=`+0.005510124/+0.003926478/+3`，相对此前GCSAM点估计=`+0.005266486/+0.003987165/+13`。Scan paired CI `[+0.002751,+0.008022]`为正；Subject CI `[-0.000379,+0.007354]`跨0。当前只可称270 scans/119 subjects的post-test/reused/non-blind/non-external三项点估计第1；不称相对GCSAM显著，不得调参或重测。raw绝对CI helper seed偏差已保留并从保存per-case行离线修正，不影响点估计/配对门。证据见[结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.md)、[机器结果](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RESULT.json)、[raw test](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json)、[Stage A](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/STAGE_A_INNER.json)与[pretest](archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/TEST270_PREACCESS.json)。",
        "handoff Flip8 status",
    )
    text = replace_once(
        text,
        "- 推理：raw logit阈值`>0`；最高logit作为自动point；无SAM、TTA或后处理。",
        "- 当前最高总体推理：固定Flip8 TTA，8种三轴reflection逆翻后float32等权raw-logit均值，阈值`>0`、最高均值logit为自动point；无SAM或其他后处理。1× identity行继续单列比较。",
        "handoff inference",
    )
    old_summary = """| 最高Scan点估计（探索性） | GCSAM fit864 inner-selected epoch184 | `0.752003` | `0.737266` | `0.933333` | `1×` locator |
| 最高Subject/Point（探索性） | Full972 random-init voxel-KD epoch168 | `0.751760` | `0.737327` | `0.970370` | `1×` locator |"""
    new_summary = """| 内部test总体三项最高点估计（post-test） | Full972 KD168固定Flip8 TTA | `0.757270` | `0.741254` | `0.981481` | `8×` locator |
| 内部test最高1× Scan点估计 | GCSAM fit864 inner-selected epoch184 | `0.752003` | `0.737266` | `0.933333` | `1×` locator |
| 内部test最高1× Subject/Point | Full972 random-init voxel-KD epoch168 identity | `0.751760` | `0.737327` | `0.970370` | `1×` locator |"""
    text = replace_once(text, old_summary, new_summary, "handoff leaderboard summary")
    write_text_atomic(path, text)


def update_idea() -> None:
    path = ROOT / "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/IDEA.md"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "状态：七问与链路审视已冻结；尚未实现、构造split、训练或访问新inner/test",
        "状态：完成；两个fresh inner复制全门通过，唯一一次reused internal test270已完成并刷新全面点估计",
        "IDEA status",
    )
    final = """Stage A两个fresh inner复制均支持直觉。A的Flip8−Identity Scan/Subject/Point=`+0.006668/+0.006930/-1`，配对95% CI下界均为正；B=`+0.006988/+0.007333/+3`，下界也均为正。平均Scan/Subject提升=`+0.006828/+0.007132`，全部冻结门通过。

唯一一次reused internal test270上，同一Full972 KD168 checkpoint的identity=`0.751759537/0.737327107/262`，与旧报告完整summary及逐病例exact；固定Flip8=`0.757269662/0.741253585/265`，即`+0.005510124/+0.003926478/+3`。Scan paired 95% CI=`[+0.002751,+0.008022]`整体为正；Subject CI=`[-0.000379,+0.007354]`跨0。四个test成功门全过，成为270 scans/119 subjects reused internal test的Scan/Subject/Point全面点估计第1；证据仍是post-test、非盲测、非外部确认，禁止调参或重测。

raw report的描述性绝对CI helper误用seed20260826而非pretest冻结的20260817；点估计、配对seed20261207和所有成功门不受影响。保留raw，并仅从已保存per-case行离线重算正确绝对CI，无模型/缓存访问。完整解释与证据见[RESULT.md](RESULT.md)和[RESULT.json](RESULT.json)。"""
    text = replace_once(text, "待完成后填写。", final, "IDEA final result")
    write_text_atomic(path, text)


def main() -> None:
    required = {
        ROOT / "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/RAW_TEST270.json": "29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad",
        ROOT / "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/TEST270_PREACCESS.json": "9ee849448d4728a31ee54268ef0197c6713c35b79c1bc7c3bd16cf8f509b1609",
    }
    import hashlib
    for path, expected in required.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"finalizer input SHA mismatch: {path}")
    update_leaderboard_json()
    update_leaderboard_markdown()
    update_method_catalog()
    update_handoff()
    update_idea()
    print(json.dumps({"status": "updated", "method": METHOD, "canonical_files": 5}, indent=2))


if __name__ == "__main__":
    main()
