"""Bundle indexed research evidence and report figures, not weights or imaging data."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from scripts.build_research_document_inventory import ROOT,DEST,EXCLUDED,sha


def main():
    manifest=json.loads((DEST/'MANIFEST.json').read_text())
    evidence=list(csv.DictReader((DEST/'EVIDENCE_FILES.tsv').open(),delimiter='\t'))
    # Scope machine files to research records, not unrelated runtime/API configuration.
    chosen=[];excluded=[]
    for row in evidence:
        path=Path(row['path']);parts=path.parts
        if parts[0]=='archive' or parts[:2]==('workspace','BraTS') or parts[:2]==('data','datasets') or len(parts)==1:
            chosen.append(row['path'])
        else:excluded.append(row['path'])
    proc=subprocess.run(['rg','--files','--hidden','--no-ignore','archive'],cwd=ROOT,text=True,capture_output=True,check=True)
    figures=[p for p in proc.stdout.splitlines() if Path(p).suffix.lower() in {'.png','.svg','.jpg','.jpeg','.webp'}
             and not any(part in EXCLUDED for part in Path(p).parts) and not (ROOT/p).is_symlink()]
    originals=sorted(set([r['path'] for r in manifest['documents']]+chosen+figures))
    entries=[{'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':sha(ROOT/p)} for p in originals]
    full_manifest={'documents':manifest['document_count'],'machine_research_files':len(chosen),'additional_report_figures':len(figures),
                   'total_original_files':len(entries),'files':entries,'excluded_nonresearch_machine_config_or_logs':excluded,
                   'scope':'existing indexed documents plus archive/BraTS-workspace/dataset-manifest research evidence and archive figures; excludes weights, raw images, caches, source code, external mounts and deleted history'}
    (DEST/'FULL_PACKAGE_MANIFEST.json').write_text(json.dumps(full_manifest,ensure_ascii=False,indent=2)+'\n')
    page=['# 从头到现在：完整科研文档与记录入口','',
          '盘点截至2026-09-14。这次提供的是现存记录的完整导航与打包，不是只提炼近期成果。旧错误、失败方法、被替代计划和备份均保留。','',
          f"- **[完整研究记录包](AFLOW_COMPLETE_RESEARCH_RECORDS_20260914.zip)**：{len(entries)}个原文件，包含{manifest['document_count']}份正文/图表/备份、{len(chosen)}份机器研究记录和{len(figures)}份额外报告配图，保留目录结构。",
          '- **[可读文档包](AFLOW_RESEARCH_DOCUMENTS_20260914.zip)**：只包含正文、PDF等可读文档与导航，适合先浏览。',
          '- **[按阶段、目录排列的全量总索引](INDEX.md)**：逐文件链接，不省略失败方法和历史备份。',
          '- **[早期长文的日期章节导航](EARLY_TIMELINE.md)**：找到六月、七月等被保存在八月归档长文内的早期记录。',
          '- [完整包文件清单与SHA](FULL_PACKAGE_MANIFEST.json)、[完整包验证](FULL_PACKAGE_VERIFICATION.json)。','',
          '## 从最早文档开始读','',
          '1. [早期项目主文档](../../handoffs_pre_20260813/AFLOW_BRATS.md)',
          '2. [早期完整周报／实验流水账](../../handoffs_pre_20260813/AFLOW_BRATS_WEEKLY_REPORT.md)',
          '3. [早期完整结果总账](../../handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md)',
          '4. [早期总交接](../../handoffs_pre_20260813/AFLOW_BRATS_HANDOFF.md)',
          '5. [全部后续文档分类导航](INDEX.md)：8月交接续篇、方法预注册、所有实验结果、外部来源审计、方法百科、9月工作流研究及部署补充。','',
          '## 完整性边界','',
          '范围是当前工作树实际留存、扫描时可访问的项目文档。没有恢复已删除Git历史、其他聊天中未保存的内容、外部挂载或符号链接目录；因此不能保证项目从创建以来从未丢失任何记录。',
          '完整包不包含模型权重、原始医学影像、二进制训练/预测缓存、源码以及与科研正文无关的运行配置。机器记录清单中未打包的非科研配置/日志路径明确列在FULL_PACKAGE_MANIFEST.json。原始文档内引用这些未打包的大文件或源码时，需回原仓库查看。',
          '两个ZIP均已验证；完整包的最终校验收据在ZIP旁，因为收据包含ZIP自身哈希。包内索引的ZIP/收据下载链接仅在原仓库目录中有效。旧材料仅作历史证据，最新成绩以当前分队列排行榜为准。']
    (DEST/'COMPLETE_INDEX.md').write_text('\n'.join(page)+'\n')
    generated=[p for p in DEST.iterdir() if p.suffix in {'.md','.json','.tsv'} and p.name!='FULL_PACKAGE_VERIFICATION.json']
    archive=DEST/'AFLOW_COMPLETE_RESEARCH_RECORDS_20260914.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in entries:
            assert sha(ROOT/row['path'])==row['sha256'],'source changed'
            z.write(ROOT/row['path'],row['path'])
        for file in generated:z.write(file,str(file.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in entries:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
        count=len(z.namelist())
    receipt={'status':'passed','original_files':len(entries),'zip_entries':count,'zip_bytes':archive.stat().st_size,
             'zip_sha256':sha(archive),'all_original_entry_sha256_verified':True,
             'machine_research_files':len(chosen),'archive_figures':len(figures),'excluded_nonresearch_machine_files':len(excluded)}
    (DEST/'FULL_PACKAGE_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
