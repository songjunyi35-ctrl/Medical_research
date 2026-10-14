"""Inventory existing research documents without editing historical evidence."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'archive/reviews/2026-09-14_complete_document_inventory'
EXCLUDED={'.git','.pytest_cache','.agents','.codex','third_party','vendor','node_modules','__pycache__'}
OFFICE={'.md','.pdf','.doc','.docx','.ppt','.pptx','.rst'}
EVIDENCE={'.json','.jsonl','.csv','.tsv','.txt','.log','.out','.yaml','.yml','.sha256','.xml'}


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def relative_link(path,label):
    target=os.path.relpath(ROOT/path,DEST)
    return f'[{label.replace("[","(").replace("]",")") }](<{target}>)'


def group(path):
    parts=Path(path).parts
    if len(parts)==1:return '01_当前根文档'
    if parts[0]=='archive':
        return {'handoffs_pre_20260813':'02_早期项目与交接',
                'research_documents':'03_早期独立研究文档',
                'handoffs_20260819':'04_八月交接与计划',
                'legacy_root_backups_20260819':'05_历史根文档备份',
                'research_records':'06_逐项实验档案',
                'reviews':'07_文献调研与审计'}.get(parts[1],'08_其他归档')
    if parts[0]=='workspace':return '09_工作目录补充文档'
    return '10_其他项目说明'


def main():
    if DEST.exists():raise FileExistsError('inventory already exists')
    proc=subprocess.run(['rg','--files','--hidden','--no-ignore'],cwd=ROOT,text=True,capture_output=True,check=True)
    names=sorted(set(proc.stdout.splitlines()))
    documents=[];evidence=[];skipped=[];headings=[]
    for name in names:
        path=Path(name)
        if any(part in EXCLUDED for part in path.parts):continue
        source=ROOT/path
        if source.is_symlink():
            skipped.append({'path':name,'reason':'symlink not packaged'});continue
        low=source.name.lower()
        backup=('.md.' in low or '.md_' in low or low.endswith('.orig'))
        readable=source.suffix.lower() in OFFICE or ('.md' in low and backup)
        if not readable:
            if source.suffix.lower() in EVIDENCE:
                evidence.append({'path':name,'bytes':source.stat().st_size,'group':group(name)})
            continue
        body=None;title=source.name
        if '.md' in low or source.suffix.lower()=='.rst':
            body=source.read_text(errors='replace')
            for line in body.splitlines():
                if re.match(r'^#{1,6}\s+\S',line):title=line.lstrip('#').strip();break
        dates=re.findall(r'(?<!\d)(2026)[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?!\d)',name)
        date='-'.join(dates[0]) if dates else ''
        documents.append({'path':name,'title':title,'group':group(name),'path_date':date,
                          'is_backup':backup,'bytes':source.stat().st_size,'sha256':sha(source)})
        if body is not None and not backup and (path.parts[:2]==('archive','handoffs_pre_20260813') or len(path.parts)==1):
            for i,line in enumerate(body.splitlines(),1):
                if not re.match(r'^#{1,3}\s+',line):continue
                match=re.search(r'(?<!\d)(2026)[-/](\d{2})[-/](\d{2})(?!\d)',line)
                if match:headings.append({'date':'-'.join(match.groups()),'path':name,'line':i,'heading':line.lstrip('#').strip()})
    assert documents and evidence
    DEST.mkdir(parents=True)
    by_group=defaultdict(list)
    for row in documents:by_group[row['group']].append(row)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),
              'scope':'existing non-symlink repository documents, including ignored files and workspace; not deleted git history or external mounts',
              'excluded_directory_components':sorted(EXCLUDED),'document_count':len(documents),
              'backup_count':sum(r['is_backup'] for r in documents),
              'document_bytes':sum(r['bytes'] for r in documents),'groups':dict(Counter(r['group'] for r in documents)),
              'documents':documents,'machine_evidence_count':len(evidence),'machine_evidence_hashed':False,
              'symlinks_excluded':skipped,'date_semantics':'path_date is a filename/directory marker, not a guaranteed original creation date'}
    (DEST/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with (DEST/'EVIDENCE_FILES.tsv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['group','path','bytes'],delimiter='\t');writer.writeheader();writer.writerows(evidence)
    pages=[]
    for number,(category,entries) in enumerate(sorted(by_group.items()),1):
        entries.sort(key=lambda r:(r['path_date'] or '0000-00-00',r['path']))
        page=f'{category}.md';pages.append((category,page,len(entries)))
        lines=[f'# {category[3:]}：全部文件','',
               '按路径日期标记、再按路径排序。日期是归档/文件名标记，不保证等于最初研究日期；备份与历史错误均保留。','']
        current=None
        for row in entries:
            parent=str(Path(row['path']).parent)
            if parent!=current:
                lines.extend(['',f'## {parent}','']);current=parent
            note='【备份】' if row['is_backup'] else ''
            lines.append(f"- {note}{relative_link(row['path'],row['title'])} — `{row['path']}`")
        (DEST/page).write_text('\n'.join(lines)+'\n')
    timeline=['# 早期长文中的日期章节导航','',
              '只索引根文档和早期交接中带明确日期的一级至三级标题；这是导航，不是完整事件年表。长文也可能包含更早但没有日期标题的记录。','']
    for row in sorted(headings,key=lambda r:(r['date'],r['path'],r['line'])):
        timeline.append(f"- {row['date']}：{relative_link(row['path'],row['heading'])}（第{row['line']}行，`{row['path']}`）")
    (DEST/'EARLY_TIMELINE.md').write_text('\n'.join(timeline)+'\n')
    start=[
        ('archive/handoffs_pre_20260813/AFLOW_BRATS.md','最早项目主文档'),
        ('archive/handoffs_pre_20260813/AFLOW_BRATS_WEEKLY_REPORT.md','早期完整周报与实验流水账'),
        ('archive/handoffs_pre_20260813/AFLOW_BRATS_RESULTS.md','早期完整结果总账'),
        ('archive/handoffs_pre_20260813/AFLOW_BRATS_HANDOFF.md','早期总交接'),
        ('AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md','8月13日起长交接'),
        ('archive/handoffs_20260819/AFLOW_BRATS_HANDOFF_20260819_CONTINUATION.md','8月19日起连续交接（全部续篇见分类目录）'),
        ('AFLOW_BRATS_METHOD_CATALOG.md','所有方法的通俗目录与成败记录'),
        ('AFLOW_BRATS_HANDOFF_CURRENT.md','当前交接'),
        ('AFLOW_BRATS_LEADERBOARD.md','分队列成绩与证据边界')]
    index=['# AFlow 从早期至今的完整科研文档总索引','',
           f"盘点日期：2026-09-14。当前工作树发现 **{len(documents)}份文档/图表文件**，其中{manifest['backup_count']}份备份；另有{len(evidence)}项机器报告、日志或配置等证据文件清单。",'',
           '这不是新的简要总结，而是逐文件全量导航。所有原文件保持原样；旧错误、失败、已被替代方案和备份均不删除。不同阶段的结果口径不可直接混排。','',
           '## 如果要从头阅读','']
    for i,(path,label) in enumerate(start,1):
        assert (ROOT/path).is_file()
        index.append(f'{i}. {relative_link(path,label)}')
    index.extend(['','[早期长文的日期章节导航](EARLY_TIMELINE.md)可帮助找到六月、七月等长文内部记录，避免只看标题日期误以为历史从八月才开始。','',
        '## 全量分类目录','', '| 分类 | 文件数 | 入口 |','|---|---:|---|'])
    for category,page,count in pages:index.append(f'| {category[3:]} | {count} | [全部文件](<{page}>) |')
    index.extend(['','## 机器证据与原始版本','',
        '- [全部文档清单及SHA-256](MANIFEST.json)：每份文档的路径、标题、大小、日期标记、是否备份与内容哈希。',
        '- [机器报告/日志/配置全路径清单](EVIDENCE_FILES.tsv)：包括JSON、JSONL、CSV/TSV、文本日志、YAML、XML和SHA文件；仅盘点文件名和大小，没有重读原始指标或大日志内容。',
        '- [全套可读文档ZIP](AFLOW_RESEARCH_DOCUMENTS_20260914.zip)：保留原目录结构，包含上面的全部正文/历史/备份/PDF等文件，以及本索引；不包含权重、影像、训练缓存和机器证据内容。',
        '- [打包验证与来源限制](VERIFICATION.json)。','',
        '## “完整”的准确范围','',
        '覆盖当前仓库实际存在的项目文档：根目录、archive、workspace以及其他项目说明，包括被gitignore忽略的文件。排除.git、第三方源码/依赖、工具缓存、agent/技能内部目录、符号链接文件；没有递归外部符号链接目录，也没有恢复Git已删除历史、外部磁盘或其他聊天里未落盘的内容。不能保证从未丢失任何历史记录。','',
        '上面的日期排序主要使用路径标记；例如8月归档的旧周报内部保留六月实验，阅读早期长文才是追溯最初记录的关键。可读文档包与机器证据清单分开，避免把数千份原始报告误当成可直接汇报的正文。'])
    (DEST/'INDEX.md').write_text('\n'.join(index)+'\n')
    generated=[DEST/'INDEX.md',DEST/'MANIFEST.json',DEST/'EVIDENCE_FILES.tsv',DEST/'EARLY_TIMELINE.md']+[DEST/p for _,p,_ in pages]
    archive=DEST/'AFLOW_RESEARCH_DOCUMENTS_20260914.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in documents:
            assert sha(ROOT/row['path'])==row['sha256'],'source changed while indexing'
            z.write(ROOT/row['path'],row['path'])
        for file in generated:z.write(file,str(file.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in documents:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
        entries=len(z.namelist())
    verification={'status':'passed','original_documents_untouched':True,'source_documents':len(documents),
                  'zip_entries':entries,'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive),
                  'machine_evidence_files_listed_not_packaged':len(evidence),
                  'generated_files_in_zip':[str(f.relative_to(ROOT)) for f in generated],
                  'verification_receipt_not_inside_zip':'receipt hashes the final zip; use adjacent VERIFICATION.json',
                  'earliest_dated_heading_found':min((r['date'] for r in headings),default=None),
                  'latest_dated_path_found':max((r['path_date'] for r in documents if r['path_date']),default=None),
                  'script_sha256':sha(Path(__file__))}
    (DEST/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({**verification,'groups':manifest['groups'],'index':str(DEST/'INDEX.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
