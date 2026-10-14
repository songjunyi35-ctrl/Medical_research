"""Read-only link verification for generated reports, excluding root-context CARD fragments."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
from urllib.parse import unquote
from scripts.structured_background_protocol import ROOT,sha,write_new


def main():
    session=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
    names=['small_rotation','image_blur','small_rotation_full864','letterbox_fidelity','development_error_anatomy','fit_subregion_labels','resumed_eight_hour']
    reports=['RESULT.md','INTERPRETATION.md','ERROR_CHANGE_DESCRIPTION.md','SAMPLING_INTERACTION.md','RESEARCH_SYNTHESIS.md','PRIMARY_SOURCE_CHECK.md','POINT_RECORD_CORRECTION.md','EXECUTION_ORDER.md']
    checked=[];issues=[];files={}
    for name in names:
        rec=ROOT/f'archive/research_records/2026-09-10_{name}'
        for report in reports:
            path=rec/report
            if not path.exists():continue
            files[str(path.relative_to(ROOT))]=sha(path)
            for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
                target=match.group(1).strip().strip('<>')
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
                target=unquote(target.split('#',1)[0])
                resolved=(path.parent/target).resolve()
                entry={'source':str(path.relative_to(ROOT)),'target':target,'exists':resolved.exists()}
                checked.append(entry)
                if not resolved.exists():issues.append(entry)
    result={'status':'passed' if not issues else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'relative_links_checked':len(checked),'files_sha256':files,'issues':issues,'card_fragments_excluded_because_they_are_rendered_at_repo_root':True,'new_forwards':0}
    output=session/('ARCHIVE_LINK_AUDIT_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json');write_new(output,result)
    print(json.dumps({'path':str(output),'status':result['status'],'links':len(checked),'issues':issues},indent=2))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
