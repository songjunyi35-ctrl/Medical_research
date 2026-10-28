"""Release explicitly listed completed derived histograms; preserve all cuts/scores."""
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.rolling_cleanup_v1 import no_readers


def main():
    for identifier in ['r1_0200_case_consistency_extension', 'd1_0000_joint_fusions', 'a2_0100_phase1_fusions', 'a2_0101_phase64_fusions', 'a2_0102_phase128_fusions', 'a2_0103_phase256_fusions', 'a3_0100_flip7_phase1_fusions', 'a3_0101_flip7_phase64_fusions', 'a3_0102_flip7_phase128_fusions', 'e1_0000_cross_phase64', 'e1_0001_cross_phase128', 'f1_0000_brain_domain', 'c1_0000_dense_crf']:
        guard();folder=R/'cleanup_records'/('histograms_'+identifier);folder.mkdir(parents=True,exist_ok=True)
        if (folder/'DONE.json').exists():continue
        consumers=[read(p) for p in (R/'cpu_queue_v2').glob('*.json') if read(p).get('source_analysis')==identifier]
        if not consumers or not all((R/'analyses'/j['id']/'DONE.json').exists() for j in consumers):continue
        assert (R/'analyses'/identifier/'DONE.json').exists()
        target=TMP/'analyses'/identifier/'histograms';assert target.resolve().is_relative_to((TMP/'analyses').resolve())
        assert not no_readers(target)
        files=[dict(path=str(p.relative_to(TMP)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.rglob('*.npy'))]
        expected=len(read(R/'analyses'/identifier/'RANKING.json')['calibration_shortlist'])*270
        assert len(files)==expected
        write(folder/'FILES.json',files)
        write(folder/'PLAN.json',dict(at=now(),target=str(target),source_sha256=sha(Path(__file__)),all_consumers_done=True,
              preserve='all score/cut matrices, ranking tables, candidate masks, core logits, native inputs and weights',
              regeneration='frozen recipes and retained core logits/phase maps/derived probability fields; same numerical source retained',directory_skeleton_preserved=True))
        for entry in files:(TMP/entry['path']).unlink()
        record=dict(at=now(),state='complete',bytes_freed=sum(x['bytes'] for x in files),files=len(files),old_cache_or_model_deleted=False,manifest_sha256=sha(folder/'FILES.json'))
        write(folder/'DONE.json',record);write(R/'analyses'/identifier/'DERIVED_HISTOGRAMS_DELETED.json',record);print(identifier,record['bytes_freed']/2**30,flush=True)
if __name__=='__main__':main()
