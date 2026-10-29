"""Shared stage convention: scan equal-weight and patient-cluster uncertainty."""
import numpy as np
def _cluster(records,values):
    subjects=sorted({r['subject_id'] for r in records})
    counts=np.array([sum(r['subject_id']==s for r in records) for s in subjects])
    sums=np.array([sum(v for r,v in zip(records,values) if r['subject_id']==s) for s in subjects],np.float64)
    ix=np.random.default_rng(20261029).integers(0,len(subjects),(10000,len(subjects)))
    return sums,counts,sums[ix].sum(1)/counts[ix].sum(1),(sums/counts)[ix].mean(1)
def summary_ci(records):
    sums,n,scan,subject=_cluster(records,[r['locator_dsc'] for r in records])
    return dict(scan_dice=float(sums.sum()/n.sum()),subject_dice=float((sums/n).mean()),scan_ci95=np.quantile(scan,[.025,.975]).tolist(),subject_ci95=np.quantile(subject,[.025,.975]).tolist(),scans=len(records),subjects=len(n),point_hits=sum(r['automatic_point_hits_gt'] for r in records),bootstrap='10000 patient whole-cluster percentile seed20261029; descriptive after selection')
def paired(candidate,baseline):
    lookup={r['case_id']:r for r in baseline}
    assert len(candidate)==len(lookup) and {r['case_id'] for r in candidate}==set(lookup)
    assert all(r['subject_id']==lookup[r['case_id']]['subject_id'] for r in candidate)
    delta=np.array([r['locator_dsc']-lookup[r['case_id']]['locator_dsc'] for r in candidate])
    sums,n,scan,subject=_cluster(candidate,delta)
    return dict(scan_delta=float(sums.sum()/n.sum()),subject_delta=float((sums/n).mean()),scan_delta_ci95=np.quantile(scan,[.025,.975]).tolist(),subject_delta_ci95=np.quantile(subject,[.025,.975]).tolist(),point_hits_delta=sum(r['automatic_point_hits_gt'] for r in candidate)-sum(r['automatic_point_hits_gt'] for r in baseline),improved_scans=int((delta>0).sum()),worse_scans=int((delta<0).sum()),tied_scans=int((delta==0).sum()),bootstrap='10000 paired patient whole-cluster percentile seed20261029; descriptive after selection')
