"""Otsu-mask student contract sharing the previously frozen clean folds/teachers."""
from scripts import no_n4_clean_protocol as parent
ROOT=parent.ROOT
RECORD=ROOT/"archive/research_records/2026-09-05_n4_otsu_clean_replication"
SHARED_RECORD=parent.RECORD
PARENT_WORK=parent.WORK
WORK=ROOT/"workspace/BraTS/sammed3d_t1ce_wt/n4_otsu_clean_20260905"
SOURCE=parent.SOURCE
CONTROL_CACHE=parent.CONTROL_CACHE
NATIVE=parent.NATIVE
SOURCE_SHA=parent.SOURCE_SHA
INDEX_SHA=parent.INDEX_SHA
LABELS_SHA=parent.LABELS_SHA
NATIVE_SHA=parent.NATIVE_SHA
SPLIT_SHA=parent.SPLIT_SHA
SEEDS=parent.SEEDS
EPOCHS=parent.EPOCHS
STEPS=parent.STEPS
DRAWS=parent.DRAWS
PARAMETERS=parent.PARAMETERS
IDEA_SHA="8ada2e9b5effcd893131caa734510656eda552251c1a0f64a86046904815ae53"
PREREG_SHA="cac4ff343f933707be4535fb3b7925a940394d4710f361187468a3c6466b3907"
MECHANISM_SHA="2b0f6e4b2b2b008d2669b79423203a2f348fee981bc3e50176721d9047d720da"
FAMILY_SHA="481b868c0d694e2c53cccd7f6b0ecaaa6bd623e3a1e56120751769b2e68f06f0"
sha=parent.sha
checked=parent.checked
write_new=parent.write_new
read_rows=parent.read_rows
own_fit=parent.own_fit


def docs():
    parent.docs();checked(RECORD/"IDEA.md",IDEA_SHA);checked(RECORD/"PREREG.md",PREREG_SHA)
    checked(RECORD/"MECHANISM_GATE.json",MECHANISM_SHA);checked(SHARED_RECORD/"VALIDATION_FAMILY_NOTE.md",FAMILY_SHA)
