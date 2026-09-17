"""
Define the location of processes
"""
import os, sys

# LOCAL_BASE_DIR is exported by setup.sh; all processes come from the same Delphes production
base_dir = os.path.join(os.environ["LOCAL_BASE_DIR"], "version_prod_12_all", "SkimEvents")

process_dict = {
    # sm sample
    "nonres_yy_jjj": {
        "file": f"{base_dir}/nonres_yy_jjj/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 1,
    },
    "ggh_yy": {
        "file": f"{base_dir}/ggh_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 2,
    },
    "vbf_yy": {
        "file": f"{base_dir}/vbf_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 3,
    },
    "vh_yy": {
        "file": f"{base_dir}/vh_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 4,
    },
    "ttH_yy": {
        "file": f"{base_dir}/ttH_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 5,
    },
    # bsm sample
    "WN_HyyN_150": {
        "file": f"{base_dir}/WN_HyyN_150/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -1,
    },
    "WN_HyyN_200": {
        "file": f"{base_dir}/WN_HyyN_200/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -2,
    },
    "WN_HyyN_300": {
        "file": f"{base_dir}/WN_HyyN_300/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -3,
    },
    "WN_HyyN_600": {
        "file": f"{base_dir}/WN_HyyN_600/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -4,
    },
}
