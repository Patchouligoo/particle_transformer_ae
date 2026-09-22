"""
Define the location of processes and their labels.

process_id  one integer per process (backgrounds > 0, signals < 0)
group_id    contrastive class, grouped like haxad (src/embedding_models/contrastive_encoder/process_groups.py):
            each SM background is its own group (>= 0), the mass points of one signal family share a group (< 0):
            0 yyjets, 1 ggH, 2 VBF, 3 VH, 4 ttH, -1 WN_HyyN, -2 Hl_Hyyl, -3 TT_tZNtHyyN, -4 HVT_VcXjjHyy
"""
import os, sys

# LOCAL_BASE_DIR is exported by setup.sh; all processes come from the same Delphes production
base_dir = os.path.join(os.environ["LOCAL_BASE_DIR"], "version_prod_12_all", "SkimEvents")

process_dict = {
    # sm sample
    "nonres_yy_jjj": {
        "file": f"{base_dir}/nonres_yy_jjj/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 1,
        "group_id": 0,
    },
    "ggh_yy": {
        "file": f"{base_dir}/ggh_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 2,
        "group_id": 1,
    },
    "vbf_yy": {
        "file": f"{base_dir}/vbf_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 3,
        "group_id": 2,
    },
    "vh_yy": {
        "file": f"{base_dir}/vh_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 4,
        "group_id": 3,
    },
    "ttH_yy": {
        "file": f"{base_dir}/ttH_yy/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": 5,
        "group_id": 4,
    },
    # bsm sample
    "WN_HyyN_150": {
        "file": f"{base_dir}/WN_HyyN_150/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -1,
        "group_id": -1,
    },
    "WN_HyyN_200": {
        "file": f"{base_dir}/WN_HyyN_200/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -2,
        "group_id": -1,
    },
    "WN_HyyN_300": {
        "file": f"{base_dir}/WN_HyyN_300/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -3,
        "group_id": -1,
    },
    "WN_HyyN_600": {
        "file": f"{base_dir}/WN_HyyN_600/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -4,
        "group_id": -1,
    },
    "Hl_Hyyl_150": {
        "file": f"{base_dir}/Hl_Hyyl_150/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -5,
        "group_id": -2,
    },
    "Hl_Hyyl_300": {
        "file": f"{base_dir}/Hl_Hyyl_300/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -6,
        "group_id": -2,
    },
    "Hl_Hyyl_450": {
        "file": f"{base_dir}/Hl_Hyyl_450/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -7,
        "group_id": -2,
    },
    "TT_tZNtHyyN_500_180_50": {
        "file": f"{base_dir}/TT_tZNtHyyN_500_180_50/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -8,
        "group_id": -3,
    },
    "TT_tZNtHyyN_1000_205_60": {
        "file": f"{base_dir}/TT_tZNtHyyN_1000_205_60/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -9,
        "group_id": -3,
    },
    "TT_tZNtHyyN_1200_205_60": {
        "file": f"{base_dir}/TT_tZNtHyyN_1200_205_60/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -10,
        "group_id": -3,
    },
    "HVT_VcXjjHyy_500_10": {
        "file": f"{base_dir}/HVT_VcXjjHyy_500_10/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -11,
        "group_id": -4,
    },
    "HVT_VcXjjHyy_500_300": {
        "file": f"{base_dir}/HVT_VcXjjHyy_500_300/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -12,
        "group_id": -4,
    },
    "HVT_VcXjjHyy_2000_300": {
        "file": f"{base_dir}/HVT_VcXjjHyy_2000_300/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -13,
        "group_id": -4,
    },
    "HVT_VcXjjHyy_2000_1000": {
        "file": f"{base_dir}/HVT_VcXjjHyy_2000_1000/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -14,
        "group_id": -4,
    },
    "HVT_VcXjjHyy_2000_1700": {
        "file": f"{base_dir}/HVT_VcXjjHyy_2000_1700/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -15,
        "group_id": -4,
    },
    "WlZvHv_Hyyl_200": {
        "file": f"{base_dir}/WlZvHv_Hyyl_200/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -16,
        "group_id": -5,
    },
    "WlZvHv_Hyyl_400": {
        "file": f"{base_dir}/WlZvHv_Hyyl_400/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -17,
        "group_id": -5,
    },
    "WlZvHv_Hyyl_600": {
        "file": f"{base_dir}/WlZvHv_Hyyl_600/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -18,
        "group_id": -5,
    },
    "XHH_300": {
        "file": f"{base_dir}/XHH_300/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -19,
        "group_id": -6,
    },
    "XHH_500": {
        "file": f"{base_dir}/XHH_500/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -20,
        "group_id": -6,
    },
    "XHH_1000": {
        "file": f"{base_dir}/XHH_1000/ecm_13000.00/ATLAS_fatjet_skimAll/fullmc/skimmed.h5",
        "process_id": -21,
        "group_id": -6,
    },
}
