import os

import pandas as pd
import numpy as np

dict_all = {
    "parts_as_rows": True,
    "composition_corporelle": {
        "continuous": False,
        "keys": [
            "corps_weight",
            "corps_fat",
            "corps_fat_free",
            "corps_ffmi",
            "corps_right_arm",
            "corps_left_arm",
            "corps_right_leg",
            "corps_left_leg",
        ],
        "operation": {
            "mean": [
                ["corps_leg_mean", "corps_right_leg", "corps_left_leg"],
                ["corps_arm_mean", "corps_right_arm", "corps_left_arm"],
            ]
        },
    },
    "clinical_evaluation": {
        "muscle_volume": {
            "continuous": False,
            "keys": [
                "mesure_cuisse_g",
                "mesure_cuisse_d",
                "cuisse_adipeux_g_mean",
                "cuisse_adipeux_d_mean",
                "mesure_mollet_g",
                "mesure_mollet_d",
                "mollet_adipeux_g_m_mean",
                "mollet_adipeux_d_m_mean",
                "mollet_adipeux_g_l_mean",
                "mollet_adipeux_d_l_mean",
            ],
            "operation": {
                "circumference": [
                    ["cuisse_g_circumference_corrected", "mesure_cuisse_g", "cuisse_adipeux_g_mean"],
                    ["cuisse_d_circumference_corrected", "mesure_cuisse_d", "cuisse_adipeux_d_mean"],
                    [
                        "mollet_g_circumference_corrected",
                        "mesure_mollet_g",
                        "mollet_adipeux_g_m_mean",
                        "mollet_adipeux_g_l_mean",
                    ],
                    [
                        "mollet_d_circumference_corrected",
                        "mesure_mollet_d",
                        "mollet_adipeux_d_m_mean",
                        "mollet_adipeux_d_l_mean",
                    ],
                ],
            },
            "comments": ["circomferences_comments"],
        },
        "motor_score": {
            "continuous": False,
            "keys": ["motor_score_total", "motor_score_g_total", "motor_score_d_total"],
            "operation": None,
            "comments": ["motor_score_comm"],
        },
        "spasticity": {
            "continuous": False,
            "keys": [
                "evalclin_hflech_g",
                "evalclin_gflech_g",
                "evalclin_cflech_dors_g",
                "evalclin_cflech_plan_g",
                "evalclin_cflech_plan_d",
                "evalclin_cflech_dors_d",
                "evalclin_gexten_g",
                "evalclin_gexten_d",
                "evalclin_gflech_d",
                "evalclin_hexten_g",
                "evalclin_hadduc_g",
                "evalclin_habduc_g",
                "evalclin_habduc_d",
                "evalclin_hadduc_d",
                "evalclin_hexten_d",
                "evalclin_hflech_d",
                "clonus_duree_d",
                "clonus_duree_g",
                "clonus_batements_d",
                "clonus_batements_g",
                "clonus_echelle_d",
                "clonus_echelle_g",
            ],
            "operation": {
                "sum": [
                    [
                        "ashworth_total_g",
                        "evalclin_hflech_g",
                        "evalclin_cflech_dors_g",
                        "evalclin_cflech_plan_g",
                        "evalclin_gexten_g",
                        "evalclin_gflech_g",
                        "evalclin_hexten_g",
                        "evalclin_hadduc_g",
                        "evalclin_habduc_g",
                    ],
                    [
                        "ashworth_total_d",
                        "evalclin_gflech_d",
                        "evalclin_cflech_dors_d",
                        "evalclin_cflech_plan_d",
                        "evalclin_gexten_d",
                        "evalclin_hexten_d",
                        "evalclin_hadduc_d",
                        "evalclin_hflech_d",
                        "evalclin_habduc_d"
                    ],
                    ["arshworth_total", "ashworth_total_g", "ashworth_total_d"],
                ]
            },
            "comments": ["ashworth_comments", "clonus_comments"],
        },
        "dermatomes": {
            "continuous": False,
            "keys": [
                "derm_touch_g_l1",
                "derm_touch_g_l2",
                "derm_touch_g_l3",
                "derm_touch_g_l4",
                "derm_touch_g_l5",
                "derm_touch_g_s1",
                "derm_touch_g_s2",
                "derm_piq_g_l1",
                "derm_piq_g_l2",
                "derm_piq_g_l3",
                "derm_piq_g_l4",
                "derm_piq_g_l5",
                "derm_piq_g_s1",
                "derm_piq_g_s2",
                "derm_touch_d_l1",
                "derm_touch_d_l2",
                "derm_touch_d_l3",
                "derm_touch_d_l4",
                "derm_touch_d_l5",
                "derm_touch_d_s1",
                "derm_touch_d_s2",
                "derm_piq_d_l1",
                "derm_piq_d_l2",
                "derm_piq_d_l3",
                "derm_piq_d_l4",
                "derm_piq_d_l5",
                "derm_piq_d_s1",
                "derm_piq_d_s2",
            ],
            "operation": {
                "sum": [
                    [
                        "derm_touch_g_total",
                        "derm_touch_g_l1",
                        "derm_touch_g_l2",
                        "derm_touch_g_l3",
                        "derm_touch_g_l4",
                        "derm_touch_g_l5",
                        "derm_touch_g_s1",
                        "derm_touch_g_s2",
                    ],
                    [
                        "derm_touch_d_total",
                        "derm_touch_d_l1",
                        "derm_touch_d_l2",
                        "derm_touch_d_l3",
                        "derm_touch_d_l4",
                        "derm_touch_d_l5",
                        "derm_touch_d_s1",
                        "derm_touch_d_s2",
                    ],
                    [
                        "derm_touch_total",
                        "derm_touch_g_total",
                        "derm_touch_d_total",
                    ],
                    [
                        "derm_piq_g_total",
                        "derm_piq_g_l1",
                        "derm_piq_g_l2",
                        "derm_piq_g_l3",
                        "derm_piq_g_l4",
                        "derm_piq_g_l5",
                        "derm_piq_g_s1",
                        "derm_piq_g_s2",
                    ],
                    [
                        "derm_piq_d_total",
                        "derm_piq_d_l1",
                        "derm_piq_d_l2",
                        "derm_piq_d_l3",
                        "derm_piq_d_l4",
                        "derm_piq_d_l5",
                        "derm_piq_d_s1",
                        "derm_piq_d_s2",
                    ],
                    [
                        "derm_piq_total",
                        "derm_piq_g_total",
                        "derm_piq_d_total",
                    ],
                    ["derm_total", "derm_touch_total", "derm_piq_total"],
                ]
            },
            "comments": ["dermatomes_comments"],
        },
        "amplitudes": {
            "continuous": False,
            "keys": [
                "flex_plantaire_gauche",
                "flex_plantaire_droite",
                "flex_dorsale_gauche",
                "flex_dorsale_droite",
                "genou_flex_gauche",
                "genou_flex_droite",
                "genou_ext_gauche",
                "genou_ext_droite",
                "hanche_flex_gauche",
                "hanche_flex_droite",
            ],
            "operation": None,
            "comments": ["amplitude_comments"],
        },
    },
    "in_training": {
        "continuous": True,
        "redcap_repeat_instrument_to_ignore": ['temps'],
        "keys": [
            "borg_douleur",
            "borg_fatigue",
            "puissance_mean",
            "temps_actif",
            "temps_passif",
            "resistance_mean",
            "asymetrie_mean",
            "stim_moyenne",
            "oxymetre_spo2",
            "oxymetre_bpm",
        ],
        "comments": ["entrainement_comments", "comments"],
        "operation": {
            "sd": [
                ["bpm_sd", "oxymetre_bpm"],
                ["spo2_sd", "oxymetre_spo2"],
            ],
            "range": [
                ["bpm_range", "oxymetre_bpm"],
                ["spo2_range", "oxymetre_spo2"],
            ],
            "mean": [
                ["bpm_mean", "oxymetre_bpm"],
                ["spo2_mean", "oxymetre_spo2"],
            ],
            "max": [
                ["bpm_max", "oxymetre_bpm"],
                ["spo2_max", "oxymetre_spo2"],
            ],
            "min": [
                ["bpm_min", "oxymetre_bpm"],
                ["spo2_min", "oxymetre_spo2"],
            ],
        },
    },
}

common_keys = ["record_id", "with_fes", "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance"]


def circumference(circum, adipeux):
    return np.round(circum.values - (adipeux.values * 1e-3 * np.pi), 2)


def create_csv(data, dict_all, part_as_rows=True, parent_key="", extracted_dir=None):
    pd_total = pd.DataFrame()
    for key, value in dict_all.items():
        pd_global = pd.DataFrame()
        if "keys" not in value:
            pd_global = create_csv(data, value, part_as_rows, key, extracted_dir)
        else:
            keys = value["keys"]
            continuous = value["continuous"] if "continuous" in value else False
            comments = value["comments"] if "comments" in value else None
            repeat_instruments = value["redcap_repeat_instrument_to_ignore"] if "redcap_repeat_instrument_to_ignore" in value else None
            if repeat_instruments is not None:
                data = data.loc[~data["redcap_repeat_instrument"].isin(repeat_instruments)]
            if comments is not None:
                keys += comments
            if continuous is False:
                pd_tmp = data.loc[~data["redcap_event_name"].str.contains("session"), common_keys + keys]
            else:
                pd_tmp = data.loc[data["redcap_event_name"].str.contains("session"), common_keys + keys]
            operation = value["operation"] if "operation" in value else None
            if operation is not None:
                for op_name in value["operation"]:
                    for new_key, *old_keys in value["operation"][op_name]:
                        if op_name == "mean":
                            pd_tmp[new_key] = pd_tmp[old_keys].mean(axis=1)
                        elif op_name == "sum":
                            pd_tmp[new_key] = pd_tmp[old_keys].sum(axis=1)
                        elif op_name == "sd":
                            pd_tmp[new_key] = pd_tmp[old_keys].std(axis=1)
                        elif op_name == "range":
                            pd_tmp[new_key] = pd_tmp[old_keys].max(axis=1) - pd_tmp[old_keys].min(axis=1)
                        elif op_name == "max":
                            pd_tmp[new_key] = pd_tmp[old_keys].max(axis=1)
                        elif op_name == "min":
                            pd_tmp[new_key] = pd_tmp[old_keys].min(axis=1)
                        elif op_name == "circumference":
                            pd_tmp[new_key] = circumference(pd_tmp[old_keys[0]], pd_tmp[old_keys[1:]].mean(axis=1))
                        else:
                            raise ValueError(f"Unknown operation {op_name}")
            pd_tmp = pd_tmp.round(2)
            name = key if parent_key == '' else f"{parent_key}_{key}"
            if not part_as_rows:
                pd_tmp = pd_tmp.T
                pd_tmp.columns = pd_tmp.iloc[0]
                pd_tmp = pd_tmp.iloc[1:]
                pd_tmp.to_csv(os.path.join(extracted_dir, f"{name}.csv"), index=False)
            else:
                pd_tmp.to_csv(os.path.join(extracted_dir, f"{name}.csv"), index=False)
            pd_total = pd.merge(pd_total, pd_tmp, on=common_keys, how='outer') if not pd_total.empty else pd_tmp
        if not pd_global.empty:
            pd_global.to_csv(os.path.join(extracted_dir, f"{key}.csv"), index=False)
    return pd_total


if __name__ == "__main__":
    data_path = r"D:\Downloads\CIMEFESBike_DATA_2026-06-04_1146.csv"
    extracted_dir = data_path.replace(".csv", "_extracted")
    os.makedirs(extracted_dir, exist_ok=True)

    data = pd.read_csv(data_path, sep="\t")
    part_as_rows = dict_all.pop("parts_as_rows")
    part_fes = data.loc[data["groupe"] == 1, "record_id"]
    part_no_stim = data.loc[data["groupe"] == 2, "record_id"]

    data.loc[:, "with_fes"] = data["record_id"].apply(
        lambda x: "True" if x in part_fes.values else ("False" if x in part_no_stim.values else "Other")
    )
    data = data.loc[(data["redcap_repeat_instrument"] != "sance_annul") & (data["with_fes"] != "Other")]

    create_csv(data, dict_all, part_as_rows, parent_key="", extracted_dir=extracted_dir)

