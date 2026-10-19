import re
import numpy as np
import pandas as pd

# ---------------------------------------------
# COSTANTI E REGOLE DI MATCHING (Regex & Filtri)
# ---------------------------------------------
ID_COL = 'participant_id'
PLACEHOLDER_VALUES = [-9999, -999, -888, -300, 88, 999, 9999]
QC_PREFIX = 'qc_'
TIMING_KEYWORDS = [
    '_timing', 'time', 'click', 'page_submit', 'clickcount', '_time_page'
]

AGGREGATE_MARKERS = [
    '_total', '_score', '_sum', '_ave', '_mean',
    '_med', '_min', '_max', '_perc', '_corr', '_err'
]

FORCE_KEEP = [
    '_subscale', 'poms066', 'poms067', 'poms068', 'poms069', 'poms070',
    'poms071', 'poms072', 'stroopconacc', 'stroopincacc', 'stroopconrt',
    'stroopincrt', 'strooptotrt'
]

FORCE_DROP_ITEMS = [
    'tci_1_5score', 'shipley1_vocab', 'ctqscore_val', 'pos_urgem', 'months', 'comment',
    'pss_month', 'ecr_p', 'rt_7point', 'rp_7point', 'rb_7point', 'non resp',
    'respondent', 'version_form', '_num_items', 'page_submit', 'nonresp',
    'month', 'block', 'trial', 'rt_', 'corr', 'err', 'acc', 'hammer',
    'stroop', 'tmb', 'subjectkey', 'interview_date', 'interview_age',
    'src_subject_id', 'completed', 'shaps_8a', 'dospert_p1_16_perc'
]


def _is_item_column(col: str) -> bool:
    col_lower = col.lower()
    if col in FORCE_KEEP:
        return False
    if col in ['participant_id', 'age', 'sex', 'Site', 'Group', 'rel_to_proband']:
        return False
    if col_lower.startswith(QC_PREFIX):
        return False
    if any(m in col_lower for m in AGGREGATE_MARKERS):
        return False
    if any(fd in col_lower for fd in FORCE_DROP_ITEMS):
        return True

    return bool(
        re.match(r'^[a-z]+[0-9]+$', col_lower)
        or re.match(r'^[a-z]+_[0-9]+$', col_lower)
        or re.match(r'^[a-z]+_q[0-9]+$', col_lower)
        or re.match(r'^[a-z]+[0-9]+[a-z]+$', col_lower)
    )


def esegui_pulizia_globale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    metrics = {'iniziale_subj': df.shape[0], 'iniziale_var': df.shape[1]}

    # 0. SOSTITUZIONE PLACEHOLDER
    df = df.replace(PLACEHOLDER_VALUES, np.nan)

    # ---------------------------------------------
    # FASE 1: QUALITY CONTROL COMPORTAMENTALE (SOGGETTI)
    # ---------------------------------------------
    QC_PAIRS = {
        'dass': [
            ('qc_dass_dep_num_items', 'qc_dass_dep_num_nonresp'),
            ('qc_dass_anx_num_items', 'qc_dass_anx_num_nonresp'),
            ('qc_dass_stress_num_items', 'qc_dass_stress_num_nonresp'),
        ],
        'asi': [('qc_asi_num_items', 'qc_asi_tot_num_nonresp')],
        'pss': [('pss_tot_num_items_x', 'pss_tot_num_nonresp_x')],
        'rsri': [('qc_rsri_tot_num_items', 'qc_rsri_tot_num_nonresp')],
        'shaps': [('qc_shaps_num_items', 'qc_shaps_tot_num_nonresp')],
        'cerq': [('qc_cerq_s_blame_num_items', 'qc_cerq_s_blame_num_nonresp')],
    }
    TEST_ITEMS_TIMING = {
        'asi': ('asi', 'asi_tot_timing_x', 16),
        'bis': ('bis', 'bis_tot_timing', 30),
        'bisbas': ('bisbas', 'bisbas_tot_timing', 24),
        'cfq': ('cogfq', 'cfq_tot_timing', 25),
        'dass': ('dass_', 'dass_tot_timing', 42),
        'rrs': ('rrs_', 'tot_timing', 22),
        'rsri': ('rsri_', 'rsri_tot_timing', 30),
        'shaps': ('shaps', 'shaps_time_Page_Submit', 14),
    }

    flags = pd.DataFrame({ID_COL: df[ID_COL]})

    for test, pairs in QC_PAIRS.items():
        tot_items = np.zeros(len(df))
        tot_nonresp = np.zeros(len(df))
        has_data = False
        for item_col, nonresp_col in pairs:
            if item_col in df.columns and nonresp_col in df.columns:
                tot_items += df[item_col].fillna(0).values
                tot_nonresp += df[nonresp_col].fillna(0).values
                has_data = True
        if has_data:
            pct = np.divide(
                tot_nonresp,
                tot_items,
                out=np.full_like(tot_nonresp, np.nan),
                where=tot_items != 0,
            )
            flags[f'flag_missing_{test}'] = pd.Series(
                pct > 0.30, index=flags.index
            ).astype('Int8')

    for test, (prefix, _, num_items) in TEST_ITEMS_TIMING.items():
        pattern = re.compile(r'.*\d+$')
        item_cols = [
            c
            for c in df.columns
            if c.startswith(prefix)
            and not c.startswith(QC_PREFIX)
            and pattern.match(c)
            and not any(m in c.lower() for m in AGGREGATE_MARKERS)
        ]
        if item_cols:
            valid_count = df[item_cols].notna().sum(axis=1)
            std_dev = df[item_cols].std(axis=1, skipna=True)
            mask = (std_dev == 0) & (valid_count >= int(num_items * 0.80))
            flags[f'flag_sl_{test}'] = mask.astype('Int8')

    for test, (_, timing_col, num_items) in TEST_ITEMS_TIMING.items():
        if timing_col in df.columns:
            threshold_fixed = 0.7 * num_items
            flags[f'flag_speed_{test}'] = (
                df[timing_col] < threshold_fixed
            ).astype('Int8')

    missing_cols = [c for c in flags.columns if c.startswith('flag_missing_')]
    sl_cols = [c for c in flags.columns if c.startswith('flag_sl_')]
    speed_cols = [c for c in flags.columns if c.startswith('flag_speed_')]

    flags['flag_excl_missing'] = (
        flags[missing_cols].max(axis=1).astype('Int8') if missing_cols else 0
    )
    flags['flag_excl_sl'] = (
        flags[sl_cols].max(axis=1).astype('Int8') if sl_cols else 0
    )
    flags['flag_excl_speed'] = (
        flags[speed_cols].max(axis=1).astype('Int8') if speed_cols else 0
    )

    flags['flag_excl_strict'] = (
        (flags['flag_excl_missing'] == 1)
        | ((flags['flag_excl_sl'] == 1) & (flags['flag_excl_speed'] == 1))
    ).astype('Int8')

    df = df.merge(flags, on=ID_COL, how='left')
    df_clean = df[df['flag_excl_strict'] != 1].copy()
    metrics['esclusi_flag_strict'] = int(df.shape[0] - df_clean.shape[0])

    # ---------------------------------------------
    # FASE 2: RIMOZIONE ITEM, TESTI E TIMING (VARIABILI)
    # ---------------------------------------------
    var_pre_regex = df_clean.shape[1]

    cols_to_drop_sys = [
        c
        for c in df_clean.columns
        if str(c).startswith('SD-') or c == 'NOTES' or c.startswith('flag_')
    ]
    df_clean = df_clean.drop(columns=cols_to_drop_sys, errors='ignore')

    items_to_drop = [c for c in df_clean.columns if _is_item_column(c)]
    df_clean = df_clean.drop(columns=items_to_drop, errors='ignore')

    qc_timing_to_drop = [
        c
        for c in df_clean.columns
        if c.lower().startswith(QC_PREFIX)
        or any(kw in c.lower() for kw in TIMING_KEYWORDS)
    ]
    df_clean = df_clean.drop(columns=qc_timing_to_drop, errors='ignore')

    # Conteggio esatto delle colonne del dataset originale rimosse via Regex/Sistema
    # (escludendo dal conteggio le colonne temporanee 'flag_' create in Fase 1)
    metrics['var_rimosse_regex'] = int(
        metrics['iniziale_var'] - df_clean.shape[1]
    )

    # ---------------------------------------------
    # FASE 3: POTATURA STRUTTURALE E CLINICA (VARIABILI)
    # ---------------------------------------------
    STRUCTURAL_REDUNDANCIES = [
        'stroopconrt', 'dospert_p1_16_perc', 'dospert_p1_19_perc',
        'dospert_riskperc_score', 'dospert_risktake_score', 'poms072', 'rsri_tot_avg_score', 'lrtot', 'bissc_total', 'cogfq_total',
        'hammer_tot_num_err', 'hammer_tot_num_corr', 'hammer_tot_per_corr', 'hammer_shapes_per_corr', 'strooptotrt',
        'crt_3_int_score', 'crt_5_int_score', 'crt_3_corr_score', 'vocabtscore', 'iqshp',
        'cfq_distract_score', 'cfq_faltrig_score', 'rt_financial_sum', 'bis_fi', 'bis_fii', 'bis_fiii', 'lrinter', 'refriend',
        'panss_general', 'poms066', 'poms067', 'bapq_total_ave', 'bis_total', 'madrstot', 'mspss_total', 'tci_total', 'pum_tot_sum_score',
        'hammer_shapes_meanRT', 'hammer_shapes_medRT', 'hammer_shapes_minRT', 'hammer_shapes_maxRT', 'hammer_shapes_meanRT_corr', 'hammer_shapes_medRT_corr', 'hammer_shapes_minRT_corr', 'hammer_shapes_maxRT_corr', 'hammer_shapes_meanRT_err', 'hammer_shapes_medRT_err', 'hammer_shapes_minRT_err', 'hammer_shapes_maxRT_err', 'hammer_faces_num_corr', 'hammer_faces_meanRT', 'hammer_faces_medRT', 'hammer_faces_minRT', 'hammer_faces_maxRT', 'hammer_faces_meanRT_corr', 'hammer_faces_medRT_corr', 'hammer_faces_minRT_corr', 'hammer_faces_maxRT_corr', 'hammer_faces_meanRT_err', 'hammer_faces_medRT_err', 'hammer_faces_minRT_err', 'hammer_faces_maxRT_err', 'hammer_faces_num_err', 'hammer_tot_meanRT', 'hammer_tot_medRT', 'hammer_tot_minRT', 'hammer_tot_maxRT', 'hammer_tot_meanRT_corr', 'hammer_tot_medRT_corr', 'hammer_tot_minRT_corr', 'hammer_tot_maxRT_corr', 'hammer_tot_meanRT_err', 'hammer_tot_medRT_err', 'hammer_tot_minRT_err', 'hammer_tot_maxRT_err', 'hammer_shapes_num_corr', 'scwt_omission', 'rt'
    ]

    CLINICAL_ISOLATED = [
        'species', 'elevmood', 'incmotor', 'sexinter', 'sleepamt', 'irritamt', 'speech', 'langdisr', 'content', 'disrupt', 'appearnc', 'insight', 'vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
        'pos_p1', 'pos_p2', 'pos_p3', 'pos_p4', 'pos_p5', 'pos_p6', 'pos_p7', 'neg_n1', 'neg_n2', 'neg_n3', 'neg_n4', 'neg_n5', 'neg_n6', 'neg_n7', 'gps_g1', 'gps_g2', 'gps_g3', 'gps_g4', 'gps_g5', 'gps_g6', 'gps_g7', 'gps_g8', 'gps_g9', 'gps_g10', 'gps_g11', 'gps_g12', 'gps_g13', 'gps_g14', 'gps_g15', 'gps_g16',
        'tci_p2_1', 'tci_p3_10', 'tci_p3_22', 'tci_p3_31', 'tci_p4_15', 'tci_p4_20', 'tci_p4_25', 'tci_p4_27',
        'ecrr_6anx', 'ecrr_7anx', 'ecrr_12anx_r', 'ecrr_13anx', 'ecrr_15anx', 'ecrr_21av_r', 'ecrr_22av', 'ecrr_23av_r', 'ecrr_25av', 'ecrr_26av', 'ecrr_27av_r', 'ecrr_31av_r', 'ecrr_33av_r', 'ecrr_34av_r', 'ecrr_anxiety_score', 'ecrr_avoidance_score', 'ecr_reassurance', 'ecr_turnto', 'ecr_avoidclose', 'ecr_frustrated',
        'q01_first_cig', 'q02_forbidden', 'q03_give_up', 'q04_per_day', 'q05_frequency', 'q06_ill',
        'asc_1_a81f80', 'asc_2_1d1e34'
    ]

    cols_drop_struct = [
        c for c in STRUCTURAL_REDUNDANCIES if c in df_clean.columns
    ]
    df_clean = df_clean.drop(columns=cols_drop_struct)
    metrics['var_rimosse_strutturali'] = len(cols_drop_struct)

    cols_drop_clin = [c for c in CLINICAL_ISOLATED if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_drop_clin)
    metrics['var_rimosse_cliniche'] = len(cols_drop_clin)

    # ---------------------------------------------
    # FASE 4: FILTRO MISSINGNESS VARIABILI (Calcolo Vettoriale Unico)
    # ---------------------------------------------
    missing_ratios = df_clean.isna().mean()
    cols_to_drop_50 = missing_ratios[missing_ratios > 0.50].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop_50)
    metrics['var_rimosse_miss_50'] = len(cols_to_drop_50)

    # ---------------------------------------------
    # FASE 5: FILTRO MISSINGNESS SOGGETTI (Sul Denominatore Depurato)
    # ---------------------------------------------
    soggetti_pre_nan = df_clean.shape[0]
    thresh = 0.3
    df_clean = df_clean.dropna(thresh=int(thresh * df_clean.shape[1]), axis=0)
    metrics['esclusi_troppi_nan'] = int(soggetti_pre_nan - df_clean.shape[0])

    # Codifica rigorosa sesso: M=1.0, F=0.0, O=NaN (preserva integrità demografica)
    if 'sex' in df_clean.columns:
        df_clean['sex'] = df_clean['sex'].replace(
            {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, 'O': np.nan}
        )
        df_clean['sex'] = pd.to_numeric(df_clean['sex'], errors='coerce')

    metrics['finale_subj'] = int(df_clean.shape[0])
    metrics['finale_var'] = int(df_clean.shape[1])

    return df_clean, metrics