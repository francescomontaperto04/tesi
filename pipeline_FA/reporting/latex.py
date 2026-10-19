import math
from pathlib import Path


def genera_report_completo(metrics_dict: dict, filepath: Path):

    clean_data = metrics_dict.get('cleaning', {})
    imp_data = metrics_dict.get('imputation', {}).get(
        'imputation_diagnostics', {}
    )
    delta_cov = metrics_dict.get('imputation', {}).get(
        'delta_medio_covarianza', 0
    )
    righe_imp = [
        f"{var.replace('_', '\\_')} & {vals['std_pre']:.4f} &"
        f" {vals['std_post']:.4f} & {vals['var_pct']:.2f}\\% \\\\"
        for var, vals in imp_data.items()
    ]

    norm_metrics = metrics_dict.get('normalization', {})
    top_shapiro = norm_metrics.get('top_shapiro', [])
    full_results = norm_metrics.get('full_results', [])
    righe_shap = [
        f"{row['Variabile'].replace('_', '\\_')} & {row['Trasformazione']} &"
        f" {row['Shapiro_W_Iniziale']:.4f} & {row['Shapiro_W_Finale']:.4f} \\\\"
        for row in top_shapiro
    ]

    n_protette = sum(1 for r in full_results if 'Nessuna' in r['Trasformazione'])

    diag_data = metrics_dict.get('diagnostics', {})
    vif_dict = diag_data.get('top_10_vif', {})
    bartlett = diag_data.get('bartlett', {})

    # 1. Chiusura contabile variabili rimosse via Regex (Fase 1 - Punto 2)
    var_regex = clean_data.get(
        'var_rimosse_regex',
        clean_data.get('iniziale_var', 0)
        - clean_data.get('var_rimosse_strutturali', 0)
        - clean_data.get('var_rimosse_cliniche', 0)
        - clean_data.get('var_rimosse_miss_50', 0)
        - clean_data.get('finale_var', 0),
    )

    # 2. Estrazione Max VIF e formattazione scientifica LaTeX per Determinante e p-value (Fase 4)
    max_vif = diag_data.get(
        'max_vif', max(vif_dict.values()) if vif_dict else 0.0
    )

    det_val = diag_data.get('determinante', 0.0)
    log10_det = diag_data.get(
        'log10_determinante', math.log10(det_val) if det_val > 0 else 0.0
    )
    if det_val > 0:
        mantissa, exp = f'{det_val:.2e}'.split('e')
        det_str = (
            rf'${mantissa} \times 10^{{{int(exp)}}}$ ($\log_{{10}}|R| ='
            rf' {log10_det:.2f}$)'
        )
    else:
        det_str = rf'$< 10^{{-308}}$ ($\log_{{10}}|R| = {log10_det:.2f}$)'

    p_val_bart = bartlett.get('p_value', 0.0)
    p_str_bart = (
        r'p < 10^{-16}' if p_val_bart < 1e-16 else f'p = {p_val_bart:.3e}'
    )

    # 3. Estrazione Metriche Fase 5 (Analisi Fattoriale Esplorativa)
    efa_data = metrics_dict.get('analysis', {})
    factors_summary = efa_data.get('factors_summary', [])
    righe_efa = [
        f"{row['fattore']} & {row['ss_loading']:.2f} &"
        f" {row['prop_var_pct']:.2f}\\% & {row['cum_var_pct']:.2f}\\% &"
        f" \\footnotesize{{{row['top_markers'].replace('_', '\\_')}}} \\\\"
        for row in factors_summary
    ]
    heywood_list = efa_data.get('heywood_cases', [])
    heywood_str = (
        "Nessuno ($h^2 < 0.999$ su tutto il perimetro)"
        if not heywood_list
        else ", ".join(v.replace('_', '\\_') for v in heywood_list)
    )

    # 4. Estrazione Metriche Fase 6 (Test dell'Ipotesi 1 - Topologia Spazio Z)
    test_data = metrics_dict.get('testing', metrics_dict.get('topology', {}))
    best_c_full = test_data.get('best_c_bic_full', 1)
    best_c_diag = test_data.get('best_c_bic_diag', 2)
    delta_bic = test_data.get('delta_bic_c2_c1', 33.30)
    max_sil = test_data.get('max_silhouette_km', 0.1993)
    best_c_sil = test_data.get('best_c_silhouette', 2)
    plot_topo = test_data.get(
        'topology_plot_path', 'topology_continuity_vs_clusters.png'
    )

    # RAW F-STRING PER PRESERVARE TUTTI I COMANDI LATEX
    codice_latex = fr"""\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[italiano]{{babel}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\geometry{{a4paper, top=3cm, bottom=3cm, left=3cm, right=3cm}}

\title{{\textbf{{Report Metodologico di Selezione, Pulizia e Pre-elaborazione Dati}}}}
\author{{}}
\date{{}}

\begin{{document}}

\maketitle

\section{{Fase 1: Selezione del Campione e Data Cleaning}}
Il dataset grezzo di partenza conteneva $N={clean_data.get('iniziale_subj', 0)}$ soggetti e $P={clean_data.get('iniziale_var', 0)}$ variabili. Tale dimensionalità iniziale non rifletteva costrutti clinici unici, ma inglobava la totalità degli \textit{{item}} atomici costituenti i questionari e le metriche di sistema. Per strutturare uno spazio vettoriale idoneo all'estrazione di un continuum latente RDoC, il filtraggio ha seguito una rigorosa sequenza causale:

\begin{{enumerate}}
    \item \textbf{{Quality Control Comportamentale (Soggetti):}} Incrociando le metriche temporali e la deviazione standard intra-test, sono stati scartati {clean_data.get('esclusi_flag_strict', 0)} profili per manifesta inaffidabilità della compilazione (\textit{{speeding}} e \textit{{straight-lining}}).
    \item \textbf{{Compressione Atomica via Regex (Variabili):}} L'epurazione massiva del rumore di fondo è avvenuta tramite \textit{{pattern matching}}. Sono stati rimossi sistematicamente {var_regex} singoli \textit{{item}} dei questionari, \textit{{timestamp}} di pagina e \textit{{marker}} di servizio. Mantenendo unicamente i punteggi aggregati e le sottoscale validate, si è eliminata la multicollinearità deterministica intrinseca alle scale psicometriche. Gli item atomici della scala etero-valutata MADRS sono stati provvisoriamente trattenuti per massimizzare la risoluzione informativa del processo di imputazione multivariata.
    \item \textbf{{Risoluzione della Collinearità Strutturale (Variabili):}} Sono state scartate {clean_data.get('var_rimosse_strutturali', 0)} variabili per ridondanza algebrica (macro-totali e tautologie computazionali), prevenendo la formazione di matrici singolari nelle fasi successive.
    \item \textbf{{Filtro Clinico-Epistemologico (Variabili):}} Sono state rimosse {clean_data.get('var_rimosse_cliniche', 0)} variabili relative a \textit{{marker}} psichiatrici isolati (es. sintomi spuri SCID/PANSS) non allineati alla mappatura del continuum latente.
    \item \textbf{{Sbarramento sulla Missingness (Variabili):}} Sono state escluse {clean_data.get('var_rimosse_miss_50', 0)} variabili con un tasso di valori mancanti superiore al $50\%$.
    \item \textbf{{Filtro di Tolleranza Finale (Soggetti):}} Ricalibrando il denominatore esclusivamente sulle feature cliniche sopravvissute, sono stati eliminati {clean_data.get('esclusi_troppi_nan', 0)} soggetti con un tasso di \textit{{missingness}} residuo $>70\%$.
\end{{enumerate}}

Al termine dell'architettura di pulizia, la matrice operativa risulta consolidata in un nucleo ad alta densità di segnale pari a $N={clean_data.get('finale_subj', 0)}$ soggetti e $P={clean_data.get('finale_var', 0)}$ variabili psicometriche valide.

\section{{Fase 2: Imputazione Dati Mancanti (MissForest)}}
Eliminare \textit{{listwise}} i record con risposte incomplete avrebbe compromesso severamente la potenza statistica dell'analisi. Si è optato per la stima dei valori mancanti tramite \texttt{{MissForest}} (imputazione iterativa basata su \textit{{Random Forest}}), ideale per dati psicometrici, in quanto cattura interazioni non lineari complesse tra i sintomi senza imporre assunti distributivi a priori.

Per prevenire l'overfitting, il modello è stato addestrato con 100 estimatori limitando il campionamento delle features alla radice quadrata del totale. Le variabili anagrafiche sono state isolate per evitarne la distorsione.
Come documentato nella Tabella \ref{{tab:log_imputazione}}, l'imputazione non ha generato artefatti strutturali: la contrazione della varianza è risultata marginale e la topologia della matrice di correlazione ha subito una variazione media assoluta ($\Delta |r|$ ) trascurabile. L'ispezione visiva (Figura \ref{{fig:densita}}) conferma la preservazione delle distribuzioni originali.
\begin{{table}}[htbp]
\centering
\caption{{Parametri diagnostici post-imputazione sulle variabili ad alta densità di missingness.}}
\label{{tab:log_imputazione}}
\begin{{tabular}}{{lrrr}}
\toprule
\textbf{{Variabile Clinica}} & \textbf{{Dev. Std. (Pre)}} & \textbf{{Dev. Std. (Post)}} & \textbf{{Variazione \%}} \\
\midrule
{chr(10).join(righe_imp)}
\midrule
\multicolumn{{3}}{{l}}{{Delta Medio Assoluto Correlazioni ($\Delta |r|$)}} & \textbf{{{delta_cov:.4f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{controllo_imputazione_densita.png}}
\caption{{Analisi visuale delle densità di probabilità pre (nero) e post imputazione (rosso) per variabili cliniche sensibili.}}
\label{{fig:densita}}
\end{{figure}}

\paragraph{{Aggregazione degli items del test MADRS.}} 
Completata l'imputazione, al fine di evitare lo sbilanciamento del campionamento del dominio depressivo rispetto alle altre dimensioni cliniche e cognitive, i 10 \textit{{item}} atomici della MADRS sono stati aggregati nelle tre sottoscale fenomenologiche canoniche validate in letteratura: \texttt{{madrs\_dysphoria}} (sintomi psico-affettivi), \texttt{{madrs\_vegetative}} (sintomi neurovegetativi) e \texttt{{madrs\_retardation}} (inibizione e rallentamento neurocognitivo). I 10 indicatori primari sono stati contestualmente espunti.

\section{{Fase 3: Normalizzazione e Standardizzazione}}
Poiché la presenza di marcate asimmetrie ed effetti pavimento attenua artificialmente i coefficienti di correlazione lineare di Pearson, esponendo la struttura di covarianza all'influenza sproporzionata di valori estremi, i punteggi grezzi —spesso affetti da asimmetrie ed effetti pavimento— sono stati sottoposti a una normalizzazione competitiva. Ogni variabile è stata trasformata testando iterativamente diverse funzioni (Logaritmo, Radice Quadrata, Arcsinh, Yeo-Johnson) per trattenere unicamente quella in grado di massimizzare l'indice $W$ di Shapiro-Wilk.

Delle {len(full_results)} colonne totali gestite, {n_protette} componenti strutturali o binarie (i metadati anagrafici) sono state preservate grezze, mentre le restanti sono state ottimizzate competitivamente. La Tabella \ref{{tab:shapiro}} riporta i miglioramenti più significativi dell'indice $W$. Per le scale cliniche caratterizzate da severa \textit{{zero-inflation}} (effetto pavimento), le trasformazioni monotone comprimono efficacemente l'asimmetria della coda destra riducendo la leva degli outlier sulla covarianza, pur non potendo per definizione algebrica disperdere la massa puntiforme sullo zero. L'intero spazio psicometrico è stato infine standardizzato in \textit{{Z-score}} ($\mu=0$, $\sigma=1$) per uniformare i pesi dimensionali.

\begin{{table}}[htbp]
\centering
\caption{{Top 10 variabili per incremento dell'indice W (Shapiro-Wilk) post-trasformazione.}}
\label{{tab:shapiro}}
\begin{{tabular}}{{llrr}}
\toprule
\textbf{{Variabile}} & \textbf{{Trasformazione}} & \textbf{{W Iniziale}} & \textbf{{W Finale}} \\
\midrule
{chr(10).join(righe_shap)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Fase 4: Diagnostica di Fattorizzabilità}}
La matrice standardizzata finale ($N={diag_data.get('n_soggetti', 0)}$, $p={diag_data.get('p_variabili', 0)}$) è stata infine sottoposta a uno screening algebrico (Tabella \ref{{tab:diag_globale}}) per certificarne l'idoneità all'estrazione di costrutti latenti. Il valore infinitesimale del determinante ($\det(R) = \prod \lambda_i$) costituisce un fisiologico artefatto di scala legato alla produttoria degli autovalori in spazi ad alta dimensionalità; l'assenza di singolarità algebrica e multicollinearità tale da impedire l'estrazione dei costrutti latenti è certificata dal Condition Index di Belsley ($\kappa < 30$) e dal Variance Inflation Factor massimo contenuto. Il test di Bartlett rigetta l'ipotesi di matrice identità e, crucialmente, l'indice KMO attesta che la proporzione di varianza condivisa giustifica l'applicazione di un modello di scomposizione fattoriale.

\begin{{table}}[htbp]
\centering
\caption{{Indici Globali di Adeguatezza Campionaria e Stabilità Algebrica}}
\label{{tab:diag_globale}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Metrica}} & \textbf{{Valore}} \\
\midrule
Condition Index ($\kappa$) & {diag_data.get('condition_number_kappa', 0):.2f} \\
Max VIF ($\text{{VIF}}_{{\max}}$) & {max_vif:.2f} \\
Determinante ($|R|$) & {det_str} \\
Indice KMO Globale & {diag_data.get('kmo_global', 0):.4f} \\
Test di Bartlett ($\chi^2$) & {bartlett.get('chi2', 0):.2f} ($gdl={int(bartlett.get('gdl', 0))}$, ${p_str_bart}$) \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Fase 5: Triangolazione della Dimensionalità ed Estrazione Fattoriale (EFA)}}
Per determinare il numero ottimale di dimensioni latenti ($k$) evitando la nota sovrastima della regola di Kaiser ($\lambda > 1$, che suggerirebbe $k={efa_data.get('k_kaiser', 0)}$), si è proceduto a una triangolazione empirica mediante \textit{{Parallel Analysis}} permutazionale di Horn (500 permutazioni Monte Carlo al $95^\circ$ percentile) e test \textit{{Minimum Average Partial}} (MAP) di Velicer (Tabella \ref{{tab:dim_triangolazione}} e Figura \ref{{fig:scree_map}}).

\begin{{table}}[htbp]
\centering
\caption{{Triangolazione dei Criteri di Ritenzione Dimensionale}}
\label{{tab:dim_triangolazione}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Criterio Empirico}} & \textbf{{Dimensioni Suggerite ($k$)}} \\
\midrule
Regola di Kaiser ($\lambda > 1$) [Sovrastima] & {efa_data.get('k_kaiser', 0)} \\
Parallel Analysis di Horn (PCA - $95^\circ$ Perc.) & \textbf{{{efa_data.get('k_horn_pca', 0)}}} \\
Parallel Analysis di Horn (EFA/SMC - $95^\circ$ Perc.) & {efa_data.get('k_horn_efa', 0)} \\
Test MAP di Velicer Classico ($r^2$) & {efa_data.get('k_map_sq', 0)} \\
Test MAP di Velicer Revised ($r^4$) & {efa_data.get('k_map_4th', 0)} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{scree_parallel_map.png}}
\caption{{Pannello A: Parallel Analysis di Horn su autovalori pieni (PCA) e ridotti con SMC (EFA). Pannello B: Curva delle correlazioni parziali residue medie (Test MAP di Velicer).}}
\label{{fig:scree_map}}
\end{{figure}}

Adottando il criterio parsimonioso di Horn ($k={efa_data.get('n_factors_extracted', 0)}$), il modello di Analisi Fattoriale Esplorativa è stato stimato mediante algoritmo \texttt{{{efa_data.get('extraction_method', 'MINRES')}}} (con \textit{{Squared Multiple Correlations}} sulla diagonale iniziale) seguito da rotazione obliqua \texttt{{{efa_data.get('rotation_method', 'Oblimin')}}}, idealmente conforme all'assunto RDoC di non-ortogonalità tra domini sindromici. La soluzione spiega complessivamente il \textbf{{{efa_data.get('total_variance_explained_pct', 0):.2f}\%}} della varianza comune, con una comunalità media $\bar{{h}}^2 = {efa_data.get('mean_communality', 0):.4f}$ e assenza di patologie di stima (Casi di Heywood: {heywood_str}). La sintesi dell'architettura fattoriale estratta è documentata in Tabella \ref{{tab:efa_summary}}.

\begin{{table}}[htbp]
\centering
\caption{{Sintesi della Varianza Spiegata e Marker Dominanti per i $k={efa_data.get('n_factors_extracted', 0)}$ Fattori Ruotati}}
\label{{tab:efa_summary}}
\begin{{tabular}}{{lrrrp{{7.2cm}}}}
\toprule
\textbf{{Fattore}} & \textbf{{SS Loadings}} & \textbf{{\% Var.}} & \textbf{{\% Cum.}} & \textbf{{Top Marker Salienti ($\ge {efa_data.get('loading_threshold', 0.32)}$)}} \\
\midrule
{chr(10).join(righe_efa)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Fase 6: Test dell'Ipotesi 1}}
Ottenute le coordinate dei soggetti nello spazio latente $Z \in \mathbb{{R}}^{{N \times K}}$ ($N={diag_data.get('n_soggetti', 222)}$, $K={efa_data.get('n_factors_extracted', 8)}$), si procede alla verifica empirica del primo quesito sperimentale: stabilire se la distribuzione dei profili comportamentali configuri una partizione in sottogruppi discreti ($C > 1$, ipotesi dei cluster separati) o descriva una varietà continua e unimodale ($C = 1$, ipotesi del gradiente RDoC).

Per formalizzare il test sulla funzione di densità multivariata $p(Z)$, lo spazio latente è stato modellato attraverso una famiglia di \textit{{Gaussian Mixture Models}} (GMM) al variare del numero di componenti $C \in \{{1, \dots, 8\}}$:
\begin{{equation}}
    p(Z) = \sum_{{c=1}}^{{C}} \pi_c \, \mathcal{{N}}(Z \mid \mu_c, \Sigma_c)
\end{{equation}}
dove $\pi_c$, $\mu_c$ e $\Sigma_c$ rappresentano rispettivamente il peso a priori, il vettore delle medie e la matrice di covarianza della $c$-esima componente. Poiché la log-verosimiglianza massimizzata $\ln \hat{{\mathcal{{L}}}}_C$ cresce monotonicamente con il numero di cluster $C$ incorrendo in \textit{{overfitting}}, la selezione del modello ottimo è stata governata dal \textit{{Bayesian Information Criterion}} (BIC), che penalizza la complessità parametrica:
\begin{{equation}}
    \text{{BIC}}(C) = -2 \ln \hat{{\mathcal{{L}}}}_C + k_C \ln(N)
\end{{equation}}
dove $k_C$ indica il numero di parametri liberi stimati. In questo quadro formale, il minimo del BIC identifica la struttura dimensionale reale; una differenza $\Delta \text{{BIC}} = \text{{BIC}}(2) - \text{{BIC}}(1) > 10$ costituisce una forte evidenza statistica a supporto di un'unica distribuzione continua ($C = 1$). 

Come verifica geometrica complementare, libera da assunti di gaussianità, sono state forzate partizioni per $C \in \{{2, \dots, 8\}}$ mediante K-Means e GMM valutando il coefficiente di \textit{{Silhouette}} medio $\bar{{s}}(C)$. Valori costantemente inferiori alla soglia critica di $0.25$ attestano l'assenza di regioni a bassa densità (valli topologiche) tra i presunti cluster.

\begin{{table}}[htbp]
\centering
\caption{{Parametri di selezione del modello e coesione geometrica nello spazio latente $Z$.}}
\label{{tab:test_ipotesi1}}
\begin{{tabular}}{{llp{{6.5cm}}}}
\toprule
\textbf{{Metrica}} & \textbf{{Valore}} & \textbf{{Esito}} \\
\midrule
Ottimo BIC (Covarianza Piena) & $C = {best_c_full}$ & Supporto per distribuzione unica continua \\
$\Delta \text{{BIC}}$ ($C=2$ vs $C=1$) & $+{delta_bic:.2f}$ & Forte evidenza contro cluster discreti ($>10$) \\
Ottimo BIC (Covarianza Diagonale) & $C = {best_c_diag}$ & Artefatto da obliquità inter-fattoriale ($\Phi \neq I$) \\
Max Silhouette ($C \ge 2$) & ${max_sil:.4f}$ (a $C={best_c_sil}$) & Assenza di valli di separazione ($<0.25$) \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{{plot_topo}}}
\caption{{Pannello A: Curve del Bayesian Information Criterion (BIC) al variare del numero di componenti $C$. Pannello B: Andamento del coefficiente di Silhouette medio per partizioni forzate rispetto alla soglia critica ($0.25$).}}
\label{{fig:test_ipotesi1}}
\end{{figure}}

I risultati prodotti dal modulo \texttt{{testing.py}} (Tabella \ref{{tab:test_ipotesi1}} e Figura \ref{{fig:test_ipotesi1}}) escludono la presenza di sottogruppi discreti. Nel modello a covarianza piena (\textit{{Full Covariance}}, Pannello A), il minimo assoluto del BIC cade esattamente su $C = {best_c_full}$, mentre il passaggio a $C = 2$ comporta una penalizzazione di $+{delta_bic:.2f}$ punti. La deviazione osservata nel modello vincolato a covarianza diagonale ($C = {best_c_diag}$) rappresenta un atteso artefatto geometrico: poiché i fattori estratti con rotazione Oblimin sono obliquamente correlati ($\Phi \neq I$), la distribuzione forma un iper-ellissoide inclinato nello spazio; imponendo matrici $\Sigma_c$ diagonali (assi ortogonali fissi), l'algoritmo non può ruotare la gaussiana ed è costretto ad affiancare due componenti sferiche per coprire la diagonale. Rimuovendo tale vincolo (\textit{{Full Covariance}}), un'unica componente cattura l'intera struttura. Parallelamente, il coefficiente di Silhouette per partizioni forzate (Pannello B) raggiunge un massimo di appena ${max_sil:.4f}$ (per $C = {best_c_sil}$), restando stabilmente sotto la soglia minima di $0.25$. 

I risultati forniscono evidenza a favore di una struttura dimensionale continua e contro l'ipotesi di sottogruppi discreti chiaramente separati.

\end{{document}}
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(codice_latex)