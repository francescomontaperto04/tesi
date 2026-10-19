from pathlib import Path

def genera_report_spls(
    metrics: dict,
    filepath: Path,
    n_subj: int,
    p_fmri: int,
    q_behav: int,
    decoder_metrics: dict
):
    best_c = metrics.get('best_c', 0.0)
    active_edges = metrics.get('active_edges', 0)
    pct_active = metrics.get('pct_active', 0.0)
    
    oos_cov = metrics.get('oos_cov', 0.0)
    oos_r = metrics.get('oos_r', 0.0)
    in_cov = metrics.get('in_cov', 0.0)
    in_r = metrics.get('in_r', 0.0)
    p_oos_cov = metrics.get('p_oos_cov', 1.0)
    p_oos_r = metrics.get('p_oos_r', 1.0)

    p_str_cov = r'p < 10^{-3}' if p_oos_cov < 0.001 else f'p = {p_oos_cov:.3f}'
    p_str_r = r'p < 10^{-3}' if p_oos_r < 0.001 else f'p = {p_oos_r:.3f}'

    # Generazione righe per tabella Top 10 Edges
    top_edges = decoder_metrics.get('top_edges_list', [])
    righe_top_edges = [
        f"\\texttt{{{row['Connessione_Network'].replace(' $\\leftrightarrow$ ', '--')}}} & {row['Peso_sPLS']:+.4f} \\\\"
        for row in top_edges
    ]

    codice_latex = fr"""\documentclass[12pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[italiano]{{babel}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\geometry{{a4paper, top=3cm, bottom=3cm, left=3cm, right=3cm}}

\title{{\textbf{{Report Analisi Multivariata fMRI-Comportamento (sPLS)}}}}
\author{{}}
\date{{}}

\begin{{document}}

\maketitle

\section{{Allineamento Topologico e Dimensionale}}
Il campione operativo, risultante dall'intersezione tra lo spazio fenotipico latente e i connettomi fMRI disponibili, consta di $N={n_subj}$ soggetti.
Lo spazio neurobiologico è stato definito proiettando le serie temporali parcellizzate sui network funzionali macroscopici, estraendo le connettività funzionali (matrici di correlazione di Pearson incrociate, stabilizzate tramite trasformata $Z$ di Fisher per additività). Il tensore finale sottoposto a decomposizione comprende $p={p_fmri}$ feature neurali (connessioni di network) e $q={q_behav}$ punteggi fattoriali latenti comportamentali.

\section{{Ottimizzazione sPLS e Controllo dell'Overfitting}}
Per isolare il gradiente di co-variazione multivariata tra cervello e comportamento minimizzando il rumore topologico, si è impiegata la \textit{{Sparse Partial Least Squares}} (sPLS). Il parametro di regolarizzazione sparsa $L_1$ sul versante neurale ($c$) è stato ricavato tramite 5-Fold Cross-Validation, selezionando il modello in grado di massimizzare la correlazione Out-of-Sample ($r_{{OOS}}$).
L'iperparametro ottimale è risultato $c={best_c:.2f}$.

\section{{Risultati sPLS e Validazione Permutazionale}}
L'applicazione del modello sull'intero campione ha estratto una singola componente latente bimodale. La validazione algebrica (Tabella \ref{{tab:spls_metrics}}) è stata testata mediante Permutation Test non parametrico (5000 iterazioni). Per eludere il \textit{{data leakage}}, l'intera 5-Fold CV è stata annidata e ricalcolata in ciascuna permutazione.

\begin{{table}}[htbp]
\centering
\caption{{Metriche di Associazione e Test Permutazionale (1000 iterazioni)}}
\label{{tab:spls_metrics}}
\begin{{tabular}}{{lrrr}}
\toprule
\textbf{{Metrica}} & \textbf{{In-Sample}} & \textbf{{Out-Of-Sample}} & \textbf{{Significatività (OOS)}} \\
\midrule
Covarianza & ${in_cov:+.4f}$ & ${oos_cov:+.4f}$ & ${p_str_cov}$ \\
Correlazione di Pearson ($r$) & ${in_r:+.4f}$ & ${oos_r:+.4f}$ & ${p_str_r}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Decodifica della Firma Neurale}}
Il modello neurale ottimizzato risulta strettamente parsimonioso: dei $p={p_fmri}$ edge funzionali originari, l'operatore di proiezione seleziona esattamente {active_edges} connessioni salienti ({pct_active:.2f}\%), confinando il resto della topologia a peso nullo. La topologia strutturale della firma latente è riepilogata in Tabella \ref{{tab:top_edges}}. (Il referto vettoriale completo è archiviato digitalmente nel file \texttt{{connessioni\_attive.csv}}).

\begin{{table}}[htbp]
\centering
\caption{{Top 10 Connessioni Neurali per Importanza Assoluta (Pesi $v$)}}
\label{{tab:top_edges}}
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Interazione Topologica (Macro-Network)}} & \textbf{{Peso Associativo (sPLS)}} \\
\midrule
{chr(10).join(righe_top_edges)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Rappresentazione Visiva dei Risultati}}
In Figura \ref{{fig:spls_plots}} sono riportati i grafici vettoriali estratti dal modello sPLS. Il profilo comportamentale (A) illustra la polarizzazione dei fattori psicometrici (\textit{{Vettore u}}). La distribuzione nulla permutazionale (B) certifica la robustezza algoritmica dell'iperparametro rispetto all'ipotesi nulla. Infine, la matrice topologica inferiore (C) traccia la polarità funzionale del connettoma (\textit{{Vettore v}}), localizzando con precisione il disaccoppiamento dei network.

\begin{{figure}}[htbp]
    \centering
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{spls_comportamento_u.pdf}}
        \vspace{{0.1cm}}
        
        \small (A) Pesi comportamentali $u$
    \end{{minipage}}
    \hfill
    \begin{{minipage}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{spls_permutazioni_null.pdf}}
        \vspace{{0.1cm}}
        
        \small (B) Distribuzione nulla 
    \end{{minipage}}
    
    \vspace{{0.6cm}}
    
    \begin{{minipage}}[b]{{0.85\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{spls_connettoma_v.pdf}}
        \vspace{{0.1cm}}
        
        \small (C) Heatmap dei pesi neurali $v$ ($19 \times 19$)
    \end{{minipage}}
    
    \caption{{Visualizzazione integrata del modello sPLS: (A) Asse fenotipico latente, (B) Validazione non parametrica a 1000 iterazioni, (C) Topologia dei circuiti neurali disaccoppiati.}}
    \label{{fig:spls_plots}}
\end{{figure}}

\end{{document}}
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(codice_latex)