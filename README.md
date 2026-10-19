# Pipeline Computazionale Integrata per Tesi di Laurea
## Framework di Analisi Multivariata RDoC: Psicometria Strutturale & Connettomica fMRI

Questo repository ospita l'intera infrastruttura computazionale sviluppata per l'elaborazione dei dati della tesi. Si adotta un'architettura **modulare e disaccoppiata**, suddivisa in due pipeline sequenziali che traducono dati clinico-comportamentali grezzi e serie temporali fMRI ad alta risoluzione in una firma neuro-comportamentale unificata.

## 📊 Reperibilità dei Dati (Data Availability)
Per motivi di privacy (GDPR) e conformità alle normative sui dati clinici, i file `.csv`, le matrici `.npy` e le serie temporali fMRI non sono inclusi in questo repository. 

I dati originali su cui si basa l'intera pipeline di elaborazione appartengono al **Transdiagnostic Connectome Project** (**OpenNeuro ds005237**; Holmes et al., 2024).

Per replicare o eseguire le analisi:
1. Scaricare il dataset di riferimento da [OpenNeuro (ds005237)](https://openneuro.org/datasets/ds005237).
2. Posizionare i file strutturati all'interno della cartella di input (`data/`) rispettando le nomenclature attese dai moduli di preprocessing.

## 📂 Architettura e Organizzazione del Progetto

tesi/
├── .gitignore
├── README.md
├── requirements.txt
├── pipeline_FA/                  <-- PRIMA PIPELINE (EFA / Psicometria)
│   ├── data/                     <-- scaricare dal dataset OpenNeuro ds005237
│   │   └── matrice_fenotipica_completa.csv
│   ├── modules/
│   │   ├── cleaning.py
│   │   ├── imputation.py
│   │   ├── normalization.py
│   │   ├── diagnostics.py
│   │   └── analysis.py
│   ├── reporting/
│   │   └── latex.py
│   ├── output_pipeline/
│   └── main.py
└── pipeline_fmri/                <-- SECONDA PIPELINE (sPLS / fMRI)
    ├── data/
    │   ├── matrice_punteggi_fattoriali.csv (Copiato/collegato dalla pipeline 1)
    │   ├── whole_brain_region_labels_ordered_Cerebellum_Seperate.npy   <-- scaricare dal dataset
    │   └── fMRI_timeseries_clean_denoised_GSR_parcellated/             <-- scaricare dal dataset
    ├── modules/
    │   ├── network_mapping.py
    │   ├── spls_engine.py
    │   ├── edge_decoder.py
    │   ├── reporting_fmri.py
    │   └── plotting_fmri.py
    ├── output_spls/
    └── main_spls.py

Il progetto è strutturato in due macro-ambienti di calcolo indipendenti:

### 1. Pipeline Psicometrica (`/pipeline_FA`)
Gestisce la purificazione dello spazio fenotipico e l'estrazione dei costrutti latenti dimensionali.
* **`cleaning.py`**: Quality control comportamentale (filtri su velocità di compilazione e risposte seriali) ed epurazione atomica via Regex dei singoli item di questionario.
* **`imputation.py`**: Imputazione non lineare iterativa tramite *MissForest* (Random Forest) per preservare la struttura di covarianza senza bias indotti dal *listwise deletion*.
* **`normalization.py`**: Aggregazione delle sottoscale cliniche (es. MADRS) e normalizzazione competitiva basata sulla massimizzazione dell'indice di Shapiro-Wilk.
* **`diagnostics.py`**: Screening algebrico preliminare (KMO, Bartlett, VIF, Condition Index $\kappa$) per attestare la fattorizzabilità della matrice.
* **`analysis.py`**: Triangolazione oggettiva della dimensionalità ($k$) mediante *Parallel Analysis* permutazionale di Horn e test MAP di Velicer, seguita da Analisi Fattoriale Esplorativa (EFA con MinRes e rotazione Oblimin).
* **`latex.py`**: Compilazione automatica del report metodologico in LaTeX.

### 2. Pipeline Multivariata fMRI (`/pipeline_fmri`)
Gestisce l'accoppiamento multivariato tra i punteggi fattoriali comportamentali e le connettività funzionali cerebrali.
* **`network_mapping.py`**: Parsing delle etichette corticali/subcorticali, costruzione della matrice di proiezione spaziale $W$ (17-Network Yeo + Cereb/Subcort) e calcolo delle matrici di correlazione di Pearson stabilizzate con trasformata $Z$ di Fisher.
* **`spls_engine.py`**: Motore *Sparse Partial Least Squares* (sPLS) ottimizzato con BLAS Level-3, Cross-Validation a 5 fold e Permutation Test non parametrico (1000 iterazioni con nested-CV per azzerare il *data leakage*).
* **`edge_decoder.py`**: Mappatura vettoriale dei pesi neurali $v$ sulle coppie inter-network per identificare la firma topologica attiva.
* **`plotting_fmri.py`**: Generazione di grafici vettoriali in PDF (Profilo comportamentale $u$, Heatmap del connettoma $19 \times 19$ e doppi istogrammi della distribuzione nulla).
* **`reporting_fmri.py`**: Compilazione automatica del report LaTeX sui risultati sPLS e integrazione delle tabelle topologiche.

---
⚠️ Nota: Le due pipeline sono disaccoppiate e indipendenti: ricordarsi di inserire/copiare manualmente la matrice_punteggi_fattoriali.csv nella cartella dei dati della seconda pipeline.

## ⚙️ Requisiti di Sistema e Installazione

Il codice richiede un ambiente Python (consigliato 3.10 o superiore). Per installare tutte le librerie necessarie (`numpy`, `pandas`, `scikit-learn`, `factor_analyzer`, `matplotlib`, `seaborn`, `joblib`, `h5py`), eseguire:

```bash
pip install numpy pandas scikit-learn factor_analyzer matplotlib seaborn joblib h5py

🚀 Esecuzione della Ricerca
Step 1: Esecuzione della Pipeline Psicometrica
Eseguire l'orchestratore per pulire i dati e calcolare i punteggi fattoriali:
cd pipeline_psicometrica
python main.py

Step 2: Sincronizzazione del File Comportamentale
Copiare il file matrice_punteggi_fattoriali.csv appena generato e posizionarlo all'interno della cartella dei dati della seconda pipeline:
cp pipeline_psicometrica/output/matrice_punteggi_fattoriali.csv pipeline_fmri/data/

Step 3: Esecuzione della Pipeline fMRI & sPLS
Procedere con l'analisi multivariata e la generazione dei report vettoriali:
cd ../pipeline_fmri
python main.py