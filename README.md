# DL-Quantum

Previsione delle dinamiche del modello PXP mediante Deep Learning

**Autore:** Chiara Costantino  
**Programma:** Laurea Magistrale in Artificial Intelligence and Data Science

---

## 📋 Descrizione del Progetto

**DL-Quantum** è uno studio di Deep Learning sulla previsione delle dinamiche di un sistema PXP a 10 qubit. Il progetto confronta due architetture neurali—una RNN e un Transformer causale—costruite esclusivamente con layer primitivi di Keras, per la predizione di 55 osservabili (10 magnetizzazioni e 45 correlazioni).

**Caratteristiche principali:**
- Dataset: 400 traiettorie simulate di 1001 istanti ciascuna
- Training unificato: teacher forcing + masked modeling + scheduled sampling
- Riproducibilità verificata: esecuzioni indipendenti restituiscono metriche identiche e pesi con hash MD5 identico (verificato bit-a-bit su tre run separati)

---

## 📁 Struttura del Progetto

| Elemento | Descrizione |
|----------|-------------|
| **DL_Quantum.ipynb** | Notebook di orchestrazione che esegue sequenzialmente tutti gli step su Google Colab |
| **src/** | Moduli Python e script della pipeline di preprocessing, training e inferenza |
| **tests/** | Suite di test per verificare riproducibilità e validazione numerica dello scheduled sampling |
| **artifacts/** | Output degli esperimenti: EDA, figure, tabelle (CSV, LaTeX), risultati JSON, pesi dei modelli e pacchetti di test |
| **report/** | Documento finale (10 pagine) in LaTeX con tabelle e figure importate direttamente da `artifacts/` |
| **presentation/** | Presentazione del progetto in formato PowerPoint |

### 📦 Pacchetti di Test

I pacchetti di test demo inclusi (`test_pack_L50_demo.npz` e `test_pack_L100_demo.npz`, in `artifacts/step3/models/`) sono sottoinsiemi regolari del test set. I pacchetti integrali (67 MB) sono stati esclusi per contenere la dimensione dell'archivio ma si rigenerano con `run_step3_experiments.py`.

**Nota:** `run_inference.py` rileva automaticamente i pacchetti disponibili e usa la versione `_demo` con una notifica a schermo. Le metriche derivate dai pacchetti demo non coincidono esattamente con quelle del documento (calcolate sul test set completo).

### 📊 Dataset

Il dataset grezzo `trajectories.csv` (~440 MB) non è incluso nel pacchetto: il notebook lo carica da Google Drive nelle prime celle. È necessario solo per rieseguire lo sweep completo da zero (Step 1–3), non per l'inferenza.

---

## ⚙️ Ambiente di Esecuzione

**Linguaggio:** Python 3  
**Piattaforma consigliata:** Google Colab con GPU T4  
**Versioni di riferimento:** Python 3.13, NumPy 2.1, TensorFlow 2.20 su singola GPU

### 📦 Dipendenze Esterne

| Libreria | Utilizzo |
|----------|----------|
| **TensorFlow** | Architetture (RNN, Transformer), training loop custom, gestione tensori |
| **NumPy** | Calcolo matriciale, metriche, finestre temporali, cache .npz |
| **Pandas** | Lettura dataset CSV |
| **Matplotlib** | Visualizzazione (EDA, curve di training, grafici di rollout) |

**Libreria standard** (nessuna installazione richiesta): `os`, `json`, `argparse`, `dataclasses`, `subprocess`, `glob`, `shutil`

---

## 🏗️ Architettura del Codice

### Moduli Principali (`src/`)

- **`data_preprocessing.py`**  
  Standardizzazione z-score, sliding windows temporali, partizionamento per traiettoria (senza data leakage), controlli di integrità, EDA e matrice di correlazione tra osservabili.

- **`models.py`**  
  Architetture RNN (GRU/LSTM) e Transformer con Model Subclassing, mascheratura causale e serializzazione dei pesi con metadati (`save_model_bundle`).

- **`training.py`**  
  Training loop unificato con `tf.GradientTape`: implementa teacher forcing, masked modeling e scheduled sampling reale basato su iterazione a punto fisso.

- **`experiments.py`**  
  Hyperparameter sweep, cross-validation per traiettoria, intervalli di confidenza al 95%, test di significatività statistica, selezione a due stadi con tie-break, ablation study sullo schedule di training.

- **`baselines.py`**  
  Predittori non parametrici (persistence, drift, context-mean, train-mean) e Forecast Skill Score.

### 🚀 Script di Esecuzione

- **`run_step1_preprocessing.py`**  
  Controlli di integrità e cacheing dei tensori

- **`run_step2_demo.py`**  
  Training dimostrativo con figure qualitative

- **`run_step3_experiments.py`**  
  Sweep completo degli iperparametri e ablation study

- **`run_inference.py`**  
  Inferenza sul test set completo da modelli salvati e pacchetti compressi, senza rileggere il CSV

- **`analyze_phase_decay.py`**  
  Analisi della degradazione del rollout: ampiezza e correlazione delle previsioni in funzione dell'orizzonte, distinzione tra collasso verso la media e perdita di fase. Output: `phase_analysis.json`

### 📈 Reportistica

- **`dump_report_numbers.py`**  
  Legge i risultati JSON (`results_full.json`, `ablation_full.json`, `step2_summary.json`) e genera `report_numbers.txt`: prospetto leggibile di tutte le metriche, configurazioni migliori, RMSE, NRMSE, skill score, intervalli di confidenza e esiti dei test di significatività.

- **`make_report_macros.py`**  
  Produce lo stesso contenuto in formato macro LaTeX per l'importazione diretta nel documento.

**Nota di verifica:** Queste utility permettono di controllare riga per riga che le cifre nel documento coincidano con quelle prodotte dall'esperimento.

---

## 🚀 Istruzioni di Esecuzione

### Su Google Colab

1. Apri il notebook **DL_Quantum.ipynb**
2. Esegui le celle in sequenza, partendo dal setup dell'ambiente
3. I dati verranno precaricati da Google Drive nelle prime celle

### ⏱️ Tempi Stimati (GPU T4)

- **Step 1** (Preprocessing): ~2 minuti
- **Step 2** (Demo e training dimostrativo): ~5 minuti
- **Step 3** (Sweep completo + Ablation): ~60–90 minuti

---

## 📄 Generazione del Report

Il documento finale (in PDF) è redatto in LaTeX e importa automaticamente le tabelle dei risultati generati dalla pipeline:
- `artifacts/step3/table1_L50.tex`
- `artifacts/step3/table1_L100.tex`
- `artifacts/step3/table_ablation.tex`
- `artifacts/step3/table_hp_configs.tex`

**Garanzia di coerenza:** Nessuna cifra è digitata manualmente nel testo; il documento non può quindi divergere dai risultati degli esperimenti. Il PDF compilato è incluso nel pacchetto.

---

## ✅ Riproducibilità

La riproducibilità è garantita attraverso:

- **Seed fissi** su tutte le operazioni stocastiche
- **Determinismo TensorFlow** abilitato per operazioni CUDA
- **Verifica empirica:** `tests/verify_reproducibility.py` confronta due esecuzioni indipendenti verificando:
  - Identità esatta delle metriche
  - Concordanza della Tabella 1
  - Equivalenza bit-a-bit dei pesi salvati (hash MD5)

- **Validazione dello scheduled sampling:** `tests/test_scheduled_sampling.py` verifica numericamente la proprietà a punto fisso dello scheduled sampling.

---
