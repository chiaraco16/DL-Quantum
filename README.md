# DL-Quantum

Progetto di Deep Learning — DL-Quantum
Autore: Chiara Costantino — Laurea Magistrale in Artificial Intelligence and Data Science
Previsione delle dinamiche del modello PXP a 10 qubit (55 osservabili: 10 magnetizzazioni + 45 correlazioni) tramite una RNN e un Transformer causale, entrambi costruiti a partire dai soli layer primitivi di Keras e addestrati con un unico training loop multi-regime (teacher forcing + masked modeling + scheduled sampling). Il dataset è costituito da 400 traiettorie simulate di 1001 istanti ciascuna. La riproducibilità è controllata tramite semi fissi e determinismo delle operazioni in TensorFlow: esecuzioni indipendenti ripetute restituiscono metriche identiche e pesi con lo stesso hash MD5 (verificato bit-a-bit su tre run separati).
Struttura della consegna
Cartella / file	Contenuto
DL_Quantum.ipynb	Notebook di orchestrazione: esegue in sequenza tutti gli step su Colab.
src/	Moduli e script della pipeline.
tests/	Verifica della riproducibilità e controllo numerico sullo scheduled sampling.
artifacts/	Output degli esperimenti: EDA, figure, tabelle CSV e LaTeX, JSON dei risultati, pesi dei modelli selezionati e pacchetti di test. Viene rigenerato per intero rieseguendo il notebook.
report/	Documento finale in PDF (10 pagine), redatto in LaTeX importando tabelle e figure direttamente da artifacts/.
presentation/	Presentazione del progetto in formato .pptx.
I pacchetti di test inclusi (test_pack_L50_demo.npz e test_pack_L100_demo.npz, in artifacts/step3/models/) sono un sottoinsieme regolare del test set: i pacchetti integrali pesano 67 MB e sono stati esclusi per contenere la dimensione dell’archivio. Servono a rieseguire la cella di inferenza senza rileggere il CSV — run_inference.py cerca test_pack_L*.npz e ricade automaticamente sulla versione _demo, dichiarandolo a schermo. Le metriche che se ne ricavano non coincidono esattamente con quelle riportate nel documento e negli output salvati del notebook, calcolate sul test set completo; i pacchetti integrali si rigenerano con run_step3_experiments.py. Il dataset grezzo trajectories.csv (~440 MB) non è incluso nel pacchetto: il notebook lo carica da Google Drive nelle prime celle ed è necessario solo per rieseguire da zero lo sweep completo (Step 1–3), non per l’inferenza.
Dipendenze e ambiente di esecuzione
Codice in Python 3, pensato per Google Colab con GPU T4. Ambiente di riferimento su cui i risultati sono stati verificati: Python 3.13, NumPy 2.1, TensorFlow 2.20 su singola GPU. Librerie esterne:
•	tensorflow — architetture (RNN, Transformer), training loop custom, gestione dei tensori;
•	numpy — calcolo matriciale, metriche, finestre temporali, cache .npz;
•	pandas — lettura del dataset CSV;
•	matplotlib — figure di EDA, curve di training e grafici di rollout.
Moduli della libreria standard usati (nessuna installazione richiesta): os, json, argparse, dataclasses, subprocess, glob, shutil.
Struttura del codice (src/)
•	data_preprocessing.py — standardizzazione z-score, finestre temporali, partizionamento per traiettoria (senza data leakage), controlli di integrità ed EDA (inclusa la matrice di correlazione tra osservabili).
•	models.py — architetture RNN (GRU/LSTM) e Transformer (Model Subclassing con mascheratura causale); salvataggio dei pesi con metadati (save_model_bundle).
•	training.py — training loop unificato con tf.GradientTape: teacher forcing, masked modeling e scheduled sampling reale basato su iterazione a punto fisso.
•	experiments.py — sweep degli iperparametri, cross-validation per traiettoria, intervalli di confidenza al 95% con test di significatività, selezione a due stadi (con tie-break) e ablation dello schedule di training.
•	baselines.py — predittori non parametrici (persistence, drift, context-mean, train-mean) e Forecast Skill Score.
•	Script di esecuzione (run_*.py):
–	run_step1_preprocessing.py — controlli di integrità e cache dei tensori;
–	run_step2_demo.py — addestramento dimostrativo e figure qualitative;
–	run_step3_experiments.py — sweep completo e ablation;
–	run_inference.py — inferenza sul test set completo a partire dai modelli salvati e dai pacchetti compressi test_pack_L*.npz, senza rileggere il CSV.
–	analyze_phase_decay.py — analisi della degradazione del rollout: misura ampiezza e correlazione delle previsioni in funzione dell’orizzonte e distingue il collasso verso la media dalla perdita di fase; scrive phase_analysis.json.
Numeri dei risultati
src/dump_report_numbers.py legge i risultati JSON (results_full.json, ablation_full.json, step2_summary.json) e produce report_numbers.txt, un prospetto leggibile di tutte le metriche: configurazione migliore per finestra, RMSE, NRMSE, skill score, intervalli di confidenza, esito dei confronti di significatività, ablation. src/make_report_macros.py produce lo stesso contenuto come macro LaTeX. Servono a controllare riga per riga che le cifre riportate nel documento coincidano con quelle prodotte dall’esperimento.
Istruzioni di esecuzione (Colab)
L’orchestrazione degli esperimenti avviene tramite il notebook DL_Quantum.ipynb.
•	Ordine: eseguire le celle in sequenza, partendo dal setup dell’ambiente e dal pre-caricamento locale dei dati.
•	Tempi stimati (GPU T4): Step 1 ≈ 2 min · Step 2 ≈ 5 min · Step 3 (sweep completo + ablation) ≈ 60–90 min.
Il documento è redatto in LaTeX importando le tabelle dei risultati direttamente dai file generati dalla pipeline (artifacts/step3/table1_L50.tex, table1_L100.tex, table_ablation.tex, table_hp_configs.tex) e le figure da artifacts/: nessuna cifra riportata nel testo è digitata a mano, quindi il documento non può divergere dai risultati degli esperimenti. Nel pacchetto è incluso il PDF compilato.
Riproducibilità
Tutti i semi sono fissati e il determinismo delle operazioni TensorFlow è abilitato. Lo script tests/verify_reproducibility.py confronta due esecuzioni indipendenti (metriche, Tabella 1 e pesi bit-a-bit); tests/test_scheduled_sampling.py verifica numericamente la proprietà a punto fisso dello scheduled sampling.
