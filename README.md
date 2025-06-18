# Mel-FSGCC_Thesis

Nello script pytorch_mel_fsgcc_cls_feature_class.py è presente la funzione extract_file_feature, che si occupa dell’estrazione delle feature dai file audio utilizzando la rappresentazione Mel-FSGCC.
La stessa funzione, ma basata sulla GCC, è disponibile nello script cls_feature_class.py, identico a quello fornito nella baseline su GitHub.

Gli script mel_fsgcc_batch_feature_extraction.py e batch_feature_extraction.py gestiscono l’invocazione delle funzioni responsabili dell’estrazione delle feature.
Infine, lo script mel_fsgcc_train_seldnet.py si occupa dell’addestramento del modello.

Per la generazione dei dataset, utilizzare lo script pyroom.py situato all'interno della cartella SpatialScaper.
Per modificare parametri specifici relativi alla generazione dei file audio, come ad esempio il SNR, fare riferimento allo script core.py.
