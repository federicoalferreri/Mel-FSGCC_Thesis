# Sound Event Localization And Detection With Mel-Scaled Frequency-Sliding Generalized Cross-Correlation

This study explores an advanced method for Sound Event Localiza- tion and Detection (SELD) by introducing Mel-Frequency Sliding Generalized Cross-Correlation (Mel-FSGCC), an extension of Frequency-Sliding Generalized Cross-Correlation (FS-GCC). The research aims to improve time delay estimation (TDE) by leveraging mel-scale frequency decomposition, which enhances spectral weighting based on human auditory perception. Unlike conventional Generalized Cross-Correlation (GCC) approaches, Mel-FSGCC adapts frequency processing to emphasize the most reliable spectral components, improving localization accuracy and robustness in reverberant and noisy environments.
The study evaluates the impact of replacing GCC with Mel-FSGCC within the SELDnet framework, a deep learning-based SELD system, using datasets gener- ated under varying noise levels, reverberation times (T60), and microphone array configurations. Performance is assessed following DCASE 2024 Task 3 challenge metrics, focusing on sound event detection accuracy (F-score), direction of arrival estimation error (DOAE), and relative distance estimation error (RDE).
Results demonstrate that Mel-FSGCC enhances sound event detection perfor- mance, particularly in setups with larger microphone spacings, while maintaining comparable localization and distance estimation accuracy to GCC.
Future work could explore more efficient ways to integrate this technique into deep learning networks, optimizing its computational feasibility for large-scale appli- cations. Additionally, testing Mel-FSGCC on new datasets may provide further insights or lead to unexpected findings and valuable advancements, refining its ap- plicability in acoustic scene analysis, spatial audio processing, and human-machine interaction.

## How to
In the script `pytorch_mel_fsgcc_cls_feature_class.py`, the function extract_file_feature is responsible for extracting features from audio files using the Mel-FSGCC representation.
The same function, but based on GCC, is available in the script `cls_feature_class.py`, which is identical to the one provided in the DCASE 2024 Task 3 baseline on GitHub.

The scripts `mel_fsgcc_batch_feature_extraction.py` and `batch_feature_extraction.py` handle the invocation of the functions responsible for feature extraction.
Finally, the script `mel_fsgcc_train_seldnet.py` is used for model training.

For dataset generation, use the script `pyroom.py` located in the SpatialScaper folder.
To modify specific parameters related to audio file generation, such as the SNR, T60 and microphone spacing, refer to the script `core.py`.
