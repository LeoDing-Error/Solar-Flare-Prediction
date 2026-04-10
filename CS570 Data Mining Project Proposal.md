CS570 Data Mining \- Project Proposal

**Detecting Solar Flare Precursors via Multi-Modal Anomaly Detection**

Leo Ding & Ermias Assefa \- Emory University \- 3/8/2026

# **1\. Problem Statement and Motivation**

Solar flares are intense radiation bursts that can disrupt satellite communications, degrade GPS accuracy, and damage power grids. In May 2024, active region AR 3664 produced a series of X-class flares and associated CMEs that triggered the strongest geomagnetic storm in over two decades, causing global GPS positioning outages lasting up to 20 hours and forcing agricultural operations to halt during planting season \[1\]. Detecting such flare activity before eruption remains a key upstream forecasting challenge. 

Before a major flare, the Sun’s active regions change: magnetic field lines twist, energy builds up in unstable configurations, and the X-ray background shifts in subtle ways \[2\]. These precursor patterns are buried in noisy, high-dimensional time series data and may involve correlations across multiple measurement channels that are invisible when examining only a single data stream. The question here is: can unsupervised anomaly detection identify these subtle precursor patterns and distinguish them from normal solar variability?

# **2\. Background and Existing Solutions**

Flares are classified by their peak X-ray intensity into C, M, and X classes, with X being the strongest \[3\]. NOAA’s Space Weather Prediction Center (SWPC) currently relies on human forecasters who look at solar images and sunspot classification tables to estimate flare probabilities over 24–48 hours. But this process is slow and error-prone: a 26-year validation study found that SWPC forecasts don’t consistently beat simple baselines like persistence (tomorrow looks like today) and climatology (use historical averages) \[4\]. An earlier study by Crown \[5\] found similar issues, with high false alarm rates and only small gains from human judgment over the look-up table.

Supervised ML methods like SVMs, random forests, and LSTMs have been tried for flare prediction \[6, 7\], but big flares are rare, which creates a severe class imbalance problem. These models also need labeled data and can only learn from flares that have already happened, so they might miss new kinds of precursors. Unsupervised anomaly detection works differently: instead of learning what flares look like, it learns what normal solar activity looks like and flags anything that doesn’t fit. This handles rare events naturally, doesn’t need labels, and can potentially catch precursor types that have never been seen before. The downside is that not every anomaly means a flare is coming, and some flares may not have detectable precursors at all. This project tests how well anomaly detection on GOES X-ray data can separate real precursor signals from noise.

# **3\. Initial Approach / Baseline**

**Feature Engineering.** Raw X-ray flux is great for detecting flares (they show up as obvious spikes) but not for catching the subtle changes before them. To pick up precursor dynamics, we’ll compute features over sliding time windows that describe how the signal is changing: basic statistics (mean, variance, skewness, kurtosis), rate-of-change metrics, frequency-domain features via FFT (dominant frequency, spectral entropy), and complexity measures (sample entropy, permutation entropy). We’ll use a 6-hour window with a 1-hour stride as our default and test other sizes during ablation.

**Baseline Algorithms.** We’ll use three classical anomaly detectors that work in different ways. Isolation Forest finds anomalies by isolating points through random splits—outliers are easier to isolate and get shorter path lengths \[8\]. Local Outlier Factor (LOF) compares each point’s local density to its neighbors and flags points in unusually sparse regions \[9\]. One-Class SVM fits a tight boundary around the normal data and flags anything outside it \[10\]. All three are trained only on non-flare periods so they learn what “normal” looks like. We then score the 24-hour windows before each M/X-class flare to see how anomalous they appear.

# **4\. Proposed Innovations**

**LSTM Autoencoder.** The baseline methods look at each time window in isolation. They don’t consider what happened in the previous windows, so they miss patterns that develop over time. Flare precursors likely build up gradually over hours, which means the order of the data matters. An LSTM autoencoder \[11\] handles this by learning what normal sequences of solar activity look like. We train it on sequences from quiet periods: it compresses each sequence down to a small representation and tries to rebuild it. On normal data, the rebuilt version closely matches the original. On unusual data—like a precursor pattern it hasn’t seen before—the rebuilt version is a poor match. We use this mismatch (the reconstruction error) as the anomaly score. The model never needs examples of precursors; it only needs to learn “normal,” and flags anything it can’t recreate. This idea has worked well for catching anomalies in industrial sensor data \[12\]. We’ll test sequence lengths of 12, 24, and 48 hours and latent sizes of 16, 32, and 64\.

**Ensemble Scoring.** Our four detectors (IF, LOF, OCSVM, LSTM autoencoder) each define “unusual” differently. A precursor that looks normal to one method might look suspicious to another. Instead of picking the best single detector, we combine them. Each detector ranks every time window from most to least anomalous, and the final score is the average rank across all four. This simple approach doesn’t need labeled data and avoids the problem of detectors using different score scales.

**Temporal Consistency Filtering.** A single anomalous window could just be noise. Real precursors should show up across several windows in a row as energy builds toward a flare. To cut down on false alarms, we smooth the ensemble scores with a rolling median and only raise an alert if the score stays above the threshold for at least a few consecutive hours (e.g., 3 hours). We’ll test different duration requirements to balance false alarm reduction against the risk of filtering out genuine short precursors.

# **5\. Datasets**

**GOES X-ray Flux.** Our primary data source is the GOES satellite X-ray irradiance record from NOAA’s National Centers for Environmental Information (NCEI). It provides 1-minute resolution measurements in two wavelength bands from 2010 to 2024, totaling about 7.5 million data points \[3\]. This covers solar cycle 24 and the start of cycle 25, giving us both quiet periods and active periods with frequent flares. 

**SWPC Flare Catalog.** We get ground-truth flare labels from NOAA’s SWPC event reports, which record start time, peak time, end time, X-ray class, and active region for each flare \[4\]. These labels define our evaluation windows and our training exclusion zones. The catalog has several hundred M-class and dozens of X-class events in our study period—enough for meaningful evaluation.

**Processed Dataset.** We’ll release a cleaned version of the data with gaps filled, quality flags applied, pre-computed sliding-window features, and documented train/test splits so others can reproduce our results and build on them.

# **6\. Evaluation Methods and Metrics**

**Procedure.** We split the data by time: 2010–2019 for training and 2020–2024 for testing. This prevents leakage and checks whether patterns learned from one solar cycle phase generalize to another. For each M/X-class flare in the test set, we check whether the detector flagged anything in the 24 hours before the flare started. Non-flare periods serve as true negatives. We’ll report M-class and X-class results separately to see if stronger flares are easier to predict.

**Metrics.** Recall (how many flares had a detected precursor), Precision (how many alerts actually preceded a flare), F1-Score, and ROC-AUC (overall discrimination). We’ll also report Lead Time—the average gap between the first alert and flare onset—which measures practical early-warning value. We compare against a random baseline, a threshold on raw flux rate-of-change, and published SWPC forecast performance \[4, 5\]. Ablation experiments will show how much each component (LSTM autoencoder, ensemble, temporal filtering) adds.

# **7\. Deliverables and Timeline**

**Deliverables:** (1) Processed GOES X-ray dataset with documentation, (2) Python code repository with pipeline and models, (3) Trained baseline \+ LSTM models, (4) Final report with results/analysis, (5) Demo Jupyter notebook.

**Timeline:**

Weeks 1-2:  Data acquisition, cleaning, EDA, feature engineering

Weeks 2-3: Implement baselines (IF, LOF, OCSVM), initial evaluation

Weeks 3-5: LSTM autoencoder, multi-modal fusion, ensemble method

Weeks 5-6: Final evaluation, ablations, report, demo notebook

# **References**

\[1\] Yang, Z., et al. (2025). "Impacts of the May 2024 Extreme Geomagnetic Storm on Global High-Accuracy GPS Positioning Solutions." *Space Weather*, 23\. [https://doi.org/10.1029/2025SW004547](https://doi.org/10.1029/2025SW004547)

\[2\] Schrijver, C. J. (2007). “A Characteristic Magnetic Field Pattern Associated with All Major Solar Flares and Its Use in Flare Forecasting.” The Astrophysical Journal Letters, 655(2), L117–L120. https://doi.org/10.1086/511857

\[3\] NOAA National Centers for Environmental Information. (2024). GOES X-ray Flux Data. Retrieved from https://www.ncei.noaa.gov/data/goes-space-environment-monitor/

\[4\] Camporeale, E. et al. (2025). “Verification of the NOAA Space Weather Prediction Center Solar Flare Forecast (1998–2024).” Space Weather, 23\. https://doi.org/10.1029/2025SW004546

\[5\] Crown, M. D. (2012). “Validation of the NOAA Space Weather Prediction Center’s Solar Flare Forecasting Look-Up Table and Forecaster-Issued Probabilities.” Space Weather, 10(6). https://doi.org/10.1029/2011SW000760

\[6\] Florios, K. et al. (2018). “Forecasting Solar Flares Using Magnetogram-Based Predictors and Machine Learning.” Solar Physics, 293, 28\. https://doi.org/10.1007/s11207-018-1250-4

\[7\] Nishizuka, N. et al. (2017). “Solar Flare Prediction Model with Three Machine-Learning Algorithms Using Ultraviolet Brightening and Vector Magnetograms.” The Astrophysical Journal, 835(2), 156\. https://doi.org/10.3847/1538-4357/835/2/156

\[8\] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). “Isolation Forest.” In Proceedings of the IEEE International Conference on Data Mining (ICDM), 413–422. https://doi.org/10.1109/ICDM.2008.17

\[9\] Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). “LOF: Identifying Density-Based Local Outliers.” In Proceedings of the ACM SIGMOD International Conference on Management of Data, 93–104. https://doi.org/10.1145/342009.335388

\[10\] Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). “Estimating the Support of a High-Dimensional Distribution.” Neural Computation, 13(7), 1443–1471. https://doi.org/10.1162/089976601750264965

\[11\] Hochreiter, S. & Schmidhuber, J. (1997). “Long Short-Term Memory.” Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

\[12\] Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). “LSTM-Based Encoder-Decoder for Multi-Sensor Anomaly Detection.” arXiv preprint arXiv:1607.00148.

\[13\] Stone, E. C., Frandsen, A. M., Mewaldt, R. A., Christian, E. R., Margolies, D., Ormes, J. F., & Snow, F. (1998). “The Advanced Composition Explorer.” Space Science Reviews, 86, 1–22. https://doi.org/10.1023/A:1005082526237