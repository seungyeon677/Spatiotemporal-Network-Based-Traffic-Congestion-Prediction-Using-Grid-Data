# Spatio-Temporal Network-Based Traffic Congestion Prediction Modeling Using Grid Data: A Case Study of Atlanta, Georgia, USA

> **Official Repository for Master's Thesis**  
> **Author:** Seungyeon Lee
> **Advisor:** Prof. Chul Sue Hwang
> **Department:** Department of Geography, Graduate School of Kyung Hee University


---

## 📌 Abstract

This study proposes a GeoAI famework that transforms image based traffic information into a spatially interpretable structure for spatiotemporal traffic congestion prediction. Focusing on the Atlanta metropolitan area in Georgia, USA, traffic images collec ted at 10 minute intervals were converted into a space time grid based network structure. The connectivity and directionality between each grid were defined based on road network data, enabling the constructed graph to reflect the space time characteristic s of traffic congestion.
Among various modeling approaches, the GC-LSTM model combined with historical traffic congestion patterns and accident variables demonstrated the highest prediction performance. Prediction accuracy varied spatially, with lower accuracy observed in regions with high temporal variability. Notably, areas such as the downtown connector, the I 285/I 85 interchange in the northeast, and the I-285/I-20 interchange in the east exhibited different optimal variables configurations depending o n their local characteristics.
By reconstructing image-based traffic data within a spatial context and enhancing prediction accuracy through this methodology, the study expands the applicability of image data in the fields of traffic analysis and GeoAI. Furthermore, by emphasizing the importance of optimizing predictive variables in consideration of the space-time uncertainty and regional characteristics of traffic data, the study provides an analytical foundation for addressing various urban issues, including smart city development, transportation planning, and disaster response.

---

## 🛠️ Research Methodology & Framework

The overall workflow consists of four main phases: (1) Grid-based Spatio-Temporal Network Construction, (2) Optimal Variable Combination Search, (3) Model Comparison, and (4) Spatial-Temporal Error Pattern Interpretation.


<img width="1181" height="974" alt="image" src="https://github.com/user-attachments/assets/aa846764-6100-4e7a-a7d7-56e11cdfab55" />


## 📂 Repository Structure

```text
.
├── models/
│   ├── optuna_GC-LSTM.py          # GC-LSTM model training & Optuna hyperparameter tuning
│   ├── optuna_STGCN.py            # STGCN model training & Optuna hyperparameter tuning
│   ├── optuna_TGCN-C.py           # TGCN-ChebNet model training & Optuna hyperparameter tuning
│   └── optuna_TGCN-G.py           # TGCN (Standard GCN) model training & Optuna hyperparameter tuning
├── preprocessing/
│   ├── build_connectivity.py      # Spatial grid network creation
│   ├── congestion_to_KmH.py       # Normalization & metric conversion to kmHour
│   ├── create_feature_target.py   # Sliding window processing for feature/target input tensors
│   ├── degrade_congestion_sum.py  # 5x5 max-kernel resampling & cumulative spatial aggregation
│   └── degrade_crop_congestion.py # Spatial cropping to Atlanta study area extent
├── sample_data/
│   ├── congestion_20241101_0000.png  # Sample 10-min traffic congestion map image
│   ├── crash_20241101_0000.png       # Sample traffic incident & construction symbol map image
│   └── precipitation_20241101_0000.png # Sample SSEC rainfall map image
└── README.md                          # Project documentation
