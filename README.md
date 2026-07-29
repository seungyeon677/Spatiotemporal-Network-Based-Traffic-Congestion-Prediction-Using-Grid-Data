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

├── data/                  # Sample grid metadata & topology definitions
│   ├── edge_index.npy     # 2x15327 edge list matrix
│   ├── edge_weight.npy    # 8308x8308 adjacency weights
│   └── grid_metadata.json # 8,308 road cell spatial coordinates
├── preprocessing/         # Image processing & graph building pipeline
│   ├── image_to_grid.py   # GDOT color-coding & distanceTime (kmHour) conversion
│   ├── match_template.py  # OpenCV template matching for accidents & construction
│   ├── rainfall_parser.py # SSEC Hydro Estimator resampling
│   └── build_network.py   # OSM road filtering & spatial join script
├── models/                # PyTorch Geometric model definitions
│   ├── stgcn.py           # Spatio-Temporal Graph Convolutional Network
│   ├── tgcn.py            # Temporal Graph Convolutional Network (GCN/ChebNet + GRU)
│   └── gc_lstm.py         # Graph Convolution embedded LSTM
├── utils/                 # Evaluation and spatial statistics
│   ├── sliding_window.py  # Input/target tensor creation
│   ├── metrics.py         # MAE, RMSE calculations
│   └── kde_analysis.py    # Spatial accuracy hotspot generator
├── train.py               # Model training script with Optuna hyperparameter tuning
├── evaluate.py            # Benchmark evaluation & visualization
├── requirements.txt       # Dependencies
└── README.md              # Documentation
