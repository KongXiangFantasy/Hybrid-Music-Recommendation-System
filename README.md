# Hybrid Music Recommendation System

A machine learning-based music recommendation system that combines collaborative filtering (Gaussian Mixture Models) and content-based filtering (K-Nearest Neighbors) to provide personalized song recommendations from the Spotify 1.2 Million Songs dataset.

## Overview

This project implements a hybrid recommendation engine that leverages both playlist-level patterns and individual audio features to generate accurate music recommendations. By addressing feature imbalance through weighted distance metrics, the system achieves a 40% reduction in average Euclidean distance, significantly improving recommendation quality.

## Key Features

- **Hybrid Architecture**: Combines GMM (collaborative filtering) with KNN (content-based filtering)
- **Large-Scale Processing**: Handles 134,712+ unique songs with 11 audio features
- **Feature Weighting**: Implements quantitative analysis to correct tempo dominance
- **Rigorous Evaluation**: Uses masking methodology on validation set
- **Dimensionality Reduction**: PCA for efficient high-dimensional feature space processing

## Dataset

- **Source**: Spotify Million Playlist Dataset + Spotify 1.2 Million Songs Dataset
- **Songs Processed**: 134,712 unique tracks
- **Playlists**: 1,000,000 playlists analyzed
- **Audio Features**: danceability, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, occurrence_count

## Technical Stack

- **Machine Learning**: Scikit-learn (GMM, KNN, PCA)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, t-SNE
- **Evaluation**: StandardScaler, Silhouette Score, Hit Rate@K

## Repository Structure
```
├── dataset_merge.ipynb          # Data integration & preprocessing
├── data_clustering.ipynb        # GMM clustering & analysis
├── validation.ipynb             # KNN recommendation & evaluation
├── Final_Report.pdf             # Complete technical documentation
└── README.md
```

## Methodology

### 1. Data Integration & Normalization

- Merged Spotify Million Playlist and 1.2M Songs datasets
- Applied z-score normalization (StandardScaler)
- Handled 134,712 songs with 11 audio features each

### 2. Collaborative Filtering (GMM)

- Computed average sound profile for each playlist
- Applied Gaussian Mixture Model with 10 components
- Used EM algorithm for soft clustering
- Achieved 95% classification accuracy

### 3. Content-Based Filtering (KNN)

- Implemented ball tree structure for efficient similarity search
- Computed Euclidean distances in feature space
- Applied feature weighting to address tempo dominance

### 4. Hybrid Strategy
```python
# Workflow
Input Playlist → GMM Clustering → Candidate Pool → KNN Ranking → Recommendations
```

## Key Results

### Feature Imbalance Correction

**Before Weighting:**
```
[1.23, 1.09, 1.14, 2.92, 1.00, 0.30]
```

**After Weighting (40% reduction):**
```
[0.73, 0.39, 0.41, 0.81, 0.55, 0.30]
```

### Model Performance

- **GMM Classification Accuracy**: 95%
- **Hit Rate@K=1**: 0.05%
- **Average Distance Reduction**: ~40%
- **Clustering Evaluation**: Silhouette score with elbow method

### Feature Analysis

- Identified tempo as dominant feature causing imbalance
- Implemented weighted distance metrics
- Achieved more balanced feature contributions

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-music-recommendation.git
cd hybrid-music-recommendation

# Install dependencies
pip install -r requirements.txt
```

## Requirements
```txt
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

## Usage

### 1. Data Preprocessing
```python
# Run dataset_merge.ipynb
# Merges datasets and normalizes features
```

### 2. Clustering Analysis
```python
# Run data_clustering.ipynb
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=10, covariance_type='full')
gmm.fit(X_pca)
labels = gmm.predict(X_pca)
```

### 3. Generate Recommendations
```python
# Run validation.ipynb
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean')
nbrs.fit(cluster_data)
distances, indices = nbrs.kneighbors(query_features)
```

## Evaluation Methodology

### Masking Strategy

- Randomly mask one song from each playlist
- Generate top-K recommendations using remaining songs
- Record "hit" if masked song appears in recommendations
- Hit Rate@K = Number of Hits / Number of Trials

### Cross-Validation

- Multiple independent runs with different seeds
- 80/20 train-test split
- Random Forest & SVM classifiers for validation

## Key Findings

1. **Feature Dominance**: Tempo dominated distance calculations before weighting
2. **Weighting Impact**: 40% reduction in Euclidean distance improved recommendations
3. **Hybrid Advantage**: Combining GMM and KNN leverages both collaborative and content signals
4. **Scalability**: Efficient processing of 134K+ songs using PCA and ball tree

## License

MIT License

## Authors

Yule Wang, Weiting (Tony) Ye, Daehwan (David) Kim, Adithyakrishna Arunkumar, Abhay Cheruthottathil

## Contact

For questions or collaboration: yule_wang2003@hotmail.com

## Acknowledgments

- Spotify for providing Million Playlist and 1.2M Songs datasets
- Scikit-learn community for ML tools
- SENG474 course staff for project guidance
