import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import shap
import xgboost as xgb
matplotlib.rcParams['font.family'] = 'sans-serif'


# Ensure output folder exists
OUTPUT_DIR = "./visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_target_distribution(y, title="Distribution of Drag Coefficients"):
    plt.figure(figsize=(6, 4))
    sns.histplot(y, bins=20, kde=True)
    plt.title(title)
    plt.xlabel("Average Cd")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/drag_distribution.png")
    plt.close()

def plot_feature_correlation(X, y):
    df = X.copy()
    df["Average Cd"] = y
    corr = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap (including Cd)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()

def plot_pca_projection(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.8)
    plt.title("PCA Projection Colored by Drag Coefficient")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Average Cd")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca_projection.png")
    plt.close()
    
def plot_xgb_feature_importance(model, feature_names):
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='gain', xlabel='Gain', show_values=False)
    plt.title("XGBoost Feature Importance (by Gain)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/xgb_feature_importance.png")
    plt.close()

# -------------------
# 2. SHAP Summary Plot
# -------------------
def plot_shap_summary(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    plt.title("SHAP Summary Plot")
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_plot.png")
    plt.close()

# -------------------
# 3. SHAP Force Plot (Single Prediction)
# -------------------
def save_shap_force_plot(model, X, index=0):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Save as HTML using JS renderer
    force_html = shap.plots.force(shap_values[index]).html()

    with open(f"{OUTPUT_DIR}/shap_force_plot_{index}.html", "w") as f:
        f.write(force_html)


