from data_processing import load_csv_features
from training import train_surrogate_model
from visualization import plot_target_distribution, plot_feature_correlation, plot_pca_projection
from visualization import plot_xgb_feature_importance, plot_shap_summary, save_shap_force_plot
from optimizer import run_drag_optimization
from vae_training import train_vae, generate_new_designs


X, y = load_csv_features("combined_data.csv")
model, rmse = train_surrogate_model(X, y)
print(f"Test RMSE: {rmse:.4f}")

plot_target_distribution(y)
plot_feature_correlation(X, y)
plot_pca_projection(X, y)
plot_xgb_feature_importance(model, X.columns)
plot_shap_summary(model, X)
save_shap_force_plot(model, X, index=5)
optimized_design, optimized_cd = run_drag_optimization(model, X)
print("Optimized Cd:", optimized_cd)
print(optimized_design)
vae_model = train_vae(X, epochs=100)
df_generated = generate_new_designs(vae_model, num_samples=10, latent_dim=4, feature_names=X.columns.tolist())