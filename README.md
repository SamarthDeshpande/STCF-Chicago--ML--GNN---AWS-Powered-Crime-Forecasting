Count Forecasting (Beat x Day) — Local

Generated: 2025-08-16T19:17:31.585326Z
Split date: 2022-05-29

Models trained: TweediePoisson, HGBPoisson, XGBPoisson
Best model (integer MAE): HGBPoisson
Best integer MAE: 0.830

Beat reweighting: MAE(naive roll7) x inverse-frequency (hard beats get higher weight)
Quantile bands: alpha_low=0.2, alpha_high=0.8

Features used (Stage-aligned):
['Crime_Lag_1Day_Beat', 'Crime_Lag_3Day_Beat', 'Crime_Lag_7Day_Beat', 'Crime_Lag_14Day_Beat', 'Crime_Lag_21Day_Beat', 'Rolling_Avg_3Day_Beat', 'Rolling_Avg_7Day_Beat', 'Rolling_Avg_14Day_Beat', 'Rolling_Avg_28Day_Beat', 'Weekday', 'IsWeekend', 'Month', 'Season', 'DoW_sin', 'DoW_cos', 'Mon_sin', 'Mon_cos', 'IsHoliday', 'Hour_Cluster', 'Lat_Bin', 'Lng_Bin', 'Embed_0', 'Embed_1', 'Embed_2', 'Embed_3', 'Embed_4', 'Embed_5', 'Embed_6', 'Embed_7']

Files saved in _out:
- beat_day_features.parquet (full features table for dashboards / downstream models)
- model_results_test.csv
- beat_day_test_actuals_preds.csv (per-row test predictions + quantiles)
- per_beat_metrics.csv
- actual_vs_pred_best.png
- error_distribution_best.png
- worst_beats_mae.png
- forecast_7d_per_beat.csv
- citywide_forecast.png
- citywide_actual_test.csv
- count_models.pkl
- count_feature_cols.pkl
- MANIFEST.json
- summary.txt