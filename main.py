
from pipeline import FraudDetectionPipeline

if __name__ == "__main__":
    pipeline = FraudDetectionPipeline(data_path="creditcard.csv")
    best_model = pipeline.run_complete_pipeline(resampling_method='smote')

    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print("✅ Dataset loaded and explored")
    print("✅ Data preprocessed and split")
    print("✅ Class imbalance handled with SMOTE")
    print("✅ Multiple models trained and evaluated")
    print("✅ Performance metrics calculated")
    print("✅ Visualizations created")
    print("✅ Feature importance analyzed")
    print("✅ Anomaly detection models evaluated")

    print(f"\n🏆 Best Model: {best_model[0] if best_model else 'N/A'}")

    print("\n📊 Key Insights for Fraud Detection:")
    print("• Recall is crucial - we want to catch fraudulent transactions")
    print("• Precision matters - avoid flagging too many legitimate transactions")
    print("• F1-score provides good balance between precision and recall")
    print("• PR-AUC is better than ROC-AUC for imbalanced datasets")
    print("• Ensemble methods often perform better")
    print("• Feature importance helps understand fraud patterns")
    