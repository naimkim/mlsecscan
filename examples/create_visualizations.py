"""
OTA Model Visualization - Generate charts for Figma dashboard
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle

# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_visualizations(predictor, data, save_dir='visualizations'):
    """Generate all visualizations for OTA dashboard"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    X = data[predictor.feature_names]
    y = data['update_success']
    X_scaled = predictor.scaler.transform(X)
    y_pred = predictor.model.predict(X_scaled)
    y_proba = predictor.model.predict_proba(X_scaled)[:, 1]
    
    # 1. Feature Importance Chart
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'feature': predictor.feature_names,
        'importance': predictor.model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'], color='#4A90E2')
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('OTA Update Success - Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: feature_importance.png")
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAIL', 'SUCCESS'],
                yticklabels=['FAIL', 'SUCCESS'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.title('OTA Update Prediction - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrix.png")
    plt.close()
    
    # 3. ROC Curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='#E94B3C', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - OTA Update Success Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: roc_curve.png")
    plt.close()
    
    # 4. Risk Distribution
    plt.figure(figsize=(10, 6))
    risk_counts = data['risk_level'].value_counts().sort_index()
    colors = ['#6BCF7F', '#FFB84D', '#E94B3C']
    labels = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
    
    plt.bar(labels, risk_counts.values, color=colors, edgecolor='white', linewidth=2)
    plt.ylabel('Number of Vehicles', fontsize=12, fontweight='bold')
    plt.title('OTA Update Risk Distribution', fontsize=14, fontweight='bold')
    for i, v in enumerate(risk_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/risk_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: risk_distribution.png")
    plt.close()
    
    # 5. Battery Health vs Success Rate
    plt.figure(figsize=(10, 6))
    battery_bins = pd.cut(data['battery_soc_percent'], bins=[0, 30, 50, 70, 100], 
                          labels=['<30%', '30-50%', '50-70%', '>70%'])
    success_by_battery = data.groupby(battery_bins)['update_success'].mean()
    
    plt.bar(range(len(success_by_battery)), success_by_battery.values, 
            color=['#E94B3C', '#FFB84D', '#FFD700', '#6BCF7F'], edgecolor='white', linewidth=2)
    plt.xticks(range(len(success_by_battery)), success_by_battery.index)
    plt.ylabel('Success Rate', fontsize=12, fontweight='bold')
    plt.xlabel('Battery State of Charge', fontsize=12, fontweight='bold')
    plt.title('OTA Update Success Rate by Battery Level', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    for i, v in enumerate(success_by_battery.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/battery_success_rate.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: battery_success_rate.png")
    plt.close()
    
    # 6. Network Quality Impact
    plt.figure(figsize=(10, 6))
    network_bins = pd.cut(data['network_signal_strength'], 
                          bins=[-120, -95, -85, -75, -40],
                          labels=['Poor', 'Fair', 'Good', 'Excellent'])
    success_by_network = data.groupby(network_bins)['update_success'].mean()
    
    colors_network = ['#E94B3C', '#FFB84D', '#A8D08D', '#6BCF7F']
    plt.bar(range(len(success_by_network)), success_by_network.values, 
            color=colors_network, edgecolor='white', linewidth=2)
    plt.xticks(range(len(success_by_network)), success_by_network.index)
    plt.ylabel('Success Rate', fontsize=12, fontweight='bold')
    plt.xlabel('Network Signal Quality', fontsize=12, fontweight='bold')
    plt.title('OTA Update Success Rate by Network Quality', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    for i, v in enumerate(success_by_network.values):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/network_success_rate.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: network_success_rate.png")
    plt.close()
    
    # 7. Model Performance Summary (for dashboard card)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('OTA Update Prediction Model - Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Accuracy meter
    ax1 = axes[0, 0]
    accuracy = (y == y_pred).mean()
    ax1.text(0.5, 0.5, f'{accuracy:.1%}', 
             ha='center', va='center', fontsize=48, fontweight='bold', color='#4A90E2')
    ax1.text(0.5, 0.2, 'Overall Accuracy', 
             ha='center', va='center', fontsize=14, color='gray')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Samples processed
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.5, f'{len(data):,}', 
             ha='center', va='center', fontsize=48, fontweight='bold', color='#6BCF7F')
    ax2.text(0.5, 0.2, 'Vehicles Analyzed', 
             ha='center', va='center', fontsize=14, color='gray')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Success rate
    ax3 = axes[1, 0]
    success_rate = y.mean()
    ax3.text(0.5, 0.5, f'{success_rate:.1%}', 
             ha='center', va='center', fontsize=48, fontweight='bold', color='#FFB84D')
    ax3.text(0.5, 0.2, 'Update Success Rate', 
             ha='center', va='center', fontsize=14, color='gray')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # High risk count
    ax4 = axes[1, 1]
    high_risk = (data['risk_level'] == 2).sum()
    ax4.text(0.5, 0.5, f'{high_risk:,}', 
             ha='center', va='center', fontsize=48, fontweight='bold', color='#E94B3C')
    ax4.text(0.5, 0.2, 'High Risk Vehicles', 
             ha='center', va='center', fontsize=14, color='gray')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dashboard_summary.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"✓ Saved: dashboard_summary.png")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to '{save_dir}/' directory")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  1. feature_importance.png - Feature importance chart")
    print("  2. confusion_matrix.png - Prediction accuracy matrix")
    print("  3. roc_curve.png - ROC curve analysis")
    print("  4. risk_distribution.png - Risk level distribution")
    print("  5. battery_success_rate.png - Battery impact analysis")
    print("  6. network_success_rate.png - Network quality impact")
    print("  7. dashboard_summary.png - Key metrics dashboard")
    print("\nUse these images in your Figma dashboard design!")


# Usage example
if __name__ == '__main__':
    # Load the trained model and data
    from ota_advanced import OTAUpdatePredictor
    
    predictor = OTAUpdatePredictor()
    data = predictor.generate_synthetic_data(n_samples=5000)
    predictor.train(data)
    
    # Generate all visualizations
    create_visualizations(predictor, data, save_dir='ota_visualizations')