import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# Customer segments data
segments = {
    "Champions": {"count": 15, "rfm": (5, 5, 5), "color": "#00ffb2"},
    "Loyal Customers": {"count": 23, "rfm": (4, 4, 3), "color": "#00cc8f"},
    "Potential Loyalists": {"count": 31, "rfm": (3, 3, 4), "color": "#7b61ff"},
    "At Risk": {"count": 18, "rfm": (2, 2, 4), "color": "#ff4d6d"},
    "Lost Customers": {"count": 13, "rfm": (1, 1, 2), "color": "#ffbe0b"}
}

model_metrics = {"XGBoost": {"auc": 0.998, "accuracy": 98, "f1": 0.96}, "Random Forest": {"auc": 0.996, "accuracy": 97, "f1": 0.94}}

def draw_rfm_matrix():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0f1320')
    ax.set_facecolor('#0f1320')
    for seg, data in segments.items():
        r, f, m = data['rfm']
        ax.bar(seg, data['count'], color=data['color'], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.text(seg, data['count'] + 1, f"R:{r} F:{f} M:{m}", ha='center', fontsize=9, color='white', fontweight='bold')
    ax.set_xlabel('Customer Segment', color='white', fontsize=11)
    ax.set_ylabel('Number of Customers', color='white', fontsize=11)
    ax.set_title('RFM-Based Customer Segmentation', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#ffffff20')
    ax.set_facecolor('#0f1320')
    plt.tight_layout()
    return fig

def draw_model_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor('#0f1320')
    ax.set_facecolor('#0f1320')
    models = list(model_metrics.keys())
    metrics = ['AUC-ROC', 'Accuracy', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    for i, (model, vals) in enumerate(model_metrics.items()):
        values = [vals['auc'], vals['accuracy']/100, vals['f1']]
        bars = ax.bar(x + i*width, values, width, label=model, color=['#00ffb2', '#7b61ff'][i], alpha=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}' if v < 1 else f'{v}%', ha='center', fontsize=9, color='white')
    ax.set_xlabel('Metric', color='white', fontsize=11)
    ax.set_ylabel('Score', color='white', fontsize=11)
    ax.set_title('Model Performance Comparison', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metrics, color='white')
    ax.legend(facecolor='#0f1320', edgecolor='#ffffff20', labelcolor='white')
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#ffffff20')
    plt.tight_layout()
    return fig

def draw_cltv_tier():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('#0f1320')
    ax.set_facecolor('#0f1320')
    tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
    counts = [35, 28, 22, 15]
    colors = ['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2']
    explode = (0, 0, 0.05, 0.1)
    wedges, texts, autotexts = ax.pie(counts, explode=explode, labels=tiers, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
    for t in texts: t.set_fontweight('bold')
    ax.set_title('CLTV Tier Distribution', color='white', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig

def predict_cltv(recency, frequency, monetary):
    # Simple scoring model
    r_score = min(5, max(1, 6 - recency // 20))
    f_score = min(5, max(1, frequency // 10))
    m_score = min(5, max(1, monetary // 500))
    total = (r_score + f_score + m_score) / 15 * 100

    tier = "Platinum" if total > 85 else "Gold" if total > 65 else "Silver" if total > 40 else "Bronze"
    risk = "Low" if r_score > 3 else "Medium" if r_score > 1 else "High"

    return f"""
## 🎯 CLTV Prediction Results

### Customer Profile
| Metric | Value |
|--------|-------|
| Recency (days since last purchase) | {recency} |
| Frequency (number of transactions) | {frequency} |
| Monetary (total spending ₹) | {monetary} |

### RFM Scores
- **R (Recency):** {r_score}/5 {'⭐' * r_score}
- **F (Frequency):** {f_score}/5 {'⭐' * f_score}
- **M (Monetary):** {m_score}/5 {'⭐' * m_score}

### CLTV Classification
- **Tier:** {tier}
- **Score:** {total:.1f}%
- **Churn Risk:** {risk}
- **Expected Value:** ₹{monetary * (frequency + 1) * (1 if r_score > 2 else 0.7):.0f}

### Recommendations
{"🎁 VIP treatment - exclusive offers" if tier == "Platinum" else "📈 Upsell opportunities - premium tiers" if tier == "Gold" else "📧 Re-engagement campaigns" if risk == "High" else "✅ Regular engagement - maintain satisfaction"}
"""

css = """
body, .gradio-container { background: #080b12 !important; }
footer { display: none !important; }
textarea, input[type='text'], input[type='number'] { background: #161c2e !important; border: 1px solid #ffffff1a !important; border-radius: 12px !important; color: #dde3f0 !important; }
.gr-button-primary { background: linear-gradient(135deg, #00ffb2, #00cc8f) !important; border: none !important; border-radius: 12px !important; color: #040810 !important; font-weight: 600 !important; }
.markdown-wrap { background: #161c2e !important; border-radius: 12px; padding: 20px; }
"""

with gr.Blocks(css=css, title="CLTV Prediction Demo") as demo:
    gr.HTML("""
    <div style="background:linear-gradient(90deg,#080b12,#0d1220);border-bottom:1px solid #ffffff0f;padding:16px 28px;display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:12px">
        <div style="width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#00ffb2,#7b61ff);display:flex;align-items:center;justify-content:center;font-size:16px">📊</div>
        <div>
          <div style="font-size:15px;font-weight:600;color:#dde3f0">CLTV Prediction Demo</div>
          <div style="font-size:10px;color:#5a6280;font-family:monospace">RFM Analysis · XGBoost · Customer Segmentation</div>
        </div>
      </div>
      <div style="display:flex;gap:10px">
        <span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:20px;border:1px solid #00ffb260;color:#00ffb2">AUC-ROC: 0.998</span>
        <span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:20px;border:1px solid #7b61ff60;color:#7b61ff">Accuracy: 98%</span>
      </div>
    </div>
    """)

    with gr.Tab("Customer Segments"):
        gr.Plot(draw_rfm_matrix, label="RFM Matrix")

    with gr.Tab("Model Performance"):
        gr.Plot(draw_model_comparison, label="Model Comparison")

    with gr.Tab("CLTV Tiers"):
        gr.Plot(draw_cltv_tier, label="Tier Distribution")

    with gr.Tab("Predict CLTV"):
        gr.Markdown("### Enter Customer Data")
        with gr.Row():
            recency = gr.Number(label="Recency (days since last purchase)", value=30)
            frequency = gr.Number(label="Frequency (number of transactions)", value=50)
            monetary = gr.Number(label="Monetary (total spending ₹)", value=10000)
        predict_btn = gr.Button("Predict CLTV", variant="primary")
        output = gr.Markdown()

        predict_btn.click(fn=predict_cltv, inputs=[recency, frequency, monetary], outputs=output)

demo.launch()