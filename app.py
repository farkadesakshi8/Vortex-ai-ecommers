from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# Load pre-trained Random Forest Classifier
with open('ecommerce_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load master transactional dataset (31,047 records)
df = pd.read_excel("E-commers sales analysis.xlsx", sheet_name='Sheet1')

# Clean Date column for temporal analytics
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
max_date = df['Date'].max()

# Global KPI Aggregations
def get_global_kpis():
    total_revenue = float(df['Amount'].sum())
    total_orders = int(df['Order ID'].nunique())
    avg_order_val = float(df['Amount'].mean())
    delivered_orders = int((df['Status'] == 'Delivered').sum())
    delivery_rate = round((delivered_orders / len(df)) * 100, 1)
    unique_customers = int(df['Cust ID'].nunique())
    total_states = int(df['State'].nunique())
    
    return {
        'total_revenue': f"₹{total_revenue:,.2f}",
        'total_orders': f"{total_orders:,}",
        'avg_order_val': f"₹{avg_order_val:.2f}",
        'delivery_rate': f"{delivery_rate}%",
        'unique_customers': f"{unique_customers:,}",
        'total_states': total_states,
        'raw_revenue': total_revenue,
        'raw_orders': total_orders
    }

@app.route('/')
def index():
    kpis = get_global_kpis()
    top_channel = df['Channel '].mode()[0]
    top_category = df['Category'].mode()[0]
    
    # State-wise top 5 revenue states
    state_revenue = df.groupby('State')['Amount'].sum().nlargest(5).to_dict()
    
    return render_template('index.html', 
                           kpis=kpis, 
                           top_channel=top_channel, 
                           top_category=top_category,
                           state_revenue=state_revenue)

@app.route('/analytics')
def analytics():
    # 1. Monthly Revenue Progression
    monthly_rev = df.groupby('Month')['Amount'].sum().to_dict()
    
    # 2. Category Sales Distribution
    category_rev = df.groupby('Category')['Amount'].sum().to_dict()
    
    # 3. Channel Distribution
    channel_rev = df.groupby('Channel ')['Amount'].sum().to_dict()
    
    # 4. Age Group Distribution
    age_rev = df.groupby('Age Group')['Amount'].sum().to_dict()

    return render_template('analytics.html', 
                           monthly=monthly_rev, 
                           categories=category_rev, 
                           channels=channel_rev,
                           age_rev=age_rev)

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/customers')
def customers():
    channels = df['Channel '].dropna().unique().tolist()
    statuses = df['Status'].dropna().unique().tolist()
    categories = df['Category'].dropna().unique().tolist()
    
    # Send top 500 records for fast initial DOM render
    records = df[['Cust ID', 'Gender', 'Age', 'Channel ', 'Category', 'Amount', 'Status', 'Date', 'State']].head(500).to_dict(orient='records')
    
    return render_template('customers.html', 
                           customers=records, 
                           channels=channels, 
                           statuses=statuses,
                           categories=categories)

@app.route('/export-csv')
def export_csv():
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='VortexAI_Master_Dataset.csv')

@app.route('/predict-api', methods=['POST'])
def predict_api():
    try:
        data = request.json
        recency = float(data.get('recency', 15))
        frequency = float(data.get('frequency', 5))
        monetary = float(data.get('monetary', 2500))
        qty = float(data.get('qty', 2))
        age = float(data.get('age', 30))
        return_rate = float(data.get('return_rate', 5)) / 100.0

        features = np.array([[recency, frequency, monetary, qty, age, return_rate]])
        
        prediction = int(model.predict(features)[0])
        churn_prob = float(model.predict_proba(features)[0][1] * 100)

        # Dynamic CLV Calculation Formula
        clv = round(monetary * frequency * 1.35, 2)
        
        # Feature impact weights (SHAP approximation)
        recency_impact = round(min(100.0, (recency / 180.0) * 100), 1)
        return_impact = round(return_rate * 100, 1)
        spend_score = round(min(100.0, (monetary / 10000.0) * 100), 1)

        if churn_prob > 65:
            segment = "Critical Churn Hazard"
            recommendation = "Customer displays severe disengagement signals. Immediately issue automated SMS/Email offer with a 25% discount voucher and assign dedicated customer success follow-up."
            priority = "Emergency Intervention"
            badge = "rose"
        elif monetary > 4000 and churn_prob < 35:
            segment = "Quantum Diamond VIP"
            recommendation = "High-value loyal asset. Enroll in exclusive VIP tier perks, early product drops, and custom referral bonus rewards."
            priority = "Secured Diamond Asset"
            badge = "emerald"
        elif recency > 60:
            segment = "Dormant Risk Node"
            recommendation = "Shopping activity has slowed down. Deploy automated win-back email sequence showcasing personalized top-selling categories."
            priority = "Re-Engagement Phase"
            badge = "amber"
        else:
            segment = "Active Growth Shopper"
            recommendation = "Customer engagement is steady. Trigger cross-sell recommendations in adjacent categories to boost basket size."
            priority = "Stable Expansion"
            badge = "cyan"

        return jsonify({
            'success': True,
            'prediction': prediction,
            'churn_risk': round(churn_prob, 1),
            'segment': segment,
            'recommendation': recommendation,
            'priority': priority,
            'badge': badge,
            'clv': f"₹{clv:,.2f}",
            'recency_impact': recency_impact,
            'return_impact': return_impact,
            'spend_score': spend_score
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)