Customer Churn Analytics Dashboard

An interactive analytics dashboard built with Streamlit and Plotly to analyze customer churn patterns in a telecom dataset.

Features
- KPI metrics: churn rate, revenue at risk, avg tenure
- Churn drivers: contract type, tenure group, internet service
- Risk segmentation with scoring model
- Retention insights: add-on services, payment methods
- High-risk customer table
- Sidebar filters for dynamic exploration

Tech Stack
- **Python** · **Pandas** · **Plotly** · **Streamlit**

Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the root folder.

Deploy on Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → Deploy