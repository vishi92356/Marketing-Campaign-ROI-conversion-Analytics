import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm

st.set_page_config(page_title="Marketing Campaign ROI Analytics", layout="wide")

st.title("📊 Marketing Campaign ROI & Conversion Analytics App")

# =========================
# DATA GENERATION
# =========================
@st.cache_data
def generate_dataset(n=1000):
    np.random.seed(42)

    channels = ['Facebook', 'Google', 'Instagram', 'Email', 'YouTube']
    regions = ['North', 'South', 'East', 'West']

    data = pd.DataFrame({
        'campaign_id': range(1, n+1),
        'channel': np.random.choice(channels, n),
        'region': np.random.choice(regions, n),
        'spend': np.random.uniform(1000, 10000, n),
        'clicks': np.random.randint(100, 5000, n),
        'leads': np.random.randint(50, 1000, n),
        'promo_code': np.random.choice(['PROMO10', 'SAVE20', 'NONE'], n)
    })

    data['conversions'] = (data['leads'] * np.random.uniform(0.1, 0.4, n)).astype(int)
    data['revenue'] = data['conversions'] * np.random.uniform(20, 150, n)

    return data

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Marketing Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = generate_dataset()
    st.info("Using Auto-Generated Sample Dataset (1000 rows)")

# =========================
# DATA CLEANING
# =========================
st.header("1️⃣ Data Cleaning")

df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

# =========================
# FEATURE ENGINEERING
# =========================
st.header("2️⃣ Feature Engineering")

df['ROI'] = (df['revenue'] - df['spend']) / df['spend']
df['CAC'] = df['spend'] / df['conversions'].replace(0, 1)
df['Conversion_Rate'] = df['conversions'] / df['clicks']
df['Promo_Used'] = df['promo_code'].apply(lambda x: 0 if x == "NONE" else 1)

# Regex Example
df['Valid_Promo'] = df['promo_code'].apply(
    lambda x: 1 if re.match(r'^[A-Z]+\d+$', str(x)) else 0
)

st.dataframe(df.head())

# =========================
# NORMALIZATION
# =========================
st.header("3️⃣ Normalization")

scaler = StandardScaler()
num_cols = ['spend', 'clicks', 'leads', 'conversions', 'revenue']

df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

st.write("Normalized Sample:")
st.dataframe(df_scaled.head())

# =========================
# BUSINESS QUESTION 1
# =========================
st.header("📌 Which channel generates highest ROI?")

roi_channel = df.groupby('channel')['ROI'].mean().sort_values(ascending=False)
st.write(roi_channel)

fig, ax = plt.subplots()
roi_channel.plot(kind='bar', ax=ax)
plt.title("Average ROI by Channel")
st.pyplot(fig)

# =========================
# BUSINESS QUESTION 2
# =========================
st.header("📌 Do promo-code customers convert better?")

promo = df[df['Promo_Used']==1]['Conversion_Rate']
nonpromo = df[df['Promo_Used']==0]['Conversion_Rate']

t_stat, p_value = stats.ttest_ind(promo, nonpromo)

st.write("Promo Avg Conversion:", promo.mean())
st.write("Non-Promo Avg Conversion:", nonpromo.mean())
st.write("T-test p-value:", p_value)

if p_value < 0.05:
    st.success("Promo users convert significantly better!")
else:
    st.warning("No significant difference detected.")

# =========================
# BUSINESS QUESTION 3
# =========================
st.header("📌 Overspending Campaigns")

overspend = df[(df['spend'] > df['revenue'])]
st.write("Number of Overspending Campaigns:", overspend.shape[0])
st.dataframe(overspend.head())

# =========================
# BUSINESS QUESTION 4
# =========================
st.header("📌 Acquisition Cost across Channels")

cac_channel = df.groupby('channel')['CAC'].mean()
st.write(cac_channel)

fig2, ax2 = plt.subplots()
cac_channel.plot(kind='bar', ax=ax2)
plt.title("CAC by Channel")
st.pyplot(fig2)

# =========================
# BUSINESS QUESTION 5
# =========================
st.header("📌 Best Performing Region")

region_roi = df.groupby('region')['ROI'].mean()
st.write(region_roi)

fig3, ax3 = plt.subplots()
region_roi.plot(kind='bar', ax=ax3)
plt.title("ROI by Region")
st.pyplot(fig3)

# =========================
# REGRESSION MODEL
# =========================
st.header("📈 Revenue Prediction Model")

X = df[['spend','clicks','leads','Promo_Used']]
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

score = model.score(X_test,y_test)

st.write("Model R² Score:", score)

# Statsmodels summary
X2 = sm.add_constant(X)
est = sm.OLS(y, X2).fit()
st.text(est.summary())

st.success("Application Ready for Deployment 🚀")
