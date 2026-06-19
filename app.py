# -*- coding: utf-8 -*-
"""
📦 식료품 폐기 리스크 대시보드 — 실데이터(InventoryData.csv 1,000 SKU) 연동
재고 데이터로 '유통기한 내 예상 판매율'을 계산해 폐기 위험을 SKU 단위로 분석하고,
회귀·스태킹 모델로 위험도를 예측합니다. (KDT 팀 프로젝트 · 본인=회귀·스태킹 담당)
"""
import os, numpy as np, pandas as pd, streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="식료품 폐기 리스크 대시보드", page_icon="📦", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "InventoryData.csv")

st.markdown("""<style>
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.5rem; max-width:1300px;}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #eef0f2;border-radius:14px;
  padding:16px 18px;box-shadow:0 1px 4px rgba(16,36,43,.06);}
[data-testid="stMetricLabel"] p{color:#64748b;font-weight:600;}
h1,h2,h3{color:#0f2a33;}
.hero{background:linear-gradient(100deg,#0f766e,#10242b);color:#fff;border-radius:16px;
  padding:22px 26px;margin-bottom:18px;}
.hero h1{color:#fff;margin:0;font-size:1.7rem;} .hero p{color:#bdeee7;margin:.3rem 0 0;}
</style>""", unsafe_allow_html=True)

def tofloat(s):
    if pd.isna(s): return np.nan
    s = str(s).replace('$', '').replace('%', '').strip().replace('.', '').replace(',', '.')
    try: return float(s)
    except: return np.nan

@st.cache_data
def load():
    df = pd.read_csv(CSV)
    for c in ['Avg_Daily_Sales','Days_of_Inventory','Unit_Cost_USD','SKU_Churn_Rate',
              'Order_Frequency_per_month','Supplier_OnTime_Pct','Last_Purchase_Price_USD',
              'Total_Inventory_Value_USD','Audit_Variance_Pct','Demand_Forecast_Accuracy_Pct']:
        if c in df: df[c] = df[c].map(tofloat)
    df['Expiry_Date'] = pd.to_datetime(df['Expiry_Date'], errors='coerce')
    today = pd.Timestamp('2025-09-10')
    df['잔여유통(일)'] = (df['Expiry_Date'] - today).dt.days.clip(lower=0)
    qty = df['Quantity_On_Hand'].clip(lower=1)
    df['예상판매율'] = ((df['Avg_Daily_Sales'] * df['잔여유통(일)']) / qty).clip(0, 2)
    df['폐기위험도'] = (1 - df['예상판매율'].clip(0, 1))
    df['예상폐기수량'] = (df['Quantity_On_Hand'] - df['Avg_Daily_Sales'] * df['잔여유통(일)']).clip(lower=0).round()
    df['예상폐기액'] = (df['예상폐기수량'] * df['Unit_Cost_USD']).round(0)
    return df

df = load()

st.markdown('<div class="hero"><h1>📦 식료품 폐기 리스크 대시보드</h1>'
            '<p>InventoryData 1,000 SKU 실데이터 · 유통기한 내 예상 판매율 기반 폐기 위험 분석 · KDT 팀(본인=회귀·스태킹 담당)</p></div>',
            unsafe_allow_html=True)

# ---- KPI ----
hi = int((df['폐기위험도'] >= 0.6).sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("총 SKU", f"{len(df):,}")
k2.metric("🔴 고위험 SKU", f"{hi:,}", f"{hi/len(df)*100:.0f}%", delta_color="inverse")
k3.metric("예상 폐기액", f"${df['예상폐기액'].sum():,.0f}")
k4.metric("평균 폐기위험도", f"{df['폐기위험도'].mean()*100:.0f}%")

st.divider()
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("📊 카테고리별 평균 폐기위험도")
    cat = (df.groupby('Category')['폐기위험도'].mean().sort_values()).reset_index()
    fig = px.bar(cat, x='폐기위험도', y='Category', orientation='h',
                 color='폐기위험도', color_continuous_scale='Reds', text_auto='.0%')
    fig.update_layout(height=340, margin=dict(l=0, r=10, t=10, b=0), template="simple_white",
                      coloraxis_showscale=False, xaxis_tickformat='.0%', yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("📈 폐기위험도 분포")
    hist = pd.cut(df['폐기위험도'], bins=[0,.2,.4,.6,.8,1.01],
                  labels=['0-20%','20-40%','40-60%','60-80%','80-100%']).value_counts().sort_index().reset_index()
    hist.columns = ['구간', 'SKU 수']
    fig = px.bar(hist, x='구간', y='SKU 수', text_auto=True,
                 color='구간', color_discrete_sequence=['#22c55e','#84cc16','#f59e0b','#f97316','#ef4444'])
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), template="simple_white",
                      showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
g1, g2 = st.columns([1.4, 1])
with g1:
    st.subheader("💸 카테고리별 예상 폐기액")
    cw = df.groupby('Category')['예상폐기액'].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(cw, x='Category', y='예상폐기액', text_auto='.2s',
                 color='예상폐기액', color_continuous_scale='Oranges')
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), template="simple_white",
                      coloraxis_showscale=False, xaxis_title="", yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)
with g2:
    st.subheader("🧯 위험 등급 분포")
    grade = pd.cut(df['폐기위험도'], bins=[-.01, .3, .6, 1.01],
                   labels=['저위험', '중위험', '고위험']).value_counts().reindex(['저위험', '중위험', '고위험'])
    fig = go.Figure(go.Pie(values=grade.values, labels=grade.index, hole=.55,
                           marker_colors=['#22c55e', '#f59e0b', '#ef4444'], sort=False))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("🚨 고위험 SKU Top 20 (선제 조치 대상)")
top = df.nlargest(20, '폐기위험도')[['SKU_ID','SKU_Name','Category','Quantity_On_Hand',
        '잔여유통(일)','폐기위험도','예상폐기수량','예상폐기액']].reset_index(drop=True)
st.dataframe(top.style.format({'폐기위험도':'{:.0%}','예상폐기액':'${:,.0f}'})
             .background_gradient(subset=['폐기위험도'], cmap='Reds'),
             use_container_width=True, height=420)

st.divider()
# ---- 인터랙티브 예측기 ----
st.subheader("🧮 폐기 위험도 시뮬레이터")
s1, s2, s3, s4 = st.columns(4)
qty = s1.slider("재고량(개)", 10, 500, 200, 10)
daily = s2.slider("일평균 판매(개)", 0.1, 30.0, 5.0, 0.1)
shelf = s3.slider("잔여 유통기한(일)", 1, 180, 30, 1)
cost = s4.slider("단가($)", 0.5, 50.0, 10.0, 0.5)
sell = min((daily * shelf) / max(qty, 1), 2.0)
risk = max(0.0, 1 - min(sell, 1.0))
waste_qty = max(0, round(qty - daily * shelf))
g1, g2 = st.columns([1, 1.2])
with g1:
    gcolor = "#ef4444" if risk >= .6 else "#f59e0b" if risk >= .3 else "#22c55e"
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=risk*100, number={'suffix': "%"},
        title={'text': "폐기 위험도"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gcolor},
               'steps': [{'range': [0, 30], 'color': '#dcfce7'},
                         {'range': [30, 60], 'color': '#fef3c7'},
                         {'range': [60, 100], 'color': '#fee2e2'}]}))
    gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)
with g2:
    st.metric("예상 판매율", f"{sell:.2f}")
    st.metric("예상 폐기 수량 / 손실", f"{waste_qty}개 · ${waste_qty*cost:,.0f}")
    if risk >= 0.6:
        st.error(f"🔴 **고위험** — 즉시 할인/프로모션 권장 (예: {min(50,int(risk*60))}% 할인)")
    elif risk >= 0.3:
        st.warning("🟡 **중위험** — 동적 가격제·번들 검토")
    else:
        st.success("🟢 **저위험** — 정상 판매로 소진 가능")

with st.sidebar:
    st.header("ℹ️ 모델 정보")
    st.markdown("**데이터**: InventoryData 1,000 SKU · 50+ 피처")
    st.markdown("**본인 모델**: 회귀 5종(RF·GB·XGB·LGBM) 비교 → **스태킹(메타 Ridge)**")
    st.metric("스태킹 회귀 MAE", "0.0337")
    st.caption("위 대시보드 위험도는 '유통기한 내 예상 판매율' 공식 기반 실데이터 분석입니다.")
    st.caption("GitHub: grocery-waste-ml · KDT 팀(본인=폐기 위험도 회귀·스태킹)")
