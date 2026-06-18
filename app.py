# -*- coding: utf-8 -*-
"""
📦 식료품 폐기 리스크 예측 — Streamlit 데모
재고 정보를 입력하면 '유통기한 내 예상 판매율'을 회귀로 예측하고
폐기 위험 등급과 권장 액션을 제시합니다.

※ 데모용: 실제 프로젝트는 1,000+ SKU 데이터로 스태킹 앙상블(MAE 0.0337)을 학습했으나,
   데이터는 용량 관계로 제외되어, 본 앱은 동일한 관계식 기반 합성 데이터로 모델을 즉석 학습합니다.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="식료품 폐기 리스크 예측", page_icon="📦", layout="centered")


@st.cache_resource
def train_demo_model():
    """관계식 기반 합성 데이터로 스태킹 회귀 모델 학습(데모)."""
    rng = np.random.default_rng(42)
    n = 4000
    qty = rng.integers(10, 500, n).astype(float)        # 재고량
    daily = rng.uniform(0.1, 30, n)                      # 일평균 판매
    shelf = rng.integers(1, 180, n).astype(float)        # 남은 유통기한(일)
    unit_cost = rng.uniform(0.5, 50, n)                  # 단가
    # 타깃: 유통기한 내 예상 판매율 (낮을수록 폐기 위험)
    sell_rate = (daily * shelf) / np.maximum(qty, 1)
    sell_rate = np.clip(sell_rate + rng.normal(0, 0.05, n), 0, 2.0)
    X = np.column_stack([qty, daily, shelf, unit_cost])
    y = sell_rate
    Xtr, Xte, ytr, yte = X[:3200], X[3200:], y[:3200], y[3200:]
    stack = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(random_state=42)),
        ],
        final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1,
    )
    stack.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, stack.predict(Xte))
    return stack, mae


st.title("📦 식료품 폐기 리스크 예측")
st.caption("재고 데이터로 '유통기한 내 예상 판매율'을 예측해 폐기 위험을 선제 식별 · KDT 팀 프로젝트(본인=회귀·스태킹 담당)")

with st.expander("ℹ️ 이 데모에 대하여", expanded=False):
    st.markdown(
        "- 실제 프로젝트: 1,000+ SKU·50+ 피처로 **5개 회귀모델 비교 + 스태킹 앙상블(메타 Ridge)** → **MAE 0.0337**\n"
        "- 본 앱: 데이터 용량 관계로 동일 관계식 기반 **합성 데이터로 즉석 학습**한 데모입니다.\n"
        "- 원리: 예상 판매율 = (일판매 × 남은 유통기한) / 재고량 → 낮을수록 폐기 위험."
    )

model, demo_mae = train_demo_model()

st.subheader("재고 정보 입력")
c1, c2 = st.columns(2)
with c1:
    qty = st.slider("재고량 (개)", 10, 500, 200, 10)
    daily = st.slider("일평균 판매량 (개/일)", 0.1, 30.0, 5.0, 0.1)
with c2:
    shelf = st.slider("남은 유통기한 (일)", 1, 180, 30, 1)
    unit_cost = st.slider("단가 ($)", 0.5, 50.0, 10.0, 0.5)

pred = float(model.predict([[qty, daily, shelf, unit_cost]])[0])
waste_risk = max(0.0, 1.0 - min(pred, 1.0))   # 0~1 (높을수록 위험)

st.subheader("예측 결과")
m1, m2, m3 = st.columns(3)
m1.metric("예상 판매율", f"{pred:.2f}")
m2.metric("폐기 위험도", f"{waste_risk*100:.0f}%")
expected_waste = int(max(0, qty - daily * shelf))
m3.metric("예상 폐기 수량", f"{expected_waste} 개")

if waste_risk >= 0.6:
    st.error(f"🔴 **고위험** — 폐기 위험 {waste_risk*100:.0f}%. 즉시 할인/프로모션 권장 (예: {min(50, int(waste_risk*60))}% 할인).")
elif waste_risk >= 0.3:
    st.warning(f"🟡 **중위험** — 폐기 위험 {waste_risk*100:.0f}%. 동적 가격제·번들 판매 검토.")
else:
    st.success(f"🟢 **저위험** — 폐기 위험 {waste_risk*100:.0f}%. 정상 판매로 소진 가능.")

st.divider()
st.caption(f"데모 모델 검증 MAE ≈ {demo_mae:.3f} · 실제 프로젝트 스태킹 MAE 0.0337 · 회귀모델 5종 비교 + 스태킹(Ridge 메타)")
st.caption("GitHub: grocery-waste-ml · 본인 역할: 폐기 위험도 회귀·스태킹 앙상블 (분류·수요예측은 팀원 담당)")
