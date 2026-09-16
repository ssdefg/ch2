import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score

st.set_page_config(page_title="HR Analytics 대시보드", layout="wide")
st.title("👥 HR Analytics: kNN 기반 고성과자 예측 대시보드")

# 1. 데이터 로드
DATA_PATH = 'ch2_knn.csv'

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
    st.success("데이터셋 'ch2_knn.csv' 로드 완료!")
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 데이터 분석", "📈 모델 평가", "🔮 신규 지원자 예측"])

# 전처리 & 모델 학습 캐싱
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['EmpID', 'AftEval'])
    y = data['AftEval']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, X_test, y_test, grid.best_params_

best_model, X_test, y_test, best_params = train_model(df)
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Tab 1: 데이터 개요
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("데이터 미리보기")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("타깃 레이블(AftEval) 분포")
        dist = df['AftEval'].value_counts()
        st.bar_chart(dist)

# Tab 2: 모델 평가 지표 & 차트
with tab2:
    st.subheader(f"최적 파라미터: {best_params}")
    m1, m2, m3 = st.columns(3)
    m1.metric("정확도 (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    m2.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
    m3.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, label='kNN')
    axes[1].plot([0, 1], [0, 1], 'r--')
    axes[1].set_title("ROC Curve")
    st.pyplot(fig)

# Tab 3: 신규 지원자 시뮬레이션
with tab3:
    st.subheader("신규 지원자 정보 입력")
    c1, c2, c3, c4, c5 = st.columns(5)
    gender = c1.selectbox("성별(0:여, 1:남)", [0, 1])
    exp = c2.slider("경력(년)", 0, 10, 2)
    interview = c3.slider("면접 점수", 0, 100, 85)
    skill = c4.slider("코딩테스트 점수", 0, 100, 90)
    pers = c5.slider("인성검사 점수", 0, 100, 80)
    
    input_data = pd.DataFrame([[gender, exp, interview, skill, pers]], 
                              columns=['Gender', 'PreviousExperience', 'InterviewScore', 'SkillScore', 'PersonalityScore'])
    
    if st.button("성과 예측 실행"):
        pred = best_model.predict(input_data)[0]
        prob = best_model.predict_proba(input_data)[0][1] * 100
        
        if pred == 1:
            st.success(f"🎯 **고성과자 예측** (우수인재 확률: {prob:.1f}%) - 채용 추천")
        else:
            st.warning(f"⚠️ **일반/관리대상 예측** (우수인재 확률: {prob:.1f}%) - 채용 신중 검토")
