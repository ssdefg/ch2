# ==============================================================================
# [HR Analytics] kNN 채용 성과 예측 및 의사결정 인터랙티브 대시보드 (app.py)
# ==============================================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ------------------------------------------------------------------------------
# 1. 대시보드 페이지 설정 및 CSS 스타일
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Analytics: kNN 채용 성과 예측 대시보드",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .callout-success { background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 15px; border-radius: 8px; }
    .callout-warning { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. 데이터 로드 함수 (캐싱 및 다중 경로 탐색)
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset(file_source=None):
    if file_source is not None:
        return pd.read_csv(file_source)
    
    candidate_paths = [
        'ch2_knn.csv',
        'data/ch2_knn.csv',
        '../data/ch2_knn.csv',
        '/.agents/workspace/ch2_knn.csv'
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


# ------------------------------------------------------------------------------
# 3. 사이드바 구성 및 메뉴 내비게이션
# ------------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/conference-call.png", width=70)
st.sidebar.title("피플 애널리틱스 센터")
st.sidebar.caption("M사 채용 전형 kNN 예측 모델링")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 채용 데이터 업로드 (CSV)", type=['csv'])
df = load_dataset(uploaded_file)

if df is None:
    st.error("데이터셋('ch2_knn.csv')을 찾을 수 없습니다. GitHub 저장소에 csv 파일을 올리거나, 사이드바에서 파일을 직접 업로드해주세요.")
    st.stop()

menu = st.sidebar.radio(
    "📌 대시보드 메뉴 이동",
    [
        "1. 데이터 탐색 및 EDA",
        "2. kNN 모델 튜닝 및 학습",
        "3. 모델 평가 및 채용 오차 분석",
        "4. 실시간 신규 지원자 성과 예측"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 모델 배경 정보**
- **모집단**: 입사 1년 차 재직자 1,147명
- **Target**: AftEval (1: 고성과자, 0: 저성과자)
- **전형 지표**: 면접, 코딩테스트, 인성검사, 경력, 성별
""")


# ==============================================================================
# MENU 1: 데이터 탐색 및 EDA
# ==============================================================================
if menu == "1. 데이터 탐색 및 EDA":
    st.markdown('<div class="main-header">👥 채용 전형 데이터 탐색 및 통계 (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">채용 전형 점수와 입사 1년 후 성과(AftEval) 간의 관계를 시각적으로 분석합니다.</div>', unsafe_allow_html=True)

    total_count = len(df)
    high_perf_count = int((df['AftEval'] == 1).sum())
    low_perf_count = int((df['AftEval'] == 0).sum())
    high_perf_ratio = (high_perf_count / total_count) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 지원자 수", f"{total_count:,} 명")
    col2.metric("우수 고성과자 (1)", f"{high_perf_count:,} 명", f"{high_perf_ratio:.1f}%")
    col3.metric("일반/저성과자 (0)", f"{low_perf_count:,} 명", f"{100 - high_perf_ratio:.1f}%")
    col4.metric("전형 독립변수", f"{len(df.columns) - 2} 개", "EmpID 제외")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 데이터셋 미리보기", "📊 전형 점수별 성과 분포", "🔥 상관관계 히트맵"])

    with tab1:
        st.subheader("데이터셋 샘플 (상위 10개 행)")
        st.dataframe(df.head(10), use_container_width=True)

        col_desc, col_pie = st.columns([3, 2])
        with col_desc:
            st.write("**기초 기술통계량 요약**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col_pie:
            st.write("**타깃 레이블(AftEval) 클래스 비율**")
            fig_pie = px.pie(
                values=[low_perf_count, high_perf_count],
                names=['저성과자 (0)', '고성과자 (1)'],
                color_discrete_sequence=['#94A3B8', '#3B82F6'],
                hole=0.45
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("채용 전형 점수와 입사 1년 후 성과의 관계")
        feature_choice = st.selectbox(
            "비교할 전형 점수를 선택하세요:",
            ["InterviewScore (면접 성적)", "SkillScore (코딩/기술 테스트)", "PersonalityScore (인성검사)", "PreviousExperience (경력 연차)"]
        )
        col_name = feature_choice.split(" ")[0]

        col_box, col_hist = st.columns(2)
        with col_box:
            fig_box = px.box(
                df, x='AftEval', y=col_name, color='AftEval',
                color_discrete_sequence=['#64748B', '#2563EB'],
                labels={'AftEval': '성과 구분 (0: 저성과, 1: 고성과)'},
                title=f"{col_name} - 성과 그룹별 박스플롯"
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_hist:
            fig_hist = px.histogram(
                df, x=col_name, color='AftEval', barmode='overlay',
                color_discrete_sequence=['#94A3B8', '#1D4ED8'],
                labels={'AftEval': '성과 구분'},
                title=f"{col_name} - 점수 분포 히스토그램"
            )
            fig_hist.update_traces(opacity=0.7)
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("전형 요소 및 성과 간 상관분석")
        corr_df = df.drop(columns=['EmpID']).corr().round(3)
        fig_corr = px.imshow(
            corr_df, text_auto=True, aspect="auto",
            color_continuous_scale='RdBu_r',
            title="피어슨 상관계수 매트릭스"
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# MENU 2: kNN 모델 튜닝 및 학습
# ==============================================================================
elif menu == "2. kNN 모델 튜닝 및 학습":
    st.markdown('<div class="main-header">⚙️ kNN 모델 구성 및 교차검증 튜닝</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">StandardScaler와 KNeighborsClassifier를 파이프라인으로 묶어 데이터 누수 없이 최적 파라미터를 탐색합니다.</div>', unsafe_allow_html=True)

    X = df.drop(columns=['EmpID', 'AftEval'])
    y = df['AftEval']

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🛠️ 튜닝 파라미터 설정")
        mode = st.radio("학습 모드 선택:", ["GridSearchCV 자동 최적화 (5-Fold)", "사용자 지정 수동 파라미터"])

        if mode == "GridSearchCV 자동 최적화 (5-Fold)":
            st.write("- **k 탐색 범위**: 3, 5, 7, 9, 11, 13, 15")
            st.write("- **가중치 방식**: uniform, distance")
            st.write("- **거리 척도**: euclidean, manhattan")
            run_btn = st.button("🚀 최적 하이퍼파라미터 탐색 시작", use_container_width=True, type="primary")
        else:
            k_val = st.slider("이웃 수 (k)", min_value=1, max_value=25, value=15, step=2)
            weight_val = st.selectbox("가중치 (weights)", ['distance', 'uniform'])
            metric_val = st.selectbox("거리 척도 (metric)", ['euclidean', 'manhattan'])
            run_btn = st.button("🎯 수동 모델 학습 실행", use_container_width=True, type="primary")

    with col_right:
        st.subheader("📈 학습 및 교차검증 결과")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        if run_btn:
            with st.spinner("교차검증을 수행하는 중입니다..."):
                if mode == "GridSearchCV 자동 최적화 (5-Fold)":
                    pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('knn', KNeighborsClassifier())
                    ])
                    param_grid = {
                        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                        'knn__weights': ['uniform', 'distance'],
                        'knn__metric': ['euclidean', 'manhattan']
                    }
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
                    grid.fit(X_train, y_train)

                    best_pipe = grid.best_estimator_
                    st.session_state['best_model'] = best_pipe
                    st.session_state['best_params'] = grid.best_params_
                    st.session_state['cv_score'] = grid.best_score_
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test

                    st.success("✅ 5-Fold 교차검증 기반 최적 모델 튜닝 완료!")
                    st.markdown(f"""
                    <div class="callout-success">
                        <h4>🏆 최적 하이퍼파라미터 결과</h4>
                        <ul>
                            <li><b>최적 k</b>: {grid.best_params_['knn__n_neighbors']}</li>
                            <li><b>가중치</b>: {grid.best_params_['knn__weights']}</li>
                            <li><b>거리 척도</b>: {grid.best_params_['knn__metric']}</li>
                            <li><b>최고 5-Fold 교차검증 정확도</b>: <b>{grid.best_score_ * 100:.2f}%</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    manual_pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('knn', KNeighborsClassifier(n_neighbors=k_val, weights=weight_val, metric=metric_val))
                    ])
                    manual_pipe.fit(X_train, y_train)
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(manual_pipe, X_train, y_train, cv=cv, scoring='accuracy')

                    st.session_state['best_model'] = manual_pipe
                    st.session_state['best_params'] = {'n_neighbors': k_val, 'weights': weight_val, 'metric': metric_val}
                    st.session_state['cv_score'] = scores.mean()
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test

                    st.success("✅ 수동 설정 모델 학습 완료!")
                    st.write(f"- 5-Fold 평균 CV 정확도: **{scores.mean()*100:.2f}%** (±{scores.std()*100:.2f}%)")
        else:
            if 'best_model' in st.session_state:
                st.info(f"학습된 모델이 세션에 유지되고 있습니다. (CV 정확도: {st.session_state['cv_score']*100:.2f}%)")
            else:
                st.info("왼쪽 패널에서 학습 모드를 선택하고 버튼을 클릭해주세요.")


# ==============================================================================
# MENU 3: 모델 평가 및 채용 오차 분석
# ==============================================================================
elif menu == "3. 모델 평가 및 채용 오차 분석":
    st.markdown('<div class="main-header">📊 모델 성능 평가 및 HR 리스크 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">테스트 데이터셋을 바탕으로 정확도, 혼동행렬, 그리고 채용 1종/2종 오차 비용을 분석합니다.</div>', unsafe_allow_html=True)

    if 'best_model' not in st.session_state:
        X = df.drop(columns=['EmpID', 'AftEval'])
        y = df['AftEval']
        if 'train_df' not in st.session_state:
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['AftEval'], random_state=42)
            st.session_state['train_df'] = train_df
            st.session_state['test_df'] = test_df
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean'))
        ])
        pipe.fit(X_train, y_train)
        st.session_state['best_model'] = pipe
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

    model = st.session_state['best_model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("정확도 (Accuracy)", f"{acc*100:.1f}%")
    m2.metric("선발 적중률 (Precision)", f"{prec*100:.1f}%")
    m3.metric("인재 발굴률 (Recall)", f"{rec*100:.1f}%")
    m4.metric("F1-Score", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.markdown("---")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    col_chart, col_risk = st.columns([1, 1])
    with col_chart:
        st.subheader("오차 행렬 (Confusion Matrix)")
        z = [[tn, fp], [fn, tp]]
        x_labels = ['예측: 저성과자 (0)', '예측: 고성과자 (1)']
        y_labels = ['실제: 저성과자 (0)', '실제: 고성과자 (1)']

        fig_cm = go.Figure(data=go.Heatmap(
            z=z, x=x_labels, y=y_labels, colorscale='Blues',
            text=[[f"TN (정상 기각)<br>{tn}명", f"FP (오류 선발)<br>{fp}명"],
                  [f"FN (오류 탈락)<br>{fn}명", f"TP (적격 선발)<br>{tp}명"]],
            texttemplate="%{text}", textfont={"size": 15}
        ))
        fig_cm.update_layout(height=380, margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_risk:
        st.subheader("💡 채용 의사결정 비즈니스 오차 분석")
        st.markdown(f"""
        - **TN (정상 기각): `{tn}명`**  
          저성과 예상자를 올바르게 미채용하여 조직 관리 비용을 절감했습니다.
        - **FP (오류 선발 / 1종 오류): <span style="color:#DC2626; font-weight:bold;">`{fp}명`</span>**  
          저성과자를 고성과자로 오판해 선발한 경우입니다. 교육/코칭 비용 및 조기 퇴사 리스크(**채용 실패 비용**)를 유발합니다.
        - **FN (오류 탈락 / 2종 오류): <span style="color:#D97706; font-weight:bold;">`{fn}명`</span>**  
          실제 우수인재를 전형에서 탈락시킨 경우입니다. 핵심 인재 유실로 인한 **기회손실**을 의미합니다.
        - **TP (적격 선발): `{tp}명`**  
          실제 고성과자를 성공적으로 선발했습니다.
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_roc, col_rep = st.columns([1, 1])
    with col_roc:
        st.subheader("ROC Curve (수신자 조작 특성 곡선)")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'kNN (AUC = {roc_auc:.3f})', line=dict(color='#2563EB', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess (0.50)', line=dict(color='#9CA3AF', dash='dash')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=360, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_rep:
        st.subheader("분류 상세 보고서 (Classification Report)")
        report_dict = classification_report(y_test, y_pred, target_names=['저성과자(0)', '고성과자(1)'], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)


# ==============================================================================
# MENU 4: 실시간 신규 지원자 성과 예측
# ==============================================================================
elif menu == "4. 실시간 신규 지원자 성과 예측":
    st.markdown('<div class="main-header">🎯 신규 지원자 성과 예측 시뮬레이터</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">지원자의 전형 점수를 입력해 1년 후 성과 확률을 산출하고, kNN 기반의 가장 유사한 과거 입사자 5명을 즉시 대조합니다.</div>', unsafe_allow_html=True)

    if 'best_model' not in st.session_state:
        X = df.drop(columns=['EmpID', 'AftEval'])
        y = df['AftEval']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean'))
        ])
        pipe.fit(X_train, y_train)
        st.session_state['best_model'] = pipe

    model = st.session_state['best_model']

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("📝 지원자 프로필 및 전형 점수 입력")
        with st.form("applicant_form"):
            cand_name = st.text_input("지원자 성명 / 식별코드", "김인재 (CAND-001)")
            gender_label = st.radio("성별", ["여성 (0)", "남성 (1)"], horizontal=True)
            gender_val = 1 if "남성" in gender_label else 0

            exp_val = st.select_slider("이전 직장 경력 연차", options=[0, 1, 2], value=1, format_func=lambda x: f"{x}년차")
            interview_val = st.slider("면접 전형 성적 (InterviewScore)", min_value=40, max_value=100, value=85, step=1)
            skill_val = st.slider("코딩 / 기술 테스트 성적 (SkillScore)", min_value=40, max_value=100, value=90, step=1)
            personality_val = st.slider("인성검사 성적 (PersonalityScore)", min_value=60, max_value=100, value=80, step=1)

            submit_btn = st.form_submit_button("🔮 1년 후 성과 예측 실행", use_container_width=True, type="primary")

    with col_result:
        st.subheader("📊 예측 결과 및 리포트")
        if submit_btn:
            new_cand = pd.DataFrame([{
                'Gender': gender_val,
                'PreviousExperience': exp_val,
                'InterviewScore': interview_val,
                'SkillScore': skill_val,
                'PersonalityScore': personality_val
            }])

            pred_class = model.predict(new_cand)[0]
            pred_probs = model.predict_proba(new_cand)[0]
            prob_low = pred_probs[0] * 100
            prob_high = pred_probs[1] * 100

            if pred_class == 1:
                st.markdown(f"""
                <div class="callout-success">
                    <h3 style="color:#065F46; margin:0;">🌟 [합격 추천] 우수 고성과자 예상</h3>
                    <p style="margin-top:5px; font-size:1.05rem;">지원자 <b>{cand_name}</b>님은 입사 1년 후 <b>고성과자(High Performer)</b>가 될 확률이 높습니다.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="callout-warning">
                    <h3 style="color:#92400E; margin:0;">⚠️ [신중 검토] 일반/관리대상 예상</h3>
                    <p style="margin-top:5px; font-size:1.05rem;">지원자 <b>{cand_name}</b>님은 입사 1년 후 <b>저성과(Average/Low)</b> 그룹에 속할 가능성이 높습니다.</p>
                </div>
                """, unsafe_allow_html=True)

            fig_donut = go.Figure(data=[go.Pie(
                labels=['고성과자 확률', '저성과자 확률'],
                values=[prob_high, prob_low],
                hole=0.55,
                marker_colors=['#2563EB', '#CBD5E1'],
                textinfo='label+percent'
            )])
            fig_donut.update_layout(height=260, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)

            st.write("**🔍 가장 유사한 과거 입사자(최근접 이웃 5명) 이력 비교**")
            scaler = model.named_steps['scaler']
            knn_model = model.named_steps['knn']

            new_scaled = scaler.transform(new_cand)
            distances, indices = knn_model.kneighbors(new_scaled, n_neighbors=5)
            train_df = st.session_state['train_df']
            neighbors_df = train_df.iloc[indices[0]].copy()
            neighbors_df['유사도 거리'] = distances[0].round(3)
            neighbors_df['1년 후 실제 성과'] = neighbors_df['AftEval'].apply(lambda x: '고성과자 (1)' if x == 1 else '저성과자 (0)')
            if 'train_df' not in st.session_state:
                train_df, _ = train_test_split(df, test_size=0.2, stratify=df['AftEval'], random_state=42)
                st.session_state['train_df'] = train_df

            st.dataframe(
                neighbors_df[['EmpID', 'InterviewScore', 'SkillScore', 'PersonalityScore', '유사도 거리', '1년 후 실제 성과']],
                use_container_width=True
            )
        else:
            st.info("왼쪽 패널에 전형 점수를 입력하고 '예측 실행' 버튼을 눌러주세요.")
