import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="نظام التحليل العددي للوجستيات", layout="wide")

# ==========================================
# 2. خوارزميات التحليل العددي (من الصفر)
# ==========================================

# المرحلة الأولى (أ): استكمال نيوتن الأمامي لتوقع السرعة
def newton_forward_interpolation(x_data, y_data, x_target):
    n = len(x_data)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_data

    # بناء جدول الفروقات الأمامية
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    h = x_data[1] - x_data[0]
    u = (x_target - x_data[0]) / h
    y_interp = diff_table[0][0]
    u_term = 1.0
    fact = 1.0

    for i in range(1, n):
        u_term *= (u - (i - 1))
        fact *= i
        y_interp += (u_term * diff_table[0][i]) / fact

    return y_interp

# المرحلة الأولى (ب): توفيق المنحنيات الخطي لتحديد الاتجاه العام
def linear_curve_fitting(x_data, y_data):
    n = len(x_data)
    sum_x = np.sum(x_data)
    sum_y = np.sum(y_data)
    sum_xy = np.sum(x_data * y_data)
    sum_x2 = np.sum(x_data ** 2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    c = (sum_y - m * sum_x) / n
    return m, c

# المرحلة الثانية: طريقة نيوتن-رافسون لتوزيع الأحمال وتوحيد زمن الوصول
def newton_raphson(func, x0, tol=1e-6, max_iter=100):
    x = x0
    history = [x]
    h = 1e-5  # خطوة صغيرة لحساب المشتقة عددياً

    for i in range(1, max_iter + 1):
        fx = func(x)
        if abs(fx) < tol:
            return x, i, history

        # المشتقة الأولى باستخدام الفروق المركزية
        dfx = (func(x + h) - func(x - h)) / (2 * h)
        if abs(dfx) < 1e-12:
            break

        x_new = x - (fx / dfx)
        history.append(x_new)

        if abs(x_new - x) < tol:
            return x_new, i, history
        x = x_new

    return x, max_iter, history

# المرحلة الثالثة (أ): الحذف لجاوس لحل نظام توزيع المخازن
def gauss_elimination(A, b):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    # الحذف الأمامي (Forward Elimination)
    for k in range(n):
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

    # التعويض الخلفي (Back Substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]

    return x

# المرحلة الثالثة (ب): طريقة جاكوبي التكرارية لحل نظام المخازن
def jacobi_method(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)

    for it in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, it
        x = x_new

    return x, max_iter

# المرحلة الرابعة (أ): قاعدة شبه المنحرف للتكامل العددي
def trapezoidal_rule(y_data, h):
    return (h / 2.0) * (y_data[0] + 2 * np.sum(y_data[1:-1]) + y_data[-1])

# المرحلة الرابعة (ب): قاعدة سيمبسون 1/3 للتكامل العددي
def simpsons_13_rule(y_data, h):
    n = len(y_data) - 1
    if n % 2 != 0:
        raise ValueError("قاعدة سيمبسون 1/3 تتطلب عدداً زوجياً من الفترات.")
    odd_sum = np.sum(y_data[1:-1:2])
    even_sum = np.sum(y_data[2:-1:2])
    return (h / 3.0) * (y_data[0] + 4 * odd_sum + 2 * even_sum + y_data[-1])

# ==========================================
# 3. البيانات الأساسية ومدخلات المستخدم
# ==========================================
hours = np.array([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
speed_data_r1 = np.array([85.0, 70.0, 55.0, 65.0, 50.0, 45.0, 75.0])  # الطريق السريع
speed_data_r2 = np.array([65.0, 60.0, 50.0, 55.0, 45.0, 40.0, 60.0])  # الطريق الدائري

st.sidebar.header("⚙️ إعدادات المحاكاة")
target_hour = st.sidebar.slider("الساعة المطلوبة لتوقع السرعة:", 8.0, 20.0, 11.0, step=0.5)
total_cargo = st.sidebar.slider("إجمالي البضاعة المطلوب نقلها (طن):", 20.0, 180.0, 100.0, step=5.0)
route1_cap = st.sidebar.slider("السعة القصوى للطريق السريع (طن):", 60.0, 150.0, 110.0, step=5.0)
route2_cap = st.sidebar.slider("السعة القصوى للطريق الدائري (طن):", 60.0, 150.0, 95.0, step=5.0)

# حماية رياضية لمنع القسمة على صفر في معادلات الزحمة
max_allowed = (route1_cap - 5.0) + (route2_cap - 5.0)
safe_cargo = min(total_cargo, max_allowed)
if total_cargo > max_allowed:
    st.sidebar.warning(f"تم ضبط الحمولة عند الحد الأقصى الآمن ({safe_cargo:.1f} طن) لتجنب اختناق الطرق.")

# ==========================================
# 4. واجهة العرض التفاعلية (Dashboard)
# ==========================================
st.title("🚛 نظام تحسين سلاسل الإمداد واللوجستيات بالتحليل العددي")
st.markdown("تطبيق عملي لدمج طرق التحليل العددي في التنبؤ بالسرعات، توزيع الأحمال، إدارة المخازن، وحساب استهلاك الوقود.")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ التنبؤ بالسرعة وتوزيع الأحمال",
    "2️⃣ شبكة توزيع المخازن (Linear Systems)",
    "3️⃣ تقرير المسافة والوقود (Integration)"
])

# ------------------------------------------
# التبويب الأول: Interpolation + Curve Fitting + Newton-Raphson
# ------------------------------------------
with tab1:
    speed_r1 = newton_forward_interpolation(hours, speed_data_r1, target_hour)
    speed_r2 = newton_forward_interpolation(hours, speed_data_r2, target_hour)
    m1, c1 = linear_curve_fitting(hours, speed_data_r1)
    m2, c2 = linear_curve_fitting(hours, speed_data_r2)

    # معادلة فرق زمن الوصول بين الطريقين (مسافة الطريق السريع 120 كم، والدائري 100 كم)
    base_time_r1 = (120.0 / max(speed_r1, 10.0)) * 60.0
    base_time_r2 = (100.0 / max(speed_r2, 10.0)) * 60.0

    def route_time_diff(x):
        x1 = np.clip(x, 0.0, route1_cap - 1.0)
        x2 = np.clip(safe_cargo - x1, 0.0, route2_cap - 1.0)
        t1 = base_time_r1 + (300.0 / (route1_cap - x1))
        t2 = base_time_r2 + (250.0 / (route2_cap - x2))
        return t1 - t2

    x0_guess = safe_cargo * (route1_cap / (route1_cap + route2_cap))
    opt_r1, iter_nr, hist_nr = newton_raphson(route_time_diff, x0_guess)
    opt_r1 = float(np.clip(opt_r1, 0.0, safe_cargo))
    opt_r2 = safe_cargo - opt_r1
    sync_time = base_time_r1 + (300.0 / (route1_cap - opt_r1))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("سرعة السريع المتوقعة", f"{speed_r1:.2f} كم/س")
    col2.metric("حصة الطريق السريع", f"{opt_r1:.2f} طن")
    col3.metric("حصة الطريق الدائري", f"{opt_r2:.2f} طن")
    col4.metric("زمن الوصول الموحد", f"{sync_time:.1f} دقيقة", f"حُلّ في {iter_nr} تكرارات")

    # رسم بياني للمقارنة بين Interpolation و Curve Fitting
    smooth_hours = np.linspace(8.0, 20.0, 100)
    interp_curve_r1 = [newton_forward_interpolation(hours, speed_data_r1, h) for h in smooth_hours]
    fit_line_r1 = m1 * smooth_hours + c1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=speed_data_r1, mode='markers', name='قراءات الرادار الفعلية', marker=dict(size=10, color='red')))
    fig.add_trace(go.Scatter(x=smooth_hours, y=interp_curve_r1, mode='lines', name='منحنى Newton Forward', line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=smooth_hours, y=fit_line_r1, mode='lines', name='خط الاتجاه العام (Curve Fitting)', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=[target_hour], y=[speed_r1], mode='markers', name='نقطة التوقع المختارة', marker=dict(size=14, symbol='star', color='orange')))
    fig.update_layout(title="تحليل سرعة الطريق السريع على مدار اليوم", xaxis_title="الساعة", yaxis_title="السرعة (كم/س)", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# التبويب الثاني: Gauss Elimination vs Jacobi Method
# ------------------------------------------
with tab2:
    st.subheader("موازنة المخزون بين 3 مستودعات رئيسية")
    # مصفوفة معاملات ذات سيطرة قطرية (Diagonally Dominant) لضمان تقارب جاكوبي
    A = np.array([
        [4.0, -1.0,  0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0,  4.0]
    ])
    # ربط ناتج المرحلة السابقة كطلب أساسي للمخزن الأول
    b = np.array([safe_cargo, 75.0, 55.0])

    x_gauss = gauss_elimination(A, b)
    x_jacobi, iter_jacobi = jacobi_method(A, b)

    df_warehouses = pd.DataFrame({
        "المخزن": ["المخزن الرئيسي (A)", "المخزن الإقليمي (B)", "المخزن الفرعي (C)"],
        "الحل المباشر (Gauss Elimination) [طن]": np.round(x_gauss, 4),
        f"الحل التكراري (Jacobi - {iter_jacobi} تكرار) [طن]": np.round(x_jacobi, 4),
        "الفرق المطلق": np.abs(x_gauss - x_jacobi)
    })
    st.table(df_warehouses)

# ------------------------------------------
# التبويب الثالث: Trapezoidal vs Simpson's 1/3 Rule
# ------------------------------------------
with tab3:
    st.subheader("حساب المسافة المقطوعة واستهلاك الوقود اليومي")
    h_step = 2.0  # الفترة الزمنية بين القراءات (ساعتان)
    fuel_efficiency = 4.0  # معدل استهلاك الشاحنة (4 كم لكل لتر)

    dist_trap_r1 = trapezoidal_rule(speed_data_r1, h_step)
    dist_simp_r1 = simpsons_13_rule(speed_data_r1, h_step)

    dist_trap_r2 = trapezoidal_rule(speed_data_r2, h_step)
    dist_simp_r2 = simpsons_13_rule(speed_data_r2, h_step)

    df_report = pd.DataFrame({
        "المسار": ["الطريق السريع (Route 1)", "الطريق الدائري (Route 2)"],
        "المسافة بـ Trapezoidal (كم)": [round(dist_trap_r1, 2), round(dist_trap_r2, 2)],
        "المسافة بـ Simpson 1/3 (كم)": [round(dist_simp_r1, 2), round(dist_simp_r2, 2)],
        "الوقود المستهلك بـ Simpson (لتر)": [round(dist_simp_r1 / fuel_efficiency, 2), round(dist_simp_r2 / fuel_efficiency, 2)]
    })
    st.table(df_report)