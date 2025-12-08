# MARKDOWN:
"""
# 📊 时间序列分析

本模块对资金费率数据进行深入的时间序列分析,包括趋势、周期性、自相关性、平稳性和波动率等多个维度。
"""

# CODE:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选库
try:
    from statsmodels.tsa.stattools import acf, pacf, adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels 未安装,部分高级分析功能将使用替代方法")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    SEASONAL_DECOMPOSE_AVAILABLE = True
except ImportError:
    SEASONAL_DECOMPOSE_AVAILABLE = False

# MARKDOWN:
"""
## 1. 趋势分析

使用移动平均线和指数移动平均线来识别资金费率的长期趋势。
"""

# CODE:
def calculate_moving_averages(df, column='funding_rate_annualized'):
    """
    计算移动平均线和指数移动平均线

    Parameters:
    -----------
    df : DataFrame
        包含时间序列数据的 DataFrame
    column : str
        要分析的列名

    Returns:
    --------
    DataFrame : 添加了移动平均线的 DataFrame
    """
    df = df.copy()

    # 7天移动平均 (7*3 = 21个数据点,因为每天3次)
    df['MA_7d'] = df[column].rolling(window=21, min_periods=1).mean()

    # 30天移动平均 (30*3 = 90个数据点)
    df['MA_30d'] = df[column].rolling(window=90, min_periods=1).mean()

    # 指数移动平均 (EMA)
    df['EMA_7d'] = df[column].ewm(span=21, adjust=False).mean()
    df['EMA_30d'] = df[column].ewm(span=90, adjust=False).mean()

    return df

# 计算移动平均线
df_trend = calculate_moving_averages(df_funding)

# 可视化趋势
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_trend['timestamp'],
    y=df_trend['funding_rate_annualized'],
    mode='lines',
    name='实际资金费率',
    line=dict(color='lightgray', width=1),
    opacity=0.6
))

fig.add_trace(go.Scatter(
    x=df_trend['timestamp'],
    y=df_trend['MA_7d'],
    mode='lines',
    name='7天移动平均',
    line=dict(color='blue', width=2)
))

fig.add_trace(go.Scatter(
    x=df_trend['timestamp'],
    y=df_trend['MA_30d'],
    mode='lines',
    name='30天移动平均',
    line=dict(color='red', width=2)
))

fig.add_trace(go.Scatter(
    x=df_trend['timestamp'],
    y=df_trend['EMA_7d'],
    mode='lines',
    name='7天指数移动平均',
    line=dict(color='green', width=2, dash='dash')
))

fig.update_layout(
    title='资金费率趋势分析 - 移动平均线',
    xaxis_title='时间',
    yaxis_title='年化资金费率 (%)',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

fig.show()

# 计算趋势统计
print("📈 趋势分析统计:")
print(f"当前值: {df_trend['funding_rate_annualized'].iloc[-1]:.2f}%")
print(f"7天均值: {df_trend['MA_7d'].iloc[-1]:.2f}%")
print(f"30天均值: {df_trend['MA_30d'].iloc[-1]:.2f}%")
print(f"\n趋势判断:")
if df_trend['funding_rate_annualized'].iloc[-1] > df_trend['MA_7d'].iloc[-1]:
    print("  ✓ 短期趋势: 上涨 (当前值 > 7天均线)")
else:
    print("  ✓ 短期趋势: 下跌 (当前值 < 7天均线)")

if df_trend['MA_7d'].iloc[-1] > df_trend['MA_30d'].iloc[-1]:
    print("  ✓ 中期趋势: 上涨 (7天均线 > 30天均线)")
else:
    print("  ✓ 中期趋势: 下跌 (7天均线 < 30天均线)")

# MARKDOWN:
"""
## 2. 周期性分析

分析资金费率在不同时间维度（小时、星期）的周期性模式。
"""

# CODE:
def analyze_periodicity(df):
    """
    分析时间序列的周期性特征

    Parameters:
    -----------
    df : DataFrame
        包含时间序列数据的 DataFrame

    Returns:
    --------
    tuple : (按小时统计, 按星期统计)
    """
    df = df.copy()

    # 提取时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['weekday_name'] = df['timestamp'].dt.day_name()

    # 按小时统计
    hourly_stats = df.groupby('hour')['funding_rate_annualized'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)

    # 按星期统计
    weekly_stats = df.groupby(['weekday', 'weekday_name'])['funding_rate_annualized'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)

    return df, hourly_stats, weekly_stats

# 周期性分析
df_period, hourly_stats, weekly_stats = analyze_periodicity(df_funding)

# 可视化 - 按小时分布
print("⏰ 按小时统计 (UTC 时间):")
print(hourly_stats)
print("\n关键观察:")
print(f"  • 00:00 UTC 平均费率: {hourly_stats.loc[0, 'mean']:.2f}%")
print(f"  • 08:00 UTC 平均费率: {hourly_stats.loc[8, 'mean']:.2f}%")
print(f"  • 16:00 UTC 平均费率: {hourly_stats.loc[16, 'mean']:.2f}%")

# 创建箱线图 - 按小时
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('按小时分布 (UTC)', '按星期分布'),
    vertical_spacing=0.12,
    row_heights=[0.5, 0.5]
)

# 按小时箱线图
for hour in sorted(df_period['hour'].unique()):
    hour_data = df_period[df_period['hour'] == hour]['funding_rate_annualized']
    fig.add_trace(
        go.Box(
            y=hour_data,
            name=f'{hour:02d}:00',
            marker_color='blue' if hour in [0, 8, 16] else 'lightblue',
            showlegend=False
        ),
        row=1, col=1
    )

# 按星期箱线图
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for weekday in weekday_order:
    if weekday in df_period['weekday_name'].values:
        weekday_data = df_period[df_period['weekday_name'] == weekday]['funding_rate_annualized']
        fig.add_trace(
            go.Box(
                y=weekday_data,
                name=weekday[:3],
                showlegend=False
            ),
            row=2, col=1
        )

fig.update_yaxes(title_text="年化费率 (%)", row=1, col=1)
fig.update_yaxes(title_text="年化费率 (%)", row=2, col=1)
fig.update_xaxes(title_text="小时 (UTC)", row=1, col=1)
fig.update_xaxes(title_text="星期", row=2, col=1)

fig.update_layout(
    title_text='资金费率周期性分析',
    height=800,
    template='plotly_white'
)

fig.show()

# 按星期统计
print("\n📅 按星期统计:")
print(weekly_stats)

# MARKDOWN:
"""
## 3. 自相关分析

通过 ACF 和 PACF 图分析时间序列的自相关性,识别可能的滞后模式。
"""

# CODE:
def plot_autocorrelation(df, column='funding_rate_annualized', lags=50):
    """
    绘制自相关和偏自相关图

    Parameters:
    -----------
    df : DataFrame
        时间序列数据
    column : str
        要分析的列名
    lags : int
        滞后阶数
    """
    data = df[column].dropna()

    if STATSMODELS_AVAILABLE:
        # 使用 statsmodels 绘制 ACF 和 PACF
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # ACF 图
        plot_acf(data, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('自相关函数 (ACF)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('滞后阶数')
        axes[0].set_ylabel('ACF')
        axes[0].grid(True, alpha=0.3)

        # PACF 图
        plot_pacf(data, lags=lags, ax=axes[1], alpha=0.05)
        axes[1].set_title('偏自相关函数 (PACF)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('滞后阶数')
        axes[1].set_ylabel('PACF')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 计算 ACF 和 PACF 值
        acf_values = acf(data, nlags=lags)
        pacf_values = pacf(data, nlags=lags)

        # 找出显著的滞后
        confidence_interval = 1.96 / np.sqrt(len(data))
        significant_acf = np.where(np.abs(acf_values[1:]) > confidence_interval)[0] + 1
        significant_pacf = np.where(np.abs(pacf_values[1:]) > confidence_interval)[0] + 1

        print("🔍 自相关分析结果:")
        print(f"\n显著的 ACF 滞后阶数 (前10个): {significant_acf[:10].tolist()}")
        print(f"显著的 PACF 滞后阶数 (前10个): {significant_pacf[:10].tolist()}")

        # 解释常见滞后
        if 1 in significant_acf[:5]:
            print("\n  ✓ 滞后1显著: 相邻时间点高度相关,序列具有短期记忆")
        if 3 in significant_acf[:10]:
            print("  ✓ 滞后3显著: 可能存在日周期 (每天3次结算)")
        if 21 in significant_acf[:30]:
            print("  ✓ 滞后21显著: 可能存在周周期 (7天×3次=21)")

    else:
        # 手动计算简单的自相关
        print("⚠️ 使用简化的自相关计算方法")

        acf_values = []
        for lag in range(lags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                acf_values.append(data.autocorr(lag=lag))

        # 绘制简单的 ACF 图
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.stem(range(lags + 1), acf_values, basefmt=' ')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=1, label='95% 置信区间')
        ax.axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=1)
        ax.set_title('自相关函数 (ACF) - 简化版', fontsize=14, fontweight='bold')
        ax.set_xlabel('滞后阶数')
        ax.set_ylabel('ACF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print("🔍 自相关分析结果:")
        print(f"滞后1自相关: {acf_values[1]:.4f}")
        print(f"滞后3自相关: {acf_values[3]:.4f} (日周期)")
        print(f"滞后21自相关: {acf_values[21]:.4f} (周周期)")

# 绘制自相关图
plot_autocorrelation(df_funding, lags=50)

# MARKDOWN:
"""
## 4. 平稳性检验

使用 ADF 检验 (Augmented Dickey-Fuller Test) 来测试时间序列的平稳性。

**平稳性的重要性:**
- 平稳序列: 均值和方差不随时间变化,更适合建模
- 非平稳序列: 存在趋势或季节性,需要差分或其他变换
"""

# CODE:
def test_stationarity(df, column='funding_rate_annualized'):
    """
    测试时间序列的平稳性

    Parameters:
    -----------
    df : DataFrame
        时间序列数据
    column : str
        要检验的列名
    """
    data = df[column].dropna()

    print("📊 平稳性检验")
    print("="*60)

    if STATSMODELS_AVAILABLE:
        # ADF 检验
        result = adfuller(data, autolag='AIC')

        print("\n🔬 ADF 检验结果 (Augmented Dickey-Fuller Test):")
        print(f"  • ADF 统计量: {result[0]:.4f}")
        print(f"  • p-value: {result[1]:.4f}")
        print(f"  • 使用的滞后阶数: {result[2]}")
        print(f"  • 观测数: {result[3]}")
        print("\n  临界值:")
        for key, value in result[4].items():
            print(f"    {key}: {value:.4f}")

        print("\n📖 结果解读:")
        if result[1] < 0.05:
            print("  ✅ p-value < 0.05: 拒绝原假设")
            print("  → 序列是平稳的")
            print("  → 可以直接用于时间序列建模 (ARIMA等)")
        else:
            print("  ⚠️ p-value >= 0.05: 不能拒绝原假设")
            print("  → 序列可能是非平稳的")
            print("  → 建议进行差分或其他变换")

            # 进行一阶差分检验
            print("\n尝试一阶差分后的检验:")
            diff_data = data.diff().dropna()
            result_diff = adfuller(diff_data, autolag='AIC')
            print(f"  • 差分后 p-value: {result_diff[1]:.4f}")
            if result_diff[1] < 0.05:
                print("  ✅ 一阶差分后序列平稳")
            else:
                print("  ⚠️ 可能需要二阶差分或其他方法")
    else:
        # 简单的滚动统计检验
        print("\n⚠️ statsmodels 不可用,使用简化的滚动统计方法")

        # 计算滚动均值和标准差
        window = 30  # 30个数据点 (约10天)
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()

        # 绘制
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 原始序列
        axes[0].plot(data.values, label='原始数据', color='blue', alpha=0.7)
        axes[0].plot(rolling_mean.values, label=f'{window}点滚动均值', color='red', linewidth=2)
        axes[0].set_title('原始时间序列与滚动均值', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 滚动标准差
        axes[1].plot(rolling_std.values, label=f'{window}点滚动标准差', color='green')
        axes[1].set_title('滚动标准差', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 一阶差分
        diff_data = data.diff().dropna()
        axes[2].plot(diff_data.values, label='一阶差分', color='purple', alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[2].set_title('一阶差分序列', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 简单判断
        mean_change = abs(rolling_mean.iloc[-1] - rolling_mean.iloc[window]) / rolling_mean.iloc[window] * 100
        std_change = abs(rolling_std.iloc[-1] - rolling_std.iloc[window]) / rolling_std.iloc[window] * 100

        print(f"\n📊 滚动统计变化:")
        print(f"  • 均值变化: {mean_change:.2f}%")
        print(f"  • 标准差变化: {std_change:.2f}%")
        print("\n📖 简单判断:")
        if mean_change < 20 and std_change < 50:
            print("  ✓ 均值和方差相对稳定,序列可能是平稳的")
        else:
            print("  ⚠️ 均值或方差变化较大,序列可能是非平稳的")

# 执行平稳性检验
test_stationarity(df_funding)

# MARKDOWN:
"""
## 5. 波动率分析

分析资金费率的波动性特征,包括滚动波动率和波动率聚集现象。
"""

# CODE:
def analyze_volatility(df, column='funding_rate_annualized', windows=[7, 14, 30]):
    """
    分析时间序列的波动率

    Parameters:
    -----------
    df : DataFrame
        时间序列数据
    column : str
        要分析的列名
    windows : list
        滚动窗口大小 (天数)

    Returns:
    --------
    DataFrame : 添加了波动率指标的 DataFrame
    """
    df = df.copy()

    # 计算收益率 (变化率)
    df['returns'] = df[column].pct_change()

    # 计算不同窗口的滚动波动率
    for window in windows:
        window_points = window * 3  # 每天3个数据点
        df[f'volatility_{window}d'] = df['returns'].rolling(window=window_points).std() * 100

    return df

# 计算波动率
df_volatility = analyze_volatility(df_funding, windows=[7, 14, 30])

# 可视化波动率
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('资金费率时间序列', '滚动波动率'),
    vertical_spacing=0.1,
    row_heights=[0.4, 0.6]
)

# 原始序列
fig.add_trace(
    go.Scatter(
        x=df_volatility['timestamp'],
        y=df_volatility['funding_rate_annualized'],
        mode='lines',
        name='资金费率',
        line=dict(color='blue', width=1)
    ),
    row=1, col=1
)

# 滚动波动率
fig.add_trace(
    go.Scatter(
        x=df_volatility['timestamp'],
        y=df_volatility['volatility_7d'],
        mode='lines',
        name='7天波动率',
        line=dict(color='orange', width=2)
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_volatility['timestamp'],
        y=df_volatility['volatility_14d'],
        mode='lines',
        name='14天波动率',
        line=dict(color='green', width=2)
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_volatility['timestamp'],
        y=df_volatility['volatility_30d'],
        mode='lines',
        name='30天波动率',
        line=dict(color='red', width=2)
    ),
    row=2, col=1
)

fig.update_yaxes(title_text="年化费率 (%)", row=1, col=1)
fig.update_yaxes(title_text="波动率 (%)", row=2, col=1)
fig.update_xaxes(title_text="时间", row=2, col=1)

fig.update_layout(
    title_text='资金费率波动率分析',
    height=700,
    template='plotly_white',
    hovermode='x unified'
)

fig.show()

# 波动率统计
print("📊 波动率统计:")
print(f"  • 当前7天波动率: {df_volatility['volatility_7d'].iloc[-1]:.4f}%")
print(f"  • 当前14天波动率: {df_volatility['volatility_14d'].iloc[-1]:.4f}%")
print(f"  • 当前30天波动率: {df_volatility['volatility_30d'].iloc[-1]:.4f}%")
print(f"\n  • 平均7天波动率: {df_volatility['volatility_7d'].mean():.4f}%")
print(f"  • 最大7天波动率: {df_volatility['volatility_7d'].max():.4f}%")
print(f"  • 最小7天波动率: {df_volatility['volatility_7d'].min():.4f}%")

# 波动率聚集现象分析
returns_squared = df_volatility['returns'].dropna() ** 2
if STATSMODELS_AVAILABLE:
    acf_returns_sq = acf(returns_squared, nlags=30)
    print("\n🔍 波动率聚集现象检验:")
    print("  (收益率平方的自相关性)")
    print(f"  • 滞后1自相关: {acf_returns_sq[1]:.4f}")
    print(f"  • 滞后5自相关: {acf_returns_sq[5]:.4f}")
    print(f"  • 滞后10自相关: {acf_returns_sq[10]:.4f}")

    if acf_returns_sq[1] > 0.1:
        print("\n  ✅ 存在明显的波动率聚集现象")
        print("     (高波动后往往跟随高波动,低波动后跟随低波动)")
        print("     → 适合使用 GARCH 类模型建模")
    else:
        print("\n  → 波动率聚集现象不明显")
else:
    lag1_corr = returns_squared.autocorr(lag=1)
    print(f"\n🔍 波动率聚集现象 (滞后1自相关): {lag1_corr:.4f}")
    if lag1_corr > 0.1:
        print("  ✅ 存在波动率聚集现象")

# MARKDOWN:
"""
## 6. 季节性分解 (可选)

如果 statsmodels 可用,使用季节性分解将时间序列拆分为趋势、季节性和残差成分。
"""

# CODE:
if SEASONAL_DECOMPOSE_AVAILABLE:
    try:
        # 设置周期 (假设每天3次,7天为一个周期)
        period = 21  # 7天 × 3次/天

        # 确保数据足够长
        if len(df_funding) >= 2 * period:
            # 执行季节性分解
            decomposition = seasonal_decompose(
                df_funding['funding_rate_annualized'],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )

            # 创建可视化
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))

            # 原始序列
            axes[0].plot(df_funding['timestamp'], df_funding['funding_rate_annualized'],
                        label='原始数据', color='blue', linewidth=1)
            axes[0].set_ylabel('费率 (%)', fontsize=10)
            axes[0].set_title('原始时间序列', fontweight='bold', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 趋势成分
            axes[1].plot(df_funding['timestamp'], decomposition.trend,
                        label='趋势', color='red', linewidth=2)
            axes[1].set_ylabel('趋势', fontsize=10)
            axes[1].set_title('趋势成分', fontweight='bold', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 季节性成分
            axes[2].plot(df_funding['timestamp'], decomposition.seasonal,
                        label='季节性', color='green', linewidth=1)
            axes[2].set_ylabel('季节性', fontsize=10)
            axes[2].set_title('季节性成分 (周期=7天)', fontweight='bold', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # 残差成分
            axes[3].plot(df_funding['timestamp'], decomposition.resid,
                        label='残差', color='purple', linewidth=1, alpha=0.7)
            axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            axes[3].set_ylabel('残差', fontsize=10)
            axes[3].set_title('残差成分', fontweight='bold', fontsize=12)
            axes[3].set_xlabel('时间', fontsize=10)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # 计算各成分的统计特征
            print("📊 季节性分解统计:")
            print(f"\n趋势成分:")
            print(f"  • 起始值: {decomposition.trend.dropna().iloc[0]:.2f}%")
            print(f"  • 结束值: {decomposition.trend.dropna().iloc[-1]:.2f}%")
            print(f"  • 变化: {decomposition.trend.dropna().iloc[-1] - decomposition.trend.dropna().iloc[0]:.2f}%")

            print(f"\n季节性成分:")
            print(f"  • 振幅: {decomposition.seasonal.max() - decomposition.seasonal.min():.2f}%")
            print(f"  • 最大值: {decomposition.seasonal.max():.2f}%")
            print(f"  • 最小值: {decomposition.seasonal.min():.2f}%")

            print(f"\n残差成分:")
            print(f"  • 标准差: {decomposition.resid.std():.2f}%")
            print(f"  • 均值: {decomposition.resid.mean():.4f}% (应接近0)")

            # 计算各成分的贡献度
            total_var = df_funding['funding_rate_annualized'].var()
            trend_var = decomposition.trend.dropna().var()
            seasonal_var = decomposition.seasonal.var()
            resid_var = decomposition.resid.dropna().var()

            print(f"\n方差贡献度:")
            print(f"  • 趋势: {trend_var/total_var*100:.2f}%")
            print(f"  • 季节性: {seasonal_var/total_var*100:.2f}%")
            print(f"  • 残差: {resid_var/total_var*100:.2f}%")

        else:
            print(f"⚠️ 数据长度 ({len(df_funding)}) 不足,至少需要 {2*period} 个数据点进行季节性分解")

    except Exception as e:
        print(f"❌ 季节性分解失败: {str(e)}")
else:
    print("⚠️ statsmodels.tsa.seasonal 不可用,跳过季节性分解")
    print("提示: 可以通过 pip install statsmodels 安装")

# MARKDOWN:
"""
## 📝 时间序列分析总结

通过以上分析,我们从多个维度深入了解了资金费率的时间序列特征:

### 关键发现:

1. **趋势特征**
   - 移动平均线显示长期趋势方向
   - EMA 对近期变化更敏感
   - 可用于判断市场情绪变化

2. **周期性特征**
   - 每日3次结算产生日内周期 (00:00, 08:00, 16:00 UTC)
   - 可能存在周周期模式
   - 不同时段费率存在系统性差异

3. **自相关特征**
   - 相邻时间点高度相关 (短期记忆)
   - 特定滞后阶数显著 (周期性证据)
   - 为 ARIMA 建模提供参数参考

4. **平稳性特征**
   - ADF 检验判断是否需要差分
   - 平稳性影响模型选择
   - 非平稳序列需要变换

5. **波动率特征**
   - 滚动波动率显示市���风险变化
   - 波动率聚集现象 (GARCH 效应)
   - 高波动期往往持续一段时间

6. **季节性分解**
   - 分离趋势、季节性和随机成分
   - 识别各成分的贡献度
   - 为预测建模提供基础

### 后续建议:

- 📊 可以基于这些特征构建预测模型 (ARIMA, GARCH, Prophet等)
- 🎯 结合交易策略,利用周期性和趋势特征
- ⚠️ 关注高波动期的风险管理
- 🔄 定期更新分析,跟踪特征变化
"""

# CODE:
print("✅ 时间序列分析模块完成!")
print(f"分析时间段: {df_funding['timestamp'].min()} 至 {df_funding['timestamp'].max()}")
print(f"数据点数量: {len(df_funding)}")
