"""
金融数学实验报告 - 第二问：股票分析与多元回归预测（教学版）
目标：杰瑞股份(002353.SZ)
数据来源：腾讯理财接口

【使用说明】
1. 本代码为教学参考骨架，核心功能完整但注释详尽
2. 部分高级分析（滚动窗口、GARCH、机器学习对比等）留作扩展练习
3. 运行前确保安装：pip install pandas numpy matplotlib statsmodels requests
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 第一步：数据获取
# ===========================

def get_kline(code, start_date, end_date, field='qfqday'):
    """
    从腾讯理财接口获取K线数据
    个股用 'qfqday'（前复权），指数用 'day'
    """
    all_data = []
    # 按半年分段请求（腾讯接口单次最多返回约640条）
    years = [(y, 1) for y in range(2020, 2027)] + [(y, 7) for y in range(2020, 2026)]
    for y, m in sorted(years):
        s = f"{y}-{m:02d}-01"
        e = f"{y}-{m+5:02d}-30" if m == 1 else f"{y}-{m+5:02d}-31"
        if e < start_date or s > end_date:
            continue
        a_s = max(s, start_date)
        a_e = min(e, end_date)
        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {'param': f'{code},day,{a_s},{a_e},640,qfq'}
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if isinstance(data.get('data'), dict) and code in data['data']:
                arr = data['data'][code].get(field, [])
                for row in arr:
                    if len(row) >= 6:
                        all_data.append(row[:6])  # 日期 开盘 收盘 最高 最低 成交量
        except Exception as e:
            print(f"  获取{a_s}~{a_e}失败: {e}")
    
    df = pd.DataFrame(all_data, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'close', 'high', 'low', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.drop_duplicates('date').sort_values('date').reset_index(drop=True)


print("=" * 50)
print("Step 1: 获取数据")
print("=" * 50)

# 获取杰瑞股份K线（前复权）
df_stock = get_kline('sz002353', '2020-01-01', '2026-05-19', 'qfqday')
print(f"杰瑞股份: {len(df_stock)}条")

# 获取沪深300指数K线（指数无前复权，用'day'字段）
df_index = get_kline('sh000300', '2020-01-01', '2026-05-19', 'day')
print(f"沪深300: {len(df_index)}条")

# TODO: 可选 - 获取个股实时PE/PB等基本面指标
# 提示：可用 https://qt.gtimg.cn/q=sz002353 接口，解析'~'分隔的数据

# ===========================
# 第二步：数据清洗与指标计算
# ===========================

print("\n" + "=" * 50)
print("Step 2: 数据清洗与技术指标")
print("=" * 50)

# 合并个股与指数
df = df_stock.merge(df_index[['date', 'close']], on='date', suffixes=('', '_idx'))
df.rename(columns={'close_idx': 'hs300_close'}, inplace=True)

# 计算技术指标
# 提示：尝试自行补充更多指标，如KDJ、威廉指标、OBV等
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()
df['MA60'] = df['close'].rolling(window=60).mean()

# MACD: DIF = EMA12 - EMA26, DEA = EMA9(DIF), MACD柱 = 2*(DIF-DEA)
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
df['DIF'] = ema12 - ema26
df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
df['MACD_HIST'] = 2 * (df['DIF'] - df['DEA'])

# RSI(14)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

# 对数成交量
df['log_volume'] = np.log(df['volume'] + 1)

# 滞后变量（用于回归）
df['lag_close'] = df['close'].shift(1)  # 昨天收盘价
df['ret'] = np.log(df['close'] / df['close'].shift(1))  # 对数收益率
df['lag5_ret'] = df['ret'].rolling(window=5).sum().shift(1)  # 前5日累计收益

# 删除缺失值
df = df.dropna().copy()
print(f"清洗后样本量: {len(df)}")

# ===========================
# 第三步：走势可视化
# ===========================

print("\n" + "=" * 50)
print("Step 3: 绘制走势图")
print("=" * 50)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

# 子图1：K线+均线
ax1 = axes[0]
ax1.plot(df['date'], df['close'], label='收盘价', color='black', lw=1)
ax1.plot(df['date'], df['MA5'], label='MA5', color='orange', alpha=0.8)
ax1.plot(df['date'], df['MA20'], label='MA20', color='blue', alpha=0.8)
ax1.set_title('杰瑞股份(002353) 走势与均线系统')
ax1.set_ylabel('价格(元)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 子图2：成交量
ax2 = axes[1]
colors = ['red' if c >= o else 'green' for c, o in zip(df['close'], df['open'])]
ax2.bar(df['date'], df['volume'], color=colors, alpha=0.6, width=1)
ax2.set_ylabel('成交量')
ax2.grid(True, alpha=0.3)

# 子图3：MACD
ax3 = axes[2]
ax3.plot(df['date'], df['DIF'], label='DIF', color='blue', lw=0.8)
ax3.plot(df['date'], df['DEA'], label='DEA', color='orange', lw=0.8)
h_colors = ['red' if h >= 0 else 'green' for h in df['MACD_HIST']]
ax3.bar(df['date'], df['MACD_HIST'], color=h_colors, alpha=0.5, width=1)
ax3.axhline(y=0, color='black', lw=0.5)
ax3.set_ylabel('MACD')
ax3.set_xlabel('日期')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('走势与MACD.png', dpi=150)
print("已保存: 走势与MACD.png")
plt.show()

# TODO: 自行补充 RSI、布林带、KDJ 等子图

# ===========================
# 第四步：多元回归建模
# ===========================

print("\n" + "=" * 50)
print("Step 4: 多元回归建模")
print("=" * 50)

# 被解释变量 Y：当日收盘价
Y = df['close']

# 解释变量 X
# 提示：可尝试增减变量，观察R2和显著性的变化
X_vars = [
    'hs300_close',   # 市场因子
    'log_volume',    # 成交量
    'lag_close',     # 滞后收盘价（AR1项）
    'lag5_ret',      # 5日动量
    'MACD_HIST',     # MACD动能
    'RSI',           # 相对强弱
]
X = df[X_vars]
X_const = sm.add_constant(X)  # 加入常数项

# OLS估计
model = sm.OLS(Y, X_const).fit()
print(model.summary())

# 模型诊断
print("\n【模型诊断】")
dw = sm.stats.stattools.durbin_watson(model.resid)
print(f"DW统计量: {dw:.4f} (<1.5或>2.5需关注自相关)")

white_test = het_white(model.resid, X_const)
print(f"White检验p值: {white_test[1]:.4f} (<0.05说明存在异方差)")

# TODO: 若存在异方差/自相关，尝试使用 Newey-West 稳健标准误
# model_robust = sm.OLS(Y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': 20})

# ===========================
# 第五步：一步向前预测未来
# ===========================

print("\n" + "=" * 50)
print("Step 5: 一步向前预测（用今天预测明天）")
print("=" * 50)

# 核心修正：不能用同期变量预测同期股价
# 正确做法：用 t 期 X 预测 t+1 期 Y

df['y_tomorrow'] = df['close'].shift(-1)  # 明天的收盘价
df_fwd = df.dropna(subset=['y_tomorrow']).copy()

Y_fwd = df_fwd['y_tomorrow']
X_fwd = sm.add_constant(df_fwd[X_vars])
model_fwd = sm.OLS(Y_fwd, X_fwd).fit()

print(f"\n一步向前模型 Adj R2: {model_fwd.rsquared_adj:.4f}")
print("显著变量:")
for var in X_vars:
    p = model_fwd.pvalues[var]
    if p < 0.05:
        sig = '***' if p < 0.01 else '**'
        print(f"  {var:15s}: coef={model_fwd.params[var]:10.4f}, p={p:.4f} {sig}")

# 预测未来5天
print("\n未来5天递归预测:")
last = df.iloc[-1]
last_close = last['close']
last_5rets = list(df['ret'].iloc[-5:].values)

for day in range(1, 6):
    pred_x = pd.DataFrame([{
        'const': 1,
        'hs300_close': last['hs300_close'],
        'log_volume': last['log_volume'],
        'lag_close': last_close,
        'lag5_ret': sum(last_5rets),
        'MACD_HIST': last['MACD_HIST'],
        'RSI': last['RSI'],
    }])
    pred = model_fwd.predict(pred_x)[0]
    
    # 递归更新：把预测值当作已知，继续预测下一天
    new_ret = np.log(pred / last_close)
    last_5rets.pop(0)
    last_5rets.append(new_ret)
    last_close = pred
    
    print(f"  T+{day}: {pred:.2f} 元")

# TODO: 自行补充样本外回测评价（MAE/RMSE/MAPE）
# TODO: 自行补充预测效果图绘制

# ===========================
# 第六步：结论（学生自行完善）
# ===========================

print("\n" + "=" * 50)
print("Step 6: 结论与讨论（请自行补充完善）")
print("=" * 50)

print("""
【需要自行补充的内容】
1. 股票基本面分析：所属行业、主营业务、财务指标（ROE/PE/PB）
2. 走势解读：结合K线图分析关键时间节点的涨跌原因
3. 回归结果解读：哪些因子显著？系数符号是否符合经济直觉？
4. 模型局限：
   - 存在异方差/自相关吗？如何修正？
   - 是否遗漏了重要变量（如行业指数、油价、汇率）？
   - 能否尝试对数收益率模型进行对比？
5. 改进方向：
   - 引入GARCH刻画波动率
   - 使用滚动窗口检验模型稳定性
   - 尝试Lasso/Ridge正则化或机器学习模型
""")

print("\n" + "=" * 50)
print("运行完毕！")
print("=" * 50)
