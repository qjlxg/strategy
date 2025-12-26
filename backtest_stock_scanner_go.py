import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 策略再优化参数 ===================
MIN_PRICE = 5.0
MAX_AVG_TURNOVER_30 = 2.0     # 强化：只要长期冷门的标的
MIN_VOLUME_RATIO = 0.3        # 强化：避免流动性死掉的僵尸股
MAX_VOLUME_RATIO = 1.0        # 优化：允许平量，但拒绝放巨量
RSI6_MAX = 28                 # 均衡点：比25宽松，比30严谨
KDJ_K_MAX = 25                # 强化：必须在超卖区磨底
MIN_PROFIT_POTENTIAL = 18     # 强化：反弹空间要求更高

# 核心新增：乖离率控制 (现价距离20日线不能太远，防止加速赶底)
MAX_BIAS_20 = -15             # 股价在20日线下方5%~15%之间

STOP_LOSS = -5.0             
TRAILING_START = 10.0        
HOLD_PERIODS = [5, 7, 15, 30]

def calculate_indicators(df):
    df = df.reset_index(drop=True)
    close = df['收盘']
    
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['rsi6'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    # KDJ
    low_list = df['最低'].rolling(window=9).min()
    high_list = df['最高'].rolling(window=9).max()
    df['kdj_k'] = ((df['收盘'] - low_list) / (high_list - low_list) * 100).ewm(com=2).mean()
    
    # MA & BIAS
    df['ma5'] = close.rolling(window=5).mean()
    df['ma20'] = close.rolling(window=20).mean()
    df['ma60'] = close.rolling(window=60).mean()
    df['bias20'] = (df['收盘'] - df['ma20']) / df['ma20'] * 100
    
    # 量能逻辑
    df['vol_ma5'] = df['成交量'].shift(1).rolling(window=5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']
    df['avg_turnover_30'] = df['换手率'].rolling(window=30).mean()
    
    return df

def simulate_trade(df, start_idx, max_days):
    # (保持之前的移动止盈与止损逻辑不变)
    buy_price = df.iloc[start_idx]['收盘']
    max_price = buy_price
    for day in range(1, max_days + 1):
        if start_idx + day >= len(df): break
        row = df.iloc[start_idx + day]
        max_price = max(max_price, row['最高'])
        if (row['最低'] - buy_price) / buy_price * 100 <= STOP_LOSS: return STOP_LOSS
        profit = (max_price - buy_price) / buy_price * 100
        if profit >= TRAILING_START:
            if (max_price - row['收盘']) / (max_price - buy_price) >= 0.3:
                return max((row['收盘'] - buy_price) / buy_price * 100, 2.0)
    return (df.iloc[min(start_idx + max_days, len(df)-1)]['收盘'] - buy_price) / buy_price * 100

def process(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        res = []
        for i in range(60, len(df) - 30):
            row = df.iloc[i]
            # 增加 Bias 逻辑：防止在远离均线的自由落体中接飞刀
            if (row['rsi6'] <= RSI6_MAX and row['kdj_k'] <= KDJ_K_MAX and
                row['bias20'] >= MAX_BIAS_20 and # 跌幅要在合理范围内
                row['收盘'] >= row['ma5'] and   # 确认站上5日线
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO and
                ((row['ma60'] - row['收盘']) / row['收盘'] * 100) >= MIN_PROFIT_POTENTIAL):
                
                trade = {'代码': os.path.basename(file_path)[:6], '日期': row['日期']}
                for p in HOLD_PERIODS:
                    trade[f'{p}日收益'] = simulate_trade(df, i, p)
                res.append(trade)
        return res
    except: return []

def main():
    print("🚀 正在执行【质量增强版】回测...")
    files = glob.glob(os.path.join('stock_data', '*.csv'))
    with Pool(cpu_count()) as p:
        all_res = [t for sub in p.map(process, files) for t in sub]
    if not all_res: return
    df = pd.DataFrame(all_res)
    summary = []
    for p in HOLD_PERIODS:
        col = f'{p}日收益'
        summary.append({'周期': f'{p}天', '胜率': f'{(df[col]>0).sum()/len(df)*100:.2f}%', '平均收益': f'{df[col].mean():.2f}%'})
    print(pd.DataFrame(summary).to_string(index=False))
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/backtest_advanced_summary.csv', index=False, encoding='utf_8_sig')

if __name__ == "__main__": main()
