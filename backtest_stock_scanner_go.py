import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 基于实战经验优化的参数 ===================
MIN_PRICE = 5.0
MAX_AVG_TURNOVER_30 = 2.0     # 评论经验：筹码锁定更重要，换手率越低说明浮筹越少
MIN_VOLUME_RATIO = 0.5        # 经验：避开量比0.3以下的僵尸股，至少要有0.5才表示有活钱
MAX_VOLUME_RATIO = 1.1        # 经验：允许“微幅放量”确认止跌
RSI6_MAX = 28                 # 均衡点：比25放宽一点，捕捉更多机会
KDJ_K_MAX = 25                
MIN_PROFIT_POTENTIAL = 18     # 反弹目标提高，确保赔率

# 新增：20日乖离率控制（BIAS20）
# 经验：BIAS低于-10%通常有反弹需求，但跌过-20%可能是基本面出事，取-7%到-18%之间
MIN_BIAS_20 = -18
MAX_BIAS_20 = -7

STOP_LOSS = -5.0              # 强制止损
TRAILING_START = 10.0         # 移动止盈起点
HOLD_PERIODS = [5, 7, 15, 30]

def calculate_indicators(df):
    df = df.reset_index(drop=True)
    close = df['收盘']
    
    # RSI6 & KDJ
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['rsi6'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    low_9 = df['最低'].rolling(9).min()
    high_9 = df['最高'].rolling(9).max()
    df['kdj_k'] = ((close - low_9) / (high_9 - low_9) * 100).ewm(com=2).mean()
    
    # BIAS & MA
    df['ma5'] = close.rolling(5).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    df['bias20'] = (close - df['ma20']) / df['ma20'] * 100
    
    # 量能
    df['vol_ma5'] = df['成交量'].shift(1).rolling(5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']
    df['avg_turnover_30'] = df['换手率'].rolling(30).mean()
    
    return df

def simulate_trade(df, start_idx, max_days):
    buy_price = df.iloc[start_idx]['收盘']
    max_p = buy_price
    for d in range(1, max_days + 1):
        if start_idx + d >= len(df): break
        row = df.iloc[start_idx + d]
        max_p = max(max_p, row['最高'])
        if (row['最低'] - buy_price) / buy_price * 100 <= STOP_LOSS: return STOP_LOSS
        curr_p = (row['收盘'] - buy_price) / buy_price * 100
        if (max_p - buy_price) / buy_price * 100 >= TRAILING_START:
            if (max_p - row['收盘']) / (max_p - buy_price) >= 0.3: return max(curr_p, 2.0)
    return (df.iloc[min(start_idx + max_days, len(df)-1)]['收盘'] - buy_price) / buy_price * 100

def process(f):
    try:
        df = pd.read_csv(f)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        trades = []
        for i in range(60, len(df) - 30):
            row = df.iloc[i]
            # 综合评论经验的筛选条件
            if (row['rsi6'] <= RSI6_MAX and row['kdj_k'] <= KDJ_K_MAX and
                MIN_BIAS_20 <= row['bias20'] <= MAX_BIAS_20 and # 乖离率合理区间
                row['收盘'] >= row['ma5'] and                  # 必须站上5日线止跌
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO and
                ((row['ma60'] - row['收盘']) / row['收盘'] * 100) >= MIN_PROFIT_POTENTIAL):
                
                t = {'代码': os.path.basename(f)[:6], '日期': row['日期']}
                for p in HOLD_PERIODS: t[f'{p}日收益'] = simulate_trade(df, i, p)
                trades.append(t)
        return trades
    except: return []

def main():
    print("🚀 正在执行【社区经验增强版】回测...")
    files = glob.glob(os.path.join('stock_data', '*.csv'))
    with Pool(cpu_count()) as p:
        all_t = [t for sub in p.map(process, files) for t in sub]
    if not all_t: return
    df = pd.DataFrame(all_t)
    sum_data = []
    for p in HOLD_PERIODS:
        c = f'{p}日收益'
        sum_data.append({'周期': f'{p}天', '胜率': f'{(df[c]>0).sum()/len(df)*100:.2f}%', '平均收益': f'{df[c].mean():.2f}%'})
    print("\n" + pd.DataFrame(sum_data).to_string(index=False))
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/backtest_advanced_summary.csv', index=False, encoding='utf_8_sig')

if __name__ == "__main__": main()
