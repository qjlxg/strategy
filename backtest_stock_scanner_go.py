import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 2025“防假突破”实战参数 ===================
MIN_PRICE = 5.0              
MAX_AVG_TURNOVER_30 = 2.0    

# --- 选股逻辑优化：拒绝僵尸，温和确认 ---
MIN_VOLUME_RATIO = 0.5       
MAX_VOLUME_RATIO = 1.2       

# --- 极致超跌 + 乖离过滤 ---
RSI6_MAX = 28                
KDJ_K_MAX = 25               
MIN_PROFIT_POTENTIAL = 18    

# --- 核心：防假突破确认信号 ---
STAND_STILL_THRESHOLD = 1.005 # 必须站上5日线0.5%
MIN_BIAS_20 = -18            # 乖离率下限（防暴雷）
MAX_BIAS_20 = -8             # 乖离率上限（保动力）
MAX_TODAY_CHANGE = 4.0       

# --- 强化交易规则 ---
STOP_LOSS = -5.0             # 强制止损
TRAILING_START = 8.0         # 8%开启移动止盈
LIFE_LINE_DAY = 3            # 3日生命线：第3天不涨(>1%)则离场
# =============================================================

def calculate_indicators(df):
    df = df.reset_index(drop=True)
    close = df['收盘']
    vol = df['成交量']
    
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['rsi6'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    # KDJ
    low_9 = close.rolling(9).min()
    high_9 = close.rolling(9).max()
    df['kdj_k'] = ((close - low_9) / (high_9 - low_9) * 100).ewm(com=2).mean()
    
    # MA & BIAS
    df['ma5'] = close.rolling(5).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    df['bias20'] = (close - df['ma20']) / df['ma20'] * 100
    
    # 均线斜率逻辑：当前5日线下降速度比昨天变慢，说明跌势放缓
    ma5_diff = df['ma5'].diff()
    df['slope_slowing'] = ma5_diff > ma5_diff.shift(1)
    
    # 量能确认
    df['vol_ma5'] = vol.shift(1).rolling(5).mean()
    df['vol_ratio'] = vol / df['vol_ma5']
    df['vol_increase'] = vol > vol.shift(1) 
    
    return df

def simulate_trade(df, start_idx, max_days):
    buy_price = df.iloc[start_idx]['收盘']
    max_p = buy_price
    
    for d in range(1, max_days + 1):
        if start_idx + d >= len(df): break
        row = df.iloc[start_idx + d]
        max_p = max(max_p, row['最高'])
        
        # 1. 触发止损
        if (row['最低'] - buy_price) / buy_price * 100 <= STOP_LOSS:
            return STOP_LOSS
            
        # 2. 核心优化：3日生命线离场
        # 如果到了第3天，收盘利润还不足1%，说明反弹动能不足，止平离场
        if d == LIFE_LINE_DAY:
            current_profit = (row['收盘'] - buy_price) / buy_price * 100
            if current_profit < 1.0:
                return current_profit

        # 3. 移动止盈 (回撤保护)
        profit = (max_p - buy_price) / buy_price * 100
        if profit >= TRAILING_START:
            drawback = (max_p - row['收盘']) / (max_p - buy_price)
            if drawback >= 0.25:
                return max((row['收盘'] - buy_price) / buy_price * 100, 1.5)
                
    return (df.iloc[min(start_idx + max_days, len(df)-1)]['收盘'] - buy_price) / buy_price * 100

def process_file(f):
    try:
        df = pd.read_csv(f)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        res = []
        for i in range(60, len(df) - 20):
            row = df.iloc[i]
            potential = (row['ma60'] - row['收盘']) / row['收盘'] * 100
            change = (row['收盘'] - df.iloc[i-1]['收盘']) / df.iloc[i-1]['收盘'] * 100
            
            # 综合判断：超跌 + 乖离 + 站稳确认 + 斜率趋缓 + 量增
            if (row['rsi6'] <= RSI6_MAX and row['kdj_k'] <= KDJ_K_MAX and
                MIN_BIAS_20 <= row['bias20'] <= MAX_BIAS_20 and
                row['收盘'] >= row['ma5'] * STAND_STILL_THRESHOLD and
                row['slope_slowing'] and                # 均线低位走平趋势
                row['vol_increase'] and                 # 带量站上
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO and
                potential >= MIN_PROFIT_POTENTIAL and
                change <= MAX_TODAY_CHANGE):
                
                trade = {'代码': os.path.basename(f)[:6], '日期': row['日期']}
                for p in [3, 5, 10, 20]:
                    trade[f'{p}日收益'] = simulate_trade(df, i, p)
                res.append(trade)
        return res
    except: return []

def main():
    print(f"🚀 启动最终实战强化版回测...")
    files = glob.glob(os.path.join('stock_data', '*.csv'))
    with Pool(cpu_count()) as p:
        raw = p.map(process_file, files)
    all_t = [t for sub in raw for t in sub]
    if not all_t: return
    
    df_res = pd.DataFrame(all_t)
    print("\n--- 强化策略看板 (含3日生命线) ---")
    sum_d = []
    for p in [3, 5, 10, 20]:
        c = f'{p}日收益'
        sum_d.append({
            '周期': f'{p}天',
            '胜率': f'{(df_res[c]>0).sum()/len(df_res)*100:.2f}%',
            '平均收益': f'{df_res[c].mean():.2f}%',
            '信号总数': len(df_res)
        })
    print(pd.DataFrame(sum_d).to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/backtest_final_pro_summary.csv', index=False, encoding='utf_8_sig')

if __name__ == "__main__":
    main()
