import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 2025“防假突破”实战参数 ===================
MIN_PRICE = 5.0              
MAX_AVG_TURNOVER_30 = 2.0    

# --- 1. 量能确认：拒绝僵尸股，转向温和放量确认 ---
MIN_VOLUME_RATIO = 0.5       
MAX_VOLUME_RATIO = 1.2       

# --- 2. 极致超跌 + 空间要求 ---
RSI6_MAX = 28                
KDJ_K_MAX = 25               
MIN_PROFIT_POTENTIAL = 18    

# --- 3. 核心：跌势衰竭与站稳确认 ---
STAND_STILL_THRESHOLD = 1.005 # 必须站上5日线0.5%
MIN_BIAS_20 = -18            
MAX_BIAS_20 = -8             
MAX_TODAY_CHANGE = 4.0       

# --- 4. 交易逻辑（针对56%胜率优化） ---
STOP_LOSS = -5.0             # 强制止损线
TRAILING_START = 8.0         # 盈利8%开启移动止盈保护
LIFE_LINE_DAY = 3            # 3日生命线：第3天利润不足1%则离场
# =============================================================

def calculate_indicators(df):
    df = df.reset_index(drop=True)
    close = df['收盘']
    vol = df['成交量']
    
    # 基础指标
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['rsi6'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    low_9 = close.rolling(9).min()
    high_9 = close.rolling(9).max()
    df['kdj_k'] = ((close - low_9) / (high_9 - low_9) * 100).ewm(com=2).mean()
    
    df['ma5'] = close.rolling(5).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    df['bias20'] = (close - df['ma20']) / df['ma20'] * 100
    
    # 核心判断逻辑：5日线斜率趋缓
    ma5_diff = df['ma5'].diff()
    df['slope_slowing'] = ma5_diff > ma5_diff.shift(1)
    
    # 量能
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
        
        current_profit = (row['收盘'] - buy_price) / buy_price * 100
        
        # 1. 触发止损
        if (row['最低'] - buy_price) / buy_price * 100 <= STOP_LOSS:
            return STOP_LOSS
            
        # 2. 核心保护：3日生命线
        if d == LIFE_LINE_DAY and current_profit < 1.0:
            return current_profit

        # 3. 移动止盈 (回撤25%离场)
        profit_peak = (max_p - buy_price) / buy_price * 100
        if profit_peak >= TRAILING_START:
            drawback = (max_p - row['收盘']) / (max_p - buy_price)
            if drawback >= 0.25:
                return max(current_profit, 1.5) # 确保至少保留一部分利润
                
    return (df.iloc[min(start_idx + max_days, len(df)-1)]['收盘'] - buy_price) / buy_price * 100

def process_file(f):
    try:
        df = pd.read_csv(f)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        res = []
        for i in range(60, len(df) - 30):
            row = df.iloc[i]
            potential = (row['ma60'] - row['收盘']) / row['收盘'] * 100
            change = (row['收盘'] - df.iloc[i-1]['收盘']) / df.iloc[i-1]['收盘'] * 100
            avg_turnover_30 = df['换手率'].rolling(30).mean().iloc[i]

            if (row['rsi6'] <= RSI6_MAX and row['kdj_k'] <= KDJ_K_MAX and
                MIN_BIAS_20 <= row['bias20'] <= MAX_BIAS_20 and
                row['收盘'] >= row['ma5'] * STAND_STILL_THRESHOLD and
                row['slope_slowing'] and                
                row['vol_increase'] and                 
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO and
                avg_turnover_30 <= MAX_AVG_TURNOVER_30 and
                potential >= MIN_PROFIT_POTENTIAL and
                change <= MAX_TODAY_CHANGE):
                
                trade = {'代码': os.path.basename(f)[:6], '日期': row['日期']}
                for p in [3, 5, 10, 20]:
                    trade[f'{p}日收益'] = simulate_trade(df, i, p)
                res.append(trade)
        return res
    except: return []

def main():
    print(f"🚀 执行最终强化回测逻辑...")
    files = glob.glob(os.path.join('stock_data', '*.csv'))
    with Pool(cpu_count()) as p:
        raw = p.map(process_file, files)
    all_t = [t for sub in raw for t in sub]
    if not all_t: 
        print("未发现信号")
        return
    
    df_res = pd.DataFrame(all_t)
    print("\n--- 优化后实战看板 ---")
    summary = []
    for p in [3, 5, 10, 20]:
        c = f'{p}日收益'
        summary.append({
            '周期': f'{p}天',
            '胜率': f'{(df_res[c]>0).sum()/len(df_res)*100:.2f}%',
            '平均收益': f'{df_res[c].mean():.2f}%',
            '信号数': len(df_res)
        })
    print(pd.DataFrame(summary).to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/backtest_final_optimized.csv', index=False, encoding='utf_8_sig')

if __name__ == "__main__":
    main()
