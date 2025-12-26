import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 优化版回测参数 (提升频率与确认感) ===================
MIN_PRICE = 5.0
MAX_AVG_TURNOVER_30 = 2.5

# 优化 1：放宽量比至 1.05，允许“平量”或“微幅放量”止跌
MIN_VOLUME_RATIO = 0.2
MAX_VOLUME_RATIO = 1.05      

# 优化 2：放宽 RSI 指标至 30，增加符合条件的个股基数
RSI6_MAX = 30                
KDJ_K_MAX = 30               
MIN_PROFIT_POTENTIAL = 15
MAX_TODAY_CHANGE = 1.5

# 风控参数
STOP_LOSS = -5.0          
TRAILING_START = 10.0     
HOLD_PERIODS = [5, 7, 15, 30]

def calculate_indicators(df):
    """计算指标"""
    df = df.reset_index(drop=True)
    close = df['收盘']
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi6'] = 100 - (100 / (1 + rs))
    
    low_list = df['最低'].rolling(window=9).min()
    high_list = df['最高'].rolling(window=9).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    
    df['ma5'] = close.rolling(window=5).mean()
    df['ma60'] = close.rolling(window=60).mean()
    df['avg_turnover_30'] = df['换手率'].rolling(window=30).mean()
    df['vol_ma5'] = df['成交量'].shift(1).rolling(window=5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']
    return df

def simulate_trade(df, start_idx, max_days):
    """模拟交易逻辑：含止损和移动止盈"""
    buy_price = df.iloc[start_idx]['收盘']
    max_price_since_buy = buy_price
    
    for day in range(1, max_days + 1):
        if start_idx + day >= len(df): break
            
        curr_row = df.iloc[start_idx + day]
        max_price_since_buy = max(max_price_since_buy, curr_row['最高'])
        
        # 触发固定止损
        if (curr_row['最低'] - buy_price) / buy_price * 100 <= STOP_LOSS:
            return STOP_LOSS
            
        # 移动止盈逻辑
        max_profit = (max_price_since_buy - buy_price) / buy_price * 100
        if max_profit >= TRAILING_START:
            drawback = (max_price_since_buy - curr_row['收盘']) / (max_price_since_buy - buy_price)
            if drawback >= 0.3: # 回撤 30% 保护
                return max((curr_row['收盘'] - buy_price) / buy_price * 100, 2.0)

    end_idx = min(start_idx + max_days, len(df) - 1)
    return (df.iloc[end_idx]['收盘'] - buy_price) / buy_price * 100

def backtest_single_stock(file_path):
    stock_code = os.path.basename(file_path).split('.')[0]
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        
        stock_signals = []
        for i in range(60, len(df) - 30):
            row = df.iloc[i]
            potential = (row['ma60'] - row['收盘']) / row['收盘'] * 100
            
            # 使用优化后的阈值进行筛选
            if (row['收盘'] >= MIN_PRICE and 
                row['avg_turnover_30'] <= MAX_AVG_TURNOVER_30 and
                potential >= MIN_PROFIT_POTENTIAL and
                row['rsi6'] <= RSI6_MAX and 
                row['kdj_k'] <= KDJ_K_MAX and
                row['收盘'] >= row['ma5'] and
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO):
                
                res = {'代码': stock_code, '日期': row['日期']}
                for p in HOLD_PERIODS:
                    res[f'{p}日收益'] = simulate_trade(df, i, p)
                stock_signals.append(res)
        return stock_signals
    except:
        return []

def main():
    print("🚀 正在执行优化版回测 (放量确认+放宽RSI)...")
    file_list = glob.glob(os.path.join('stock_data', '*.csv'))
    
    with Pool(processes=cpu_count()) as pool:
        raw = pool.map(backtest_single_stock, file_list)
    
    all_trades = [t for sub in raw for t in sub]
    if not all_trades:
        print("❌ 未发现符合条件的交易信号")
        return

    df_res = pd.DataFrame(all_trades)
    print("\n--- 优化版策略性能看板 ---")
    summary = []
    for p in HOLD_PERIODS:
        col = f'{p}日收益'
        win_rate = (df_res[col] > 0).sum() / len(df_res) * 100
        avg_ret = df_res[col].mean()
        summary.append({'周期': f'{p}天', '胜率': f'{win_rate:.2f}%', '平均收益': f'{avg_ret:.2f}%'})
    
    print(pd.DataFrame(summary).to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/backtest_advanced_summary.csv', index=False, encoding='utf_8_sig')
    print(f"\n✅ 优化版明细已保存。总交易信号数: {len(df_res)}")

if __name__ == "__main__":
    main()
