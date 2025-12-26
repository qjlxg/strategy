import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 严格引用原脚本参数 (不可改动) ===================
MIN_PRICE = 5.0
MAX_AVG_TURNOVER_30 = 2.5
MIN_VOLUME_RATIO = 0.2
MAX_VOLUME_RATIO = 0.85
RSI6_MAX = 25
KDJ_K_MAX = 30
MIN_PROFIT_POTENTIAL = 15
MAX_TODAY_CHANGE = 1.5

# ==================== 增强版回测参数 (风险控制) =====================
STOP_LOSS = -5.0          # 固定止损 5%
TRAILING_START = 10.0     # 移动止盈触发门槛 10%
# 持有期对比：5, 7, 15, 30天
HOLD_PERIODS = [5, 7, 15, 30]

def calculate_indicators(df):
    """逻辑同步原脚本"""
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
    """
    模拟交易逻辑：含 5% 止损和 10% 起步的移动止盈
    """
    buy_price = df.iloc[start_idx]['收盘']
    max_price_since_buy = buy_price
    
    for day in range(1, max_days + 1):
        if start_idx + day >= len(df):
            break
            
        curr_row = df.iloc[start_idx + day]
        curr_close = curr_row['收盘']
        curr_low = curr_row['最低']
        
        # 更新持有期间最高价
        max_price_since_buy = max(max_price_since_buy, curr_row['最高'])
        curr_profit = (curr_close - buy_price) / buy_price * 100
        max_profit = (max_price_since_buy - buy_price) / buy_price * 100

        # 1. 触发固定止损 (以盘中最低价触发)
        if (curr_low - buy_price) / buy_price * 100 <= STOP_LOSS:
            return STOP_LOSS
            
        # 2. 移动止盈逻辑
        # 如果最高涨幅曾达到 TRAILING_START (10%)，则当价格从最高点回撤 30% 时止盈
        if max_profit >= TRAILING_START:
            drawback = (max_price_since_buy - curr_close) / (max_price_since_buy - buy_price)
            if drawback >= 0.3: # 回撤 30% 保护利润
                return max(curr_profit, 2.0) # 确保至少保留一部分利润

    # 时间到，按收盘价卖出
    return (df.iloc[min(start_idx + max_days, len(df)-1)]['收盘'] - buy_price) / buy_price * 100

def backtest_single_stock(file_path):
    stock_code = os.path.basename(file_path).split('.')[0]
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        
        stock_signals = []
        for i in range(60, len(df) - 30):
            row = df.iloc[i]
            # 严格遵循原脚本过滤逻辑
            potential = (row['ma60'] - row['收盘']) / row['收盘'] * 100
            if (row['收盘'] >= MIN_PRICE and 
                row['avg_turnover_30'] <= MAX_AVG_TURNOVER_30 and
                potential >= MIN_PROFIT_POTENTIAL and
                row['rsi6'] <= RSI6_MAX and 
                row['kdj_k'] <= KDJ_K_MAX and
                row['收盘'] >= row['ma5'] and
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO):
                
                # 计算不同周期的实际收益
                res = {'代码': stock_code, '日期': row['日期']}
                for p in HOLD_PERIODS:
                    res[f'{p}日收益'] = simulate_trade(df, i, p)
                stock_signals.append(res)
        return stock_signals
    except:
        return []

def main():
    print("🚀 正在执行带风控的高级回测...")
    files = glob.glob(os.path.join('stock_data', '*.csv'))
    with Pool(processes=cpu_count()) as pool:
        raw = pool.map(backtest_single_stock, files)
    
    all_trades = [t for sub in raw for t in sub]
    if not all_trades:
        print("❌ 历史数据中未匹配到策略信号")
        return

    df_res = pd.DataFrame(all_trades)
    
    print("\n--- 策略性能看板 (含止损与移动止盈) ---")
    summary = []
    for p in HOLD_PERIODS:
        col = f'{p}日收益'
        win_rate = (df_res[col] > 0).sum() / len(df_res) * 100
        avg_ret = df_res[col].mean()
        summary.append({'周期': f'{p}天', '胜率': f'{win_rate:.2f}%', '平均收益': f'{avg_ret:.2f}%'})
    
    print(pd.DataFrame(summary).to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/backtest_advanced_summary.csv', index=False, encoding='utf_8_sig')
    print("\n✅ 详细明细已存至 results/backtest_advanced_summary.csv")

if __name__ == "__main__":
    main()
