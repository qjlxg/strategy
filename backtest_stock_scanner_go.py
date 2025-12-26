import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 保持与原脚本一致的参数 ===================
MIN_PRICE = 5.0
MAX_AVG_TURNOVER_30 = 2.5
MIN_VOLUME_RATIO = 0.2
MAX_VOLUME_RATIO = 0.85
RSI6_MAX = 25
KDJ_K_MAX = 30
MIN_PROFIT_POTENTIAL = 15
MAX_TODAY_CHANGE = 1.5
HOLD_DAYS = 5  # 默认持仓 5 个交易日进行回测
# =============================================================

def calculate_indicators(df):
    """计算核心指标 (逻辑与原脚本完全一致)"""
    df = df.reset_index(drop=True)
    close = df['收盘']
    
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi6'] = 100 - (100 / (1 + rs))
    
    # KDJ (9,3,3)
    low_list = df['最低'].rolling(window=9).min()
    high_list = df['最高'].rolling(window=9).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    
    # MA & Turnover
    df['ma5'] = close.rolling(window=5).mean()
    df['ma60'] = close.rolling(window=60).mean()
    df['avg_turnover_30'] = df['换手率'].rolling(window=30).mean()
    df['vol_ma5'] = df['成交量'].shift(1).rolling(window=5).mean()
    df['vol_ratio'] = df['成交量'] / df['vol_ma5']
    
    return df

def backtest_single_stock(file_path):
    """对单只股票进行历史滚动回测"""
    stock_code = os.path.basename(file_path).split('.')[0]
    try:
        df_raw = pd.read_csv(file_path)
        if len(df_raw) < 70: return []
        
        df = calculate_indicators(df_raw)
        trades = []

        # 从第60行开始回测，确保指标已计算完成
        for i in range(60, len(df) - HOLD_DAYS):
            row = df.iloc[i]
            
            # 策略准入条件判断
            potential = (row['ma60'] - row['收盘']) / row['收盘'] * 100
            change = row['涨跌幅'] if '涨跌幅' in row else 0
            
            condition = (
                row['收盘'] >= MIN_PRICE and
                row['avg_turnover_30'] <= MAX_AVG_TURNOVER_30 and
                potential >= MIN_PROFIT_POTENTIAL and
                change <= MAX_TODAY_CHANGE and
                row['rsi6'] <= RSI6_MAX and
                row['kdj_k'] <= KDJ_K_MAX and
                row['收盘'] >= row['ma5'] and
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO
            )

            if condition:
                # 模拟操作：信号日收盘买入，HOLD_DAYS 后收盘卖出
                buy_price = row['收盘']
                sell_price = df.iloc[i + HOLD_DAYS]['收盘']
                profit_pct = (sell_price - buy_price) / buy_price * 100
                
                trades.append({
                    '代码': stock_code,
                    '日期': row['日期'],
                    '买入价': round(buy_price, 2),
                    '卖出价': round(sell_price, 2),
                    '收益率': round(profit_pct, 2)
                })
        return trades
    except:
        return []

def main():
    print("🔎 开始历史策略回测 (信号触发 5 日后卖出)...")
    file_list = glob.glob(os.path.join('stock_data', '*.csv'))
    
    with Pool(processes=cpu_count()) as pool:
        raw_results = pool.map(backtest_single_stock, file_list)

    all_trades = [t for sublist in raw_results for t in sublist]
    
    if all_trades:
        df_bt = pd.DataFrame(all_trades)
        win_rate = (df_bt['收益率'] > 0).sum() / len(df_bt) * 100
        avg_profit = df_bt['收益率'].mean()
        
        print(f"\n✅ 回测完成!")
        print(f"总交易次数: {len(df_bt)}")
        print(f"胜率: {win_rate:.2f}%")
        print(f"平均收益率: {avg_profit:.2f}%")
        
        # 保存回测结果明细
        os.makedirs('results', exist_ok=True)
        df_bt.to_csv('results/backtest_summary.csv', index=False, encoding='utf_8_sig')
    else:
        print("\n❌ 历史数据中未发现符合策略条件的信号。")

if __name__ == "__main__":
    main()
