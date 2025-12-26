import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# --- 配置 ---
DATA_DIR = "stock_data"
NAME_MAP_FILE = 'stock_names.csv'
LOOKBACK_WINDOW = 120  
HOLD_DAYS = 10         
STOP_LOSS_PCT = -5.0   

def calculate_indicators(df):
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA5V'] = df['Volume'].rolling(5).mean()
    df['MA3V'] = df['Volume'].rolling(3).mean()
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))
    # KDJ
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = (df['DIF'] - df['DEA']) * 2
    df['DIF_MA60'] = df['DIF'].rolling(60).mean()
    df['OBV'] = (np.sign(close.diff()) * df['Volume']).fillna(0).cumsum()
    return df

def run_backtest_on_file(file_path):
    code = os.path.basename(file_path).split('.')[0]
    if not (code.startswith('60') or code.startswith('00')): return None
    
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return None
        df.columns = df.columns.str.strip()
        df.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','最高':'High','最低':'Low','成交量':'Volume'}, inplace=True)
        df = calculate_indicators(df)
        
        results = []
        start_idx = len(df) - LOOKBACK_WINDOW
        if start_idx < 65: start_idx = 65
        
        for i in range(start_idx, len(df) - 1):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # --- C策略核心逻辑 + V4 买点强化 ---
            ma5_slope = np.polyfit(np.arange(5), df['MA5'].iloc[i-4:i+1].values, 1)[0]
            is_trend = (ma5_slope > 0) and (curr['Close'] > curr['MA20'])
            
            prev_high_40 = df['High'].iloc[i-40:i].max()
            # 突破过滤：要求收盘价真突破，且当日为阳线
            is_breakout = (curr['Close'] > prev_high_40 * 1.01) and (curr['Close'] > curr['Open'])
            
            # 量能过滤：温和放量，避免过度透支
            is_vol = (2.0 * curr['MA5V'] < curr['Volume'] < 4.5 * curr['MA5V']) and (curr['MA3V'] >= curr['MA5V'])
            
            # RSI过滤：处于强势区间但未见顶
            is_rsi = (65 < curr['RSI6'] < 82) and (curr['RSI6'] > prev['RSI6'])
            
            is_kdj = (curr['K'] > curr['D']) and (prev['K'] <= prev['D']) and (curr['K'] < 70)
            
            # MACD过滤：红柱加速增长 (斜率要求)
            is_macd = (curr['DIF'] > curr['DEA']) and (curr['DIF'] > -0.05) and \
                      (curr['DIF'] > curr['DIF_MA60']) and (curr['MACD_HIST'] > prev['MACD_HIST'] * 1.2)

            if is_trend and is_breakout and is_vol and is_rsi and is_kdj and is_macd:
                # 入场买点过滤：剔除极端高开
                next_day = df.iloc[i+1]
                open_jump = ((next_day['Open'] - curr['Close']) / curr['Close']) * 100
                if not (-1.0 < open_jump < 4.5): continue 

                post_df = df.iloc[i+1 : i+1+HOLD_DAYS]
                if post_df.empty: continue
                
                final_ret = 0.0
                max_reach = 0.0
                triggered_price = curr['Close']
                is_stopped = False
                
                for _, row in post_df.iterrows():
                    day_high_reach = ((row['High'] - triggered_price) / triggered_price) * 100
                    max_reach = max(max_reach, day_high_reach)
                    
                    # 5% 硬止损
                    day_low_ret = ((row['Low'] - triggered_price) / triggered_price) * 100
                    if day_low_ret <= STOP_LOSS_PCT:
                        final_ret = STOP_LOSS_PCT
                        is_stopped = True
                        break
                    
                    final_ret = ((row['Close'] - triggered_price) / triggered_price) * 100
                
                results.append({
                    "代码": code, "触发日期": curr['Date'], "触发价": curr['Close'],
                    "T+1开盘涨幅": round(open_jump, 2),
                    "最高冲击%": round(max_reach, 2), 
                    "持有10日收益%": round(final_ret, 2),
                    "状态": "止损离场" if is_stopped else "持有期满"
                })
        return results
    except: return None

def main():
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    with Pool(cpu_count()) as pool:
        all_results = pool.map(run_backtest_on_file, files)
    
    flattened = [item for sublist in all_results if sublist for item in sublist]
    if not flattened:
        print("V4 优化后回测期间未发现信号。")
        return

    res_df = pd.DataFrame(flattened)
    res_df['名称'] = res_df['代码'].apply(lambda x: names_dict.get(x, "未知"))
    
    # --- 统计输出 ---
    total_signals = len(res_df)
    win_count = len(res_df[res_df['持有10日收益%'] > 0])
    win_rate = (win_count / total_signals) * 100
    avg_max_reach = res_df['最高冲击%'].mean()
    avg_final_ret = res_df['持有10日收益%'].mean()
    
    now = datetime.now()
    dir_name = "backtest_reports/" + now.strftime("%Y-%m")
    os.makedirs(dir_name, exist_ok=True)
    save_path = os.path.join(dir_name, f"C_Strategy_V4_{now.strftime('%Y%m%d_%H%M')}.csv")
    res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"🚀 C策略 V4 深度优化回测报告")
    print("-" * 40)
    print(f"📂 信号总数: {total_signals}")
    print(f"📈 策略胜率: {win_rate:.2f}% (收益为正)")
    print(f"🔥 最高平均冲击: {avg_max_reach:.2f}%")
    print(f"💰 10日平均净收益: {avg_final_ret:.2f}%")
    print(f"💾 结果已保存至: {save_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
