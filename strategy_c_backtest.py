import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# --- 核心配置 ---
DATA_DIR = "stock_data"
NAME_MAP_FILE = 'stock_names.csv'
LOOKBACK_WINDOW = 120  
HOLD_DAYS = 10         
STOP_LOSS_PCT = -5.0   

def calculate_indicators(df):
    close = df['Close']
    # 1. 均线系统 (V6 同步实战)
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA5V'] = df['Volume'].rolling(5).mean()
    
    # 2. RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))
    
    # 3. KDJ
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # 4. MACD (V6 加速逻辑)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = (df['DIF'] - df['DEA']) * 2
    
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
            
            # --- V6 同步实战筛选条件 ---
            # 条件1：均线发散（MA5 > MA10 > MA20）
            is_trend = (curr['MA5'] > curr['MA10'] > curr['MA20'])
            
            # 条件2：突破40日新高且收阳线
            prev_high_40 = df['High'].iloc[i-40:i].max()
            is_breakout = (curr['Close'] > prev_high_40 * 1.01) and (curr['Close'] > curr['Open'])
            
            # 条件3：MACD红柱加速 (对齐实战 1.1倍)
            is_macd = (curr['DIF'] > curr['DEA']) and (curr['MACD_HIST'] > prev['MACD_HIST'] * 1.1)
            
            # 条件4：RSI强势区 + KDJ金叉
            is_rsi = (65 < curr['RSI6'] < 82)
            is_kdj = (curr['K'] > curr['D']) and (prev['K'] <= prev['D'])
            
            # 条件5：温和放量
            is_vol = (2.0 * curr['MA5V'] < curr['Volume'] < 4.5 * curr['MA5V'])

            if is_trend and is_breakout and is_macd and is_rsi and is_kdj and is_vol:
                # --- V6 核心：实战入场限制 ---
                # 获取次日数据
                next_day = df.iloc[i+1]
                # 计算次日开盘涨幅
                open_jump = ((next_day['Open'] - curr['Close']) / curr['Close']) * 100
                
                # 如果次日高开超过 4.5%，实战中我们会放弃，所以回测也要剔除
                if not (-1.0 < open_jump < 4.5): continue 

                post_df = df.iloc[i+1 : i+1+HOLD_DAYS]
                if post_df.empty: continue
                
                final_ret, max_reach, is_stopped = 0.0, 0.0, False
                triggered_price = curr['Close']
                
                # 逐日追踪 10 日表现
                for _, row in post_df.iterrows():
                    day_high_reach = ((row['High'] - triggered_price) / triggered_price) * 100
                    max_reach = max(max_reach, day_high_reach)
                    
                    # 5% 强制止损 (实战守则)
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
                    "MACD增速": round(curr['MACD_HIST'] / prev['MACD_HIST'], 2) if prev['MACD_HIST'] != 0 else 0,
                    "状态": "止损离场" if is_stopped else "持有期满"
                })
        return results
    except: return None

def main():
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"⌛ [V6同步版] 正在对齐实战逻辑并运行回测...")
    
    with Pool(cpu_count()) as pool:
        all_results = pool.map(run_backtest_on_file, files)
    
    flattened = [item for sublist in all_results if sublist for item in sublist]
    if not flattened:
        print("未发现符合 V6 严苛条件的信号。")
        return

    res_df = pd.DataFrame(flattened)
    res_df['名称'] = res_df['代码'].apply(lambda x: names_dict.get(x, "未知"))
    
    # 统计核心数据
    total = len(res_df)
    wins = len(res_df[res_df['持有10日收益%'] > 0])
    win_rate = (wins / total) * 100
    avg_ret = res_df['持有10日收益%'].mean()
    
    # 目录与保存
    now = datetime.now()
    dir_name = "backtest_reports/" + now.strftime("%Y-%m")
    if not os.path.exists(dir_name): os.makedirs(dir_name)
    save_path = os.path.join(dir_name, f"C_Strategy_V6_Sync_{now.strftime('%Y%m%d_%H%M')}.csv")
    res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*35)
    print(f"📊 策略 V6 (实战同步) 回测报告")
    print("-" * 35)
    print(f"📂 信号总数: {total}")
    print(f"📈 最终胜率: {win_rate:.2f}%")
    print(f"💰 平均收益: {avg_ret:.2f}%")
    print(f"🚀 结果已推送到: {save_path}")
    print("="*35 + "\n")

if __name__ == "__main__":
    main()
