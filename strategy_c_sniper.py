import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# --- 核心配置 ---
DATA_DIR = "stock_data"
NAME_MAP_FILE = 'stock_names.csv'

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # 1. 均线系统 (V5 多头排列)
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA5V'] = vol.rolling(5).mean()

    # 2. RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))

    # 3. KDJ
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    # 4. MACD (动能加速)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = (df['DIF'] - df['DEA']) * 2
    
    return df

def process_file(file_path):
    code = os.path.basename(file_path).split('.')[0]
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return None
        df.columns = df.columns.str.strip()
        df.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','最高':'High','最低':'Low','成交量':'Volume','成交额':'Amount'}, inplace=True)
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # --- V5 实战硬性条件 ---
        # 1. 趋势：均线多头排列
        is_trend = (curr['MA5'] > curr['MA10'] > curr['MA20'])
        
        # 2. 突破：40日新高且收阳
        prev_high_40 = df['High'].iloc[-41:-1].max()
        is_breakout = (curr['Close'] > prev_high_40 * 1.01) and (curr['Close'] > curr['Open'])
        
        # 3. 量能：温和放量 (2-4.5倍)
        is_vol = (2.0 * curr['MA5V'] < curr['Volume'] < 4.5 * curr['MA5V'])
        
        # 4. 动能：RSI强势区 + MACD红柱加速 (1.1倍)
        is_rsi = (65 < curr['RSI6'] < 82)
        is_kdj = (curr['K'] > curr['D']) and (prev['K'] <= prev['D'])
        is_macd = (curr['DIF'] > curr['DEA']) and (curr['MACD_HIST'] > prev['MACD_HIST'] * 1.1)

        if is_trend and is_breakout and is_vol and is_rsi and is_kdj and is_macd:
            # 计算明天实战的入场上限 (高开不超 4.5%)
            buy_limit = curr['Close'] * 1.045
            
            return {
                "代码": code, 
                "今日收盘": round(curr['Close'], 2), 
                "成交额(万)": round(curr['Amount']/10000, 0),
                "RSI6": round(curr['RSI6'], 1),
                "MACD增速": round(curr['MACD_HIST'] / prev['MACD_HIST'], 2) if prev['MACD_HIST'] != 0 else 0,
                "明天买入上限": round(buy_limit, 2)
            }
    except:
        return None

def main():
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🚀 [V5实战版] 启动深度扫描: {len(files)} 只个股...")

    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, files)

    found = [r for r in results if r is not None]
    
    final_list = []
    if found:
        for item in found:
            name = names_dict.get(item['代码'], "未知")
            if any(x in name for x in ["ST", "退"]): continue
            item['名称'] = name
            final_list.append(item)
            
    if final_list:
        df_res = pd.DataFrame(final_list)
        df_res = df_res.sort_values(by='成交额(万)', ascending=False)
        
        print("\n" + "★"*10 + " V5 策略明日实战监控名单 " + "★"*10)
        print(df_res[['代码', '名称', '今日收盘', '明天买入上限', '成交额(万)', 'MACD增速']].to_string(index=False))
        print("★"*45)
        print("💡 实战提醒：明天 9:25 集合竞价若价格超过[明天买入上限]，请务必放弃！")
        
        now = datetime.now()
        df_res.to_csv(f"Daily_Sniper_V5_{now.strftime('%Y%m%d')}.csv", index=False, encoding='utf-8-sig')
    else:
        print("\n当前市场无符合 V5 核心条件的信号。建议空仓等待或复盘近期妖股规律。")

if __name__ == "__main__":
    main()
