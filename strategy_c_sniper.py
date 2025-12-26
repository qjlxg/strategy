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

    # 指标计算（与 V6 回测完全一致）
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA5V'] = vol.rolling(5).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

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
        curr, prev = df.iloc[-1], df.iloc[-2]

        # --- 基础过滤条件 (所有名单共有) ---
        prev_high_40 = df['High'].iloc[-41:-1].max()
        is_breakout = (curr['Close'] > prev_high_40 * 1.01) and (curr['Close'] > curr['Open'])
        is_vol = (2.0 * curr['MA5V'] < curr['Volume'] < 4.5 * curr['MA5V'])
        is_rsi_base = (60 < curr['RSI6'] < 85)
        
        # --- V6 严苛条件 (用于正式名单) ---
        is_trend_v6 = (curr['MA5'] > curr['MA10'] > curr['MA20'])
        is_macd_v6 = (curr['DIF'] > curr['DEA']) and (curr['MACD_HIST'] > prev['MACD_HIST'] * 1.1)
        is_kdj_v6 = (curr['K'] > curr['D']) and (prev['K'] <= prev['D'])

        data = {
            "代码": code, "价格": round(curr['Close'], 2), 
            "额(万)": round(curr['Amount']/10000, 0), "RSI6": round(curr['RSI6'], 1),
            "MACD速": round(curr['MACD_HIST']/prev['MACD_HIST'], 2) if prev['MACD_HIST']!=0 else 0,
            "上限": round(curr['Close'] * 1.045, 2)
        }

        # 逻辑判定
        if is_breakout and is_vol and is_rsi_base:
            if is_trend_v6 and is_macd_v6 and is_kdj_v6:
                data["类型"] = "正式信号"
                return data
            elif curr['MA5'] > curr['MA20'] and curr['MACD_HIST'] > 0:
                data["类型"] = "观察储备"
                return data
                
    except: return None

def main():
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"🔍 [V6 复合扫描] 正在分析 {len(files)} 只个股...")
    with Pool(cpu_count()) as pool:
        results = [r for r in pool.map(process_file, files) if r is not None]

    final_results = []
    for item in results:
        name = names_dict.get(item['代码'], "未知")
        if any(x in name for x in ["ST", "退"]): continue
        item['名称'] = name
        final_results.append(item)

    if not final_results:
        print("❌ 全市场暂无符合条件的个股。")
        return

    df_res = pd.DataFrame(final_results)
    
    # 1. 打印正式信号
    official = df_res[df_res['类型'] == "正式信号"]
    if not official.empty:
        print("\n" + "★"*10 + " V6 正式实战信号 (明日高开限价买入) " + "★"*10)
        print(official[['代码', '名称', '价格', '上限', '额(万)', 'MACD速']].to_string(index=False))
    else:
        print("\n[!] 今日无正式严苛信号。")

    # 2. 打印观察名单
    observer = df_res[df_res['类型'] == "观察储备"].sort_values(by='额(万)', ascending=False).head(15)
    if not observer.empty:
        print("\n" + "⊙"*10 + " 潜力观察名单 (蓄势待发/强度略欠) " + "⊙"*10)
        print(observer[['代码', '名称', '价格', '额(万)', 'RSI6']].to_string(index=False))
        print("💡 观察名单建议：关注其回踩MA5的机会，若明日MACD增速补足则转为正式信号。")

if __name__ == "__main__":
    main()
