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
    """手写核心指标，确保逻辑与 V6 回测完全一致"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # 1. 均线系统 (V6 严苛排列)
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
    """并行处理单只个股逻辑"""
    code = os.path.basename(file_path).split('.')[0]
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return None
        df.columns = df.columns.str.strip()
        df.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','最高':'High','最低':'Low','成交量':'Volume','成交额':'Amount'}, inplace=True)
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # --- 共有基础条件 (突破与量能) ---
        prev_high_40 = df['High'].iloc[-41:-1].max()
        is_breakout = (curr['Close'] > prev_high_40 * 1.01) and (curr['Close'] > curr['Open'])
        is_vol = (2.0 * curr['MA5V'] < curr['Volume'] < 4.5 * curr['MA5V'])
        is_rsi_base = (60 < curr['RSI6'] < 85)
        
        # --- V6 正式信号硬性条件 ---
        is_trend_v6 = (curr['MA5'] > curr['MA10'] > curr['MA20'])
        # MACD 增速要求 1.1 倍
        macd_growth = curr['MACD_HIST'] / prev['MACD_HIST'] if prev['MACD_HIST'] != 0 else 0
        is_macd_v6 = (curr['DIF'] > curr['DEA']) and (macd_growth > 1.1)
        is_kdj_v6 = (curr['K'] > curr['D']) and (prev['K'] <= prev['D'])

        # 结果基础数据
        data = {
            "代码": code, 
            "价格": round(curr['Close'], 2), 
            "额(万)": int(curr['Amount']/10000),
            "RSI6": round(curr['RSI6'], 1),
            "MACD速": round(macd_growth, 2),
            "上限": round(curr['Close'] * 1.045, 2)
        }

        # 逻辑判定分类
        if is_breakout and is_vol and is_rsi_base:
            if is_trend_v6 and is_macd_v6 and is_kdj_v6:
                data["类型"] = "正式信号"
                return data
            elif curr['MA5'] > curr['MA20'] and curr['MACD_HIST'] > 0:
                # 观察名单条件：放宽了均线三头排列和MACD增速限制
                data["类型"] = "观察储备"
                return data
                
    except:
        return None

def update_readme(official_df, observer_df):
    """将扫描结果格式化写入 README.md"""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    content = f"# 🏹 C-Strategy V6 每日狙击报告\n\n"
    content += f"> **最后更新**: {now_str} (北京时间)\n\n"
    
    content += "## 🚀 正式实战信号 (V6 严苛模式)\n"
    if not official_df.empty:
        # 使用 to_markdown 需要安装 tabulate 库
        content += official_df[['代码', '名称', '价格', '上限', '额(万)', 'MACD速']].to_markdown(index=False)
        content += "\n\n⚠️ **实战提示**：次日集合竞价价格若超过 **[上限]** 则放弃入场。\n"
    else:
        content += "_今日无符合 V6 严苛条件的正式信号。_\n"
    
    content += "\n---\n\n## ⊙ 潜力观察名单 (趋势蓄势中)\n"
    if not observer_df.empty:
        content += observer_df[['代码', '名称', '价格', '额(万)', 'RSI6']].to_markdown(index=False)
        content += "\n\n> 💡 **观察建议**：此类个股已具备初步突破形态，但动能尚未完全爆发。建议关注明日早盘量能及回踩 MA5 的机会。\n"
    else:
        content += "_当前市场暂无具备潜力的观察标的。_\n"
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    # 1. 加载股票名称
    if not os.path.exists(NAME_MAP_FILE):
        print(f"错误: 找不到 {NAME_MAP_FILE}")
        return
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))

    # 2. 获取数据文件并并行扫描
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        print(f"错误: {DATA_DIR} 目录下没有数据文件")
        return
        
    print(f"🔍 [V6 复合扫描] 正在分析 {len(files)} 只个股...")
    with Pool(cpu_count()) as pool:
        results = [r for r in pool.map(process_file, files) if r is not None]

    # 3. 过滤及汇总结果
    final_results = []
    for item in results:
        name = names_dict.get(item['代码'], "未知")
        if any(x in name for x in ["ST", "退"]): continue
        item['名称'] = name
        final_results.append(item)

    if not final_results:
        print("❌ 全市场今日无符合条件的信号。")
        update_readme(pd.DataFrame(), pd.DataFrame())
        return

    # 4. 数据分类与排序
    df_res = pd.DataFrame(final_results)
    official = df_res[df_res['类型'] == "正式信号"].sort_values(by='额(万)', ascending=False)
    observer = df_res[df_res['类型'] == "观察储备"].sort_values(by='额(万)', ascending=False).head(15)
    
    # 5. 更新 README.md
    update_readme(official, observer)
    
    # 6. 保存归档 CSV (保留原有功能)
    now = datetime.now()
    save_filename = f"Daily_Sniper_V6_{now.strftime('%Y%m%d')}.csv"
    df_res.to_csv(save_filename, index=False, encoding='utf-8-sig')
    
    # 7. 终端输出预览
    print(f"\n✅ 扫描完成! \n- 正式信号: {len(official)} 个\n- 观察名单: {len(observer)} 个")
    print(f"- 归档文件: {save_filename}")

if __name__ == "__main__":
    main()
