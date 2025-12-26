import pandas as pd
from datetime import datetime
import os
import pytz
import glob
from multiprocessing import Pool, cpu_count
import numpy as np

# ==================== 2025“防假突破”极致精选参数 ===================
MIN_PRICE = 5.0              
MAX_AVG_TURNOVER_30 = 2.0    # 强化：只要筹码稳定的标的

# --- 选股逻辑优化：避开僵尸股，允许微幅放量确认 ---
MIN_VOLUME_RATIO = 0.5       # 避开完全无买盘的僵尸股
MAX_VOLUME_RATIO = 1.2       # 允许小幅放量确认止跌
RSI6_MAX = 28                # 严谨超跌区
KDJ_K_MAX = 25               # 底部磨底确认
MIN_PROFIT_POTENTIAL = 18    # 空间要求

# --- 核心：确认信号强度 ---
# 股价必须高于 5 日线 0.5%，且成交量必须大于昨天（量增价涨）
STAND_STILL_THRESHOLD = 1.005 
# 20日乖离率控制：锁定“弹簧压到极致”的区域
MIN_BIAS_20 = -18
MAX_BIAS_20 = -8

MAX_TODAY_CHANGE = 4.0       # 允许适当涨幅以确认站上均线，但拒绝大阳线追高
# =====================================================================

SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')
STOCK_DATA_DIR = 'stock_data'
NAME_MAP_FILE = 'stock_names.csv' 

def process_single_stock(args):
    file_path, name_map = args
    code = os.path.basename(file_path).split('.')[0]
    
    try:
        df = pd.read_csv(file_path)
        if len(df) < 65: return None
        
        # 计算指标
        close = df['收盘']
        vol = df['成交量']
        
        # RSI6
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rsi6 = 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1]))) if loss.iloc[-1] != 0 else 100
        
        # KDJ_K
        low_9 = df['最低'].rolling(9).min()
        high_9 = df['最高'].rolling(9).max()
        kdj_k = ((close - low_9) / (high_9 - low_9) * 100).ewm(com=2).mean().iloc[-1]
        
        # MA & BIAS
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        bias20 = (close.iloc[-1] - ma20) / ma20 * 100
        
        # 量能确认
        vol_ma5 = vol.shift(1).rolling(5).mean().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol_ma5
        vol_increase = vol.iloc[-1] > vol.iloc[-2] # 今天的量大于昨天
        
        # 潜在反弹空间
        potential = (ma60 - close.iloc[-1]) / close.iloc[-1] * 100
        change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        avg_turnover_30 = df['换手率'].rolling(30).mean().iloc[-1]

        # ================= 严格筛选条件 =================
        if (close.iloc[-1] >= MIN_PRICE and
            avg_turnover_30 <= MAX_AVG_TURNOVER_30 and
            rsi6 <= RSI6_MAX and
            kdj_k <= KDJ_K_MAX and
            MIN_BIAS_20 <= bias20 <= MAX_BIAS_20 and
            close.iloc[-1] >= ma5 * STAND_STILL_THRESHOLD and # 站稳确认
            vol_increase and                                  # 量增确认
            MIN_VOLUME_RATIO <= vol_ratio <= MAX_VOLUME_RATIO and
            potential >= MIN_PROFIT_POTENTIAL and
            change <= MAX_TODAY_CHANGE):

            return {
                '代码': code,
                '名称': name_map.get(code, "未知"),
                '现价': close.iloc[-1],
                '今日量比': round(vol_ratio, 2),
                'RSI6': round(rsi6, 1),
                '20日乖离': f\"{round(bias20, 1)}%\",
                '反弹空间': f\"{round(potential, 1)}%\",
                '今日涨跌': f\"{round(change, 1)}%\"
            }
    except:
        return None

def main():
    now_shanghai = datetime.now(SHANGHAI_TZ)
    print(f"🚀 极致缩量 + 防假突破扫描开始... 当前时间: {now_shanghai.strftime('%Y-%m-%d %H:%M')}")

    name_map = {}
    if os.path.exists(NAME_MAP_FILE):
        n_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
        name_map = dict(zip(n_df['code'].str.zfill(6), n_df['name']))

    file_list = glob.glob(os.path.join(STOCK_DATA_DIR, '*.csv'))
    tasks = [(file_path, name_map) for file_path in file_list]

    with Pool(processes=cpu_count()) as pool:
        raw_results = pool.map(process_single_stock, tasks)

    results = [r for r in raw_results if r is not None]
        
    if results:
        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values(by='今日量比', ascending=True)
        
        print(f"\n🎯 扫描完成，精选出 {len(results)} 只“站稳”标的：")
        print(df_result.to_string(index=False))
        
        # 存入结果
        os.makedirs('results', exist_ok=True)
        df_result.to_csv('results/selected_stocks.csv', index=False, encoding='utf_8_sig')
    else:
        print("\n🤔 市场暂未发现符合“防假突破”逻辑的极品信号。")

if __name__ == "__main__":
    main()
