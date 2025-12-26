import pandas as pd
from datetime import datetime
import os
import pytz
import glob
from multiprocessing import Pool, cpu_count
import numpy as np

# ==================== 2025“防假突破”极致精选参数 ===================
MIN_PRICE = 5.0              
MAX_AVG_TURNOVER_30 = 2.0    # 换手率更低，只要筹码锁定的票

# --- 选股逻辑优化：避开僵尸股，转向温和放量确认 ---
MIN_VOLUME_RATIO = 0.5       # 避开量比过小的死票
MAX_VOLUME_RATIO = 1.2       # 0.5-1.2是最健康的止跌放量区间

# --- 极度超跌 + 乖离过滤 ---
RSI6_MAX = 28                
KDJ_K_MAX = 25               
MIN_PROFIT_POTENTIAL = 18    # 空间要求

# --- 核心：防假突破确认信号 ---
STAND_STILL_THRESHOLD = 1.005 # 必须站上5日线0.5%
MIN_BIAS_20 = -18            # 乖离率下限（防止加速赶底）
MAX_BIAS_20 = -8             # 乖离率上限（确保弹簧压得够紧）

MAX_TODAY_CHANGE = 4.0       # 允许适度涨幅以确认站稳
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
        
        close = df['收盘']
        vol = df['成交量']
        
        # 1. 计算 RSI6
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        last_gain = gain.iloc[-1]
        last_loss = loss.iloc[-1]
        rsi6 = 100 - (100 / (1 + (last_gain / last_loss))) if last_loss != 0 else 100
        
        # 2. 计算 KDJ_K
        low_9 = df['最低'].rolling(9).min()
        high_9 = df['最高'].rolling(9).max()
        kdj_k = ((close - low_9) / (high_9 - low_9) * 100).ewm(com=2).mean().iloc[-1]
        
        # 3. 计算 MA & BIAS
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        bias20 = (close.iloc[-1] - ma20) / ma20 * 100
        
        # 4. 量能确认
        vol_ma5 = vol.shift(1).rolling(5).mean().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol_ma5
        vol_increase = vol.iloc[-1] > vol.iloc[-2] # 今天的量大于昨天
        
        # 5. 辅助信息
        potential = (ma60 - close.iloc[-1]) / close.iloc[-1] * 100
        change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        avg_turnover_30 = df['换手率'].rolling(30).mean().iloc[-1]

        # ================= 严格筛选条件 =================
        if (close.iloc[-1] >= MIN_PRICE and
            avg_turnover_30 <= MAX_AVG_TURNOVER_30 and
            rsi6 <= RSI6_MAX and
            kdj_k <= KDJ_K_MAX and
            MIN_BIAS_20 <= bias20 <= MAX_BIAS_20 and
            close.iloc[-1] >= ma5 * STAND_STILL_THRESHOLD and 
            vol_increase and                                  
            MIN_VOLUME_RATIO <= vol_ratio <= MAX_VOLUME_RATIO and
            potential >= MIN_PROFIT_POTENTIAL and
            change <= MAX_TODAY_CHANGE):

            return {
                '代码': code,
                '名称': name_map.get(code, "未知"),
                '现价': close.iloc[-1],
                '今日量比': round(vol_ratio, 2),
                'RSI6': round(rsi6, 1),
                '20日乖离': f"{round(bias20, 1)}%",
                '反弹空间': f"{round(potential, 1)}%",
                '今日涨跌': f"{round(change, 1)}%"
            }
    except:
        return None

def main():
    now_shanghai = datetime.now(SHANGHAI_TZ)
    print(f"🚀 极致缩量精选扫描开始... 目标：防假突破高胜率低吸")

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
        # 排序：RSI越低代表超跌越重，潜力越大
        df_result = df_result.sort_values(by='RSI6', ascending=True)
        
        print(f"\n🎯 扫描完成，精选出 {len(results)} 只“带量站稳”标的：")
        print(df_result.to_string(index=False))
        
        os.makedirs('results', exist_ok=True)
        df_result.to_csv('results/selected_stocks.csv', index=False, encoding='utf_8_sig')
    else:
        print("\n🤔 市场暂未发现符合“防假突破”逻辑的信号。")

if __name__ == "__main__":
    main()
