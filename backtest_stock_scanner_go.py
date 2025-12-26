import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count

# ==================== 2025“防假突破”回测参数 (根据您的要求代入) ===================
MIN_PRICE = 5.0              
MAX_AVG_TURNOVER_30 = 2.0    # 换手率更低，只要筹码锁定的票

# --- 选股逻辑优化：避开僵尸股，转向温和放量确认 ---
MIN_VOLUME_RATIO = 0.5       
MAX_VOLUME_RATIO = 1.2       # 0.5-1.2是最健康的止跌放量区间

# --- 极度超跌 + 乖离过滤 ---
RSI6_MAX = 28                
KDJ_K_MAX = 25               
MIN_PROFIT_POTENTIAL = 18    

# --- 核心：防假突破确认信号 ---
STAND_STILL_THRESHOLD = 1.005 # 必须站上5日线0.5%
MIN_BIAS_20 = -18            # 乖离率下限
MAX_BIAS_20 = -8             # 乖离率上限
MAX_TODAY_CHANGE = 4.0       

# 交易规则
STOP_LOSS = -5.0             # 5%止损
TRAILING_START = 10.0        # 10%移动止盈触发
HOLD_PERIODS = [3, 5, 10, 20] # 侧重短中期表现观察
# =====================================================================

def calculate_indicators(df):
    df = df.reset_index(drop=True)
    close = df['收盘']
    vol = df['成交量']
    
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['rsi6'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    # KDJ (9,3,3)
    low_list = df['最低'].rolling(window=9).min()
    high_list = df['最高'].rolling(window=9).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2).mean()
    
    # MA / BIAS / Turnover
    df['ma5'] = close.rolling(5).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    df['bias20'] = (close - df['ma20']) / df['ma20'] * 100
    df['avg_turnover_30'] = df['换手率'].rolling(30).mean()
    
    # Volume
    df['vol_ma5'] = vol.shift(1).rolling(5).mean()
    df['vol_ratio'] = vol / df['vol_ma5']
    df['vol_increase'] = vol > vol.shift(1) # 成交量需大于昨日确认有买盘
    
    return df

def simulate_trade(df, start_idx, max_days):
    buy_price = df.iloc[start_idx]['收盘']
    max_price = buy_price
    
    for day in range(1, max_days + 1):
        if start_idx + day >= len(df): break
        curr_row = df.iloc[start_idx + day]
        max_price = max(max_price, curr_row['最高'])
        
        # 1. 触发止损
        if (curr_row['最低'] - buy_price) / buy_price * 100 <= STOP_LOSS:
            return STOP_LOSS
            
        # 2. 移动止盈逻辑
        profit = (max_price - buy_price) / buy_price * 100
        if profit >= TRAILING_START:
            drawback = (max_price - curr_row['收盘']) / (max_price - buy_price)
            if drawback >= 0.3: # 回撤30%保护
                return max((curr_row['收盘'] - buy_price) / buy_price * 100, 2.0)

    # 3. 到期卖出
    end_idx = min(start_idx + max_days, len(df) - 1)
    return (df.iloc[end_idx]['收盘'] - buy_price) / buy_price * 100

def process_file(f):
    stock_code = os.path.basename(f)[:6]
    try:
        df = pd.read_csv(f)
        if len(df) < 100: return []
        df = calculate_indicators(df)
        
        results = []
        for i in range(60, len(df) - 20):
            row = df.iloc[i]
            potential = (row['ma60'] - row['收盘']) / row['收盘'] * 100
            change = (row['收盘'] - df.iloc[i-1]['收盘']) / df.iloc[i-1]['收盘'] * 100
            
            # --- 代入精选参数逻辑 ---
            if (row['收盘'] >= MIN_PRICE and
                row['avg_turnover_30'] <= MAX_AVG_TURNOVER_30 and
                row['rsi6'] <= RSI6_MAX and
                row['kdj_k'] <= KDJ_K_MAX and
                MIN_BIAS_20 <= row['bias20'] <= MAX_BIAS_20 and
                row['收盘'] >= row['ma5'] * STAND_STILL_THRESHOLD and # 站稳
                row['vol_increase'] and                               # 量增
                MIN_VOLUME_RATIO <= row['vol_ratio'] <= MAX_VOLUME_RATIO and
                potential >= MIN_PROFIT_POTENTIAL and
                change <= MAX_TODAY_CHANGE):
                
                trade = {'代码': stock_code, '日期': row['日期']}
                for p in HOLD_PERIODS:
                    trade[f'{p}日收益'] = simulate_trade(df, i, p)
                results.append(trade)
        return results
    except:
        return []

def main():
    print(f"🚀 启动“防假突破”高级回测 (参数: BIAS[{MIN_BIAS_20},{MAX_BIAS_20}], 量比[{MIN_VOLUME_RATIO},{MAX_VOLUME_RATIO}])")
    files = glob.glob(os.path.join('stock_data', '*.csv'))
    
    with Pool(cpu_count()) as p:
        raw = p.map(process_file, files)
        
    all_trades = [t for sub in raw for t in sub]
    if not all_trades:
        print("❌ 未匹配到任何信号")
        return

    df_res = pd.DataFrame(all_trades)
    print("\n" + "="*40 + "\n🎯 优化后策略看板")
    summary = []
    for p in HOLD_PERIODS:
        col = f'{p}日收益'
        summary.append({
            '周期': f'{p}天',
            '胜率': f'{(df_res[col]>0).sum()/len(df_res)*100:.2f}%',
            '平均收益': f'{df_res[col].mean():.2f}%',
            '信号数': len(df_res)
        })
    print(pd.DataFrame(summary).to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df_res.to_csv('results/backtest_advanced_summary.csv', index=False, encoding='utf_8_sig')
    print(f"\n✅ 报告已导出至 results/ (总信号: {len(df_res)})")

if __name__ == "__main__":
    main()
