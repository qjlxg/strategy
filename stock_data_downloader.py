import os
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

DATA_DIR = "stock_data"
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")
DEBUG_LIMIT = 100 # 调试完成后请改为 None

# 严格按照 603348.csv 的表头
TARGET_COLUMNS = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

def process_data_frame(df, symbol_plain):
    """加工数据，使其与 603348.csv 格式完全一致"""
    df = df.reset_index()
    
    # 1. 日期改为 YYYY-MM-DD 格式
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # 2. 字段重命名
    df = df.rename(columns={
        'Date': '日期', 'Open': '开盘', 'Close': '收盘', 
        'High': '最高', 'Low': '最低', 'Volume': '成交量'
    })
    
    # 3. 基础字段
    df['股票代码'] = symbol_plain
    
    # 4. 计算指标并严格四舍五入
    # 成交额使用 float 并保留 1 位小数 (对应 603348.csv 风格)
    df['成交额'] = (df['成交量'] * df['收盘']).round(1)
    
    # 涨跌计算
    df['涨跌额'] = df['收盘'].diff().round(2)
    df['涨跌幅'] = (df['收盘'].pct_change() * 100).round(2)
    
    # 振幅计算
    prev_close = df['收盘'].shift(1)
    df['振幅'] = (((df['最高'] - df['最低']) / prev_close) * 100).round(2)
    
    # 换手率 (yf不提供，设为0.0)
    df['换手率'] = 0.0
    
    # 5. 修正数值精度，确保没有长尾小数
    num_cols = ['开盘', '收盘', '最高', '最低', '振幅', '涨跌幅', '涨跌额']
    for col in num_cols:
        df[col] = df[col].astype(float).round(2)
    
    # 成交量转为整数
    df['成交量'] = df['成交量'].astype(int)
    
    # 填充空值
    df.fillna(0, inplace=True)
    
    return df[TARGET_COLUMNS]

def download_item(symbol_yf):
    symbol_plain = symbol_yf.split('.')[0]
    file_path = os.path.join(DATA_DIR, f"{symbol_plain}.csv")
    
    try:
        ticker = yf.Ticker(symbol_yf)
        
        # 增量下载逻辑
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path) # 使用默认逗号读取
            if not existing_df.empty:
                last_date = pd.to_datetime(existing_df.iloc[-1]['日期']).date()
                if last_date >= datetime.now().date(): return True
                
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                new_data = ticker.history(start=start_date)
                if new_data is not None and not new_data.empty:
                    processed = process_data_frame(new_data, symbol_plain)
                    # 增量追加，不写表头
                    processed.to_csv(file_path, mode='a', index=False, header=False, encoding='utf-8')
                    print(f"增量更新: {symbol_plain}")
                return True

        # 全量下载逻辑
        data = ticker.history(period="max")
        if data is not None and not data.empty:
            processed = process_data_frame(data, symbol_plain)
            # 全量写入 (标准逗号分隔)
            processed.to_csv(file_path, index=False, encoding='utf-8')
            print(f"全量下载: {symbol_plain}")
        return True
    except Exception as e:
        print(f"跳过 {symbol_plain}: {e}")
        return False

def main():
    if not os.path.exists(FILTERED_LIST_PATH):
        print("错误: 找不到名单")
        return

    df_list = pd.read_csv(FILTERED_LIST_PATH)
    symbols = df_list['yf_code'].tolist()
    
    if DEBUG_LIMIT:
        print(f"调试模式：处理前 {DEBUG_LIMIT} 只")
        symbols = symbols[:DEBUG_LIMIT]

    # 并行下载
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_item, symbols)
    print("所有任务执行完毕。")

if __name__ == "__main__":
    main()
