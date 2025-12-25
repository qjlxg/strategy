import os
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

DATA_DIR = "stock_data"
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")

# --- 调试开关 ---
DEBUG_LIMIT = 100  # 调试完成后请改为 None

# 字段映射表
COLUMNS_MAP = {
    'Date': '日期',
    'Open': '开盘',
    'Close': '收盘',
    'High': '最高',
    'Low': '最低',
    'Volume': '成交量'
}

# 目标列顺序
TARGET_COLUMNS = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

def process_data_frame(df, symbol_plain):
    """将 yfinance 的原始数据转换为目标格式"""
    df = df.reset_index()
    # 转换日期格式为 YYYY/MM/DD
    df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')
    
    # 字段重命名
    df = df.rename(columns=COLUMNS_MAP)
    
    # 补充缺失字段
    df['股票代码'] = symbol_plain
    # yfinance 默认没有直接提供“成交额”，用 成交量 * 收盘价 估算（或从其他列获取，yf 主要是数据点）
    df['成交额'] = (df['成交量'] * df['收盘']).astype(int)
    
    # 计算涨跌额和涨跌幅 (基于前一收盘价)
    df['涨跌额'] = df['收盘'].diff()
    df['涨跌幅'] = (df['收盘'].pct_change() * 100).round(2)
    
    # 计算振幅: (最高-最低) / 昨收
    prev_close = df['收盘'].shift(1)
    df['振幅'] = (((df['最高'] - df['最低']) / prev_close) * 100).round(2)
    
    # 换手率：yfinance 网页端有，但 API 历史数据通常不带，设为 0 或 NaN
    df['换手率'] = 0.0
    
    # 填充第一行的 NaN
    df.fillna(0, inplace=True)
    
    return df[TARGET_COLUMNS]

def download_item(symbol_yf):
    # 提取纯数字代码，例如 000551.SZ -> 000551
    symbol_plain = symbol_yf.split('.')[0]
    file_path = os.path.join(DATA_DIR, f"{symbol_plain}.csv")
    
    try:
        ticker = yf.Ticker(symbol_yf)
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                last_date_str = str(existing_df.iloc[-1]['日期']).replace('/', '-')
                last_date = pd.to_datetime(last_date_str).date()
                
                if last_date >= datetime.now().date():
                    return True
                
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                new_data = ticker.history(start=start_date)
                
                if new_data is not None and not new_data.empty:
                    processed_new = process_data_frame(new_data, symbol_plain)
                    # 追加保存，不写表头
                    processed_new.to_csv(file_path, mode='a', index=False, header=False, encoding='utf-8-sig', sep='\t')
                    print(f"增量更新: {symbol_plain}")
            else:
                os.remove(file_path) # 空文件删除重下
        
        # 全量下载逻辑
        if not os.path.exists(file_path):
            data = ticker.history(period="max")
            if data is not None and not data.empty:
                processed_all = process_data_frame(data, symbol_plain)
                processed_all.to_csv(file_path, index=False, encoding='utf-8-sig', sep='\t')
                print(f"全量下载: {symbol_plain}")
                
        return True
    except Exception as e:
        print(f"跳过 {symbol_plain}: {str(e)[:50]}")
        return False

def main():
    if not os.path.exists(FILTERED_LIST_PATH):
        print(f"错误: 找不到名单文件 {FILTERED_LIST_PATH}")
        return

    df_list = pd.read_csv(FILTERED_LIST_PATH)
    # 排除北交所
    df_list = df_list[~df_list['代码'].astype(str).str.startswith(('8', '9', '4'))]
    
    symbols = df_list['yf_code'].tolist()
    
    if DEBUG_LIMIT:
        print(f"⚠️ 调试模式: 仅处理前 {DEBUG_LIMIT} 只股票")
        symbols = symbols[:DEBUG_LIMIT]

    print(f"开始同步数据，保存格式: 纯数字.csv，目标数量: {len(symbols)}...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_item, symbols)
    
    print("任务完成。")

if __name__ == "__main__":
    main()
