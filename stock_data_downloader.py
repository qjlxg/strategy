import os
import pandas as pd
import akshare as ak
import time
from datetime import datetime
import sys

DATA_DIR = "stock_data"
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "checkpoint.txt") 

COLUMN_MAPPING = {
    "日期": "日期", "开盘": "开盘", "收盘": "收盘", "最高": "最高",
    "最低": "最低", "成交量": "成交量", "成交额": "成交额",
    "振幅": "振幅", "涨跌幅": "涨跌幅", "涨跌额": "涨跌额", "换手率": "换手率"
}
TARGET_COLUMNS = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

def download_item(symbol_short):
    """功能不变：下载并处理数据，仅在出错时报错"""
    file_path = os.path.join(DATA_DIR, f"{symbol_short}.csv")
    try:
        existing_dates = set()
        start_date = "19900101"
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                existing_dates = set(existing_df['日期'].astype(str).tolist())
                # 获取最后一天日期用于增量
                start_date = str(existing_df.iloc[-1]['日期']).replace("-", "")

        # 调用 akshare
        df = ak.stock_zh_a_hist(symbol=symbol_short, period="daily", start_date=start_date, adjust="")
        
        if df is not None and not df.empty:
            # --- 原有数据处理逻辑开始 ---
            df = df.rename(columns=COLUMN_MAPPING)
            df['股票代码'] = symbol_short
            df['日期'] = df['日期'].astype(str)
            
            # 严格去重逻辑
            df = df[~df['日期'].isin(existing_dates)]
            
            if not df.empty:
                df['成交额'] = df['成交额'].round(1)
                for col in ['开盘', '收盘', '最高', '最低', '振幅', '涨跌幅', '涨跌额', '换手率']:
                    df[col] = df[col].astype(float).round(2)
                df['成交量'] = df['成交量'].astype(int)
                
                df = df[TARGET_COLUMNS]
                header = not os.path.exists(file_path)
                # 追加模式写入
                df.to_csv(file_path, mode='a', index=False, header=header, encoding='utf-8')
            # --- 原有数据处理逻辑结束 ---
        
        time.sleep(0.2) # 防封禁休眠
        return True
    except Exception as e:
        # 仅显示错误日志
        print(f"Error at {symbol_short}: {e}")
        return False

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    if not os.path.exists(FILTERED_LIST_PATH):
        print("错误: 找不到名单文件")
        sys.exit(1)

    df_list = pd.read_csv(FILTERED_LIST_PATH)
    symbols = df_list['代码'].astype(str).str.zfill(6).tolist()

    # 读取进度断点 (如果是新任务从0开始)
    start_index = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                start_index = int(f.read().strip())
        except:
            start_index = 0

    if start_index >= len(symbols):
        # 如果已经全部跑完，重置断点并退出
        if os.path.exists(CHECKPOINT_PATH): os.remove(CHECKPOINT_PATH)
        return

    # 按顺序执行，确保断点准确
    for i in range(start_index, len(symbols)):
        if download_item(symbols[i]):
            # 成功则记录断点
            with open(CHECKPOINT_PATH, 'w') as f:
                f.write(str(i + 1))
        else:
            # 失败则报错退出，让 GitHub Actions 接管重试逻辑
            sys.exit(1)

    # 顺利完成后清除断点文件
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
