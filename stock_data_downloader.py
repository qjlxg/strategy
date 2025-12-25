import os
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

DATA_DIR = "stock_data"
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")

# --- 调试开关 ---
DEBUG_LIMIT = 100  # 调试完成后请改为 None 或一个巨大的数字

def download_item(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        ticker = yf.Ticker(symbol)
        
        # 增量更新逻辑
        if os.path.exists(file_path):
            # 读取已有数据的最后日期
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                # 假设第一列是 Date
                last_date_str = existing_df.iloc[-1]['Date']
                # 解析日期并加 1 天作为起始点
                last_date = pd.to_datetime(last_date_str).date()
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                
                # 如果最后日期就是今天，则跳过
                if last_date >= datetime.now().date():
                    return True
                
                # 下载从最后日期之后的数据
                new_data = ticker.history(start=start_date)
                if not new_data.empty:
                    # 将 Index (Date) 转换为列以匹配 CSV 格式
                    new_data.reset_index(inplace=True)
                    # 确保日期列格式一致
                    new_data['Date'] = new_data['Date'].dt.strftime('%Y-%m-%d')
                    # 追加保存 (不写表头)
                    new_data.to_csv(file_path, mode='a', index=False, header=False)
                    print(f"增量更新成功: {symbol}")
            else:
                # 文件为空则重新下载全量
                data = ticker.history(period="max")
                data.reset_index().to_csv(file_path, index=False)
        else:
            # 第一次下载，获取全量历史数据
            data = ticker.history(period="max")
            if not data.empty:
                data.reset_index(inplace=True)
                data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
                data.to_csv(file_path, index=False)
                print(f"全量下载成功: {symbol}")
        return True
    except Exception as e:
        print(f"处理 {symbol} 失败: {e}")
        return False

def main():
    if not os.path.exists(FILTERED_LIST_PATH):
        print(f"错误: 找不到名单文件 {FILTERED_LIST_PATH}")
        return

    df = pd.read_csv(FILTERED_LIST_PATH)
    symbols = df['yf_code'].tolist()
    
    # 调试限制
    if DEBUG_LIMIT:
        print(f"⚠️ 调试模式: 仅处理前 {DEBUG_LIMIT} 只股票")
        symbols = symbols[:DEBUG_LIMIT]

    print(f"开始执行下载/更新任务，目标数量: {len(symbols)}...")
    
    # 使用线程池并行下载
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_item, symbols)
    
    print("任务执行完毕。")

if __name__ == "__main__":
    main()
