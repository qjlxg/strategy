import os
import pandas as pd
import akshare as ak
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

DATA_DIR = "stock_data"
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")
DEBUG_LIMIT = None # 调试完成后请改为 None

# 目标列名映射 (akshare 返回的列名 -> 你的目标列名)
COLUMN_MAPPING = {
    "日期": "日期",
    "开盘": "开盘",
    "收盘": "收盘",
    "最高": "最高",
    "最低": "最低",
    "成交量": "成交量",
    "成交额": "成交额",
    "振幅": "振幅",
    "涨跌幅": "涨跌幅",
    "涨跌额": "涨跌额",
    "换手率": "换手率"
}

TARGET_COLUMNS = ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

def download_item(symbol_short):
    """使用 akshare 下载历史数据"""
    file_path = os.path.join(DATA_DIR, f"{symbol_short}.csv")
    
    try:
        # 判断是增量还是全量
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                # 获取最后一天日期，格式为 YYYYMMDD
                last_date = str(existing_df.iloc[-1]['日期']).replace("-", "")
                if last_date >= datetime.now().strftime("%Y%m%d"):
                    return True
                # 增量下载从最后一天开始（akshare 会包含当天）
                df = ak.stock_zh_a_hist(symbol=symbol_short, period="daily", start_date=last_date, adjust="")
                if df is not None and len(df) > 1:
                    df = df.iloc[1:] # 去掉重复的第一行
                else:
                    return True
            else:
                df = ak.stock_zh_a_hist(symbol=symbol_short, period="daily", adjust="")
        else:
            # 全量下载
            df = ak.stock_zh_a_hist(symbol=symbol_short, period="daily", adjust="")

        if df is not None and not df.empty:
            # 数据加工
            df = df.rename(columns=COLUMN_MAPPING)
            df['股票代码'] = symbol_short
            
            # 格式化数值：成交额保留 1 位小数，其他保留 2 位
            df['成交额'] = df['成交额'].round(1)
            for col in ['开盘', '收盘', '最高', '最低', '振幅', '涨跌幅', '涨跌额', '换手率']:
                df[col] = df[col].astype(float).round(2)
            df['成交量'] = df['成交量'].astype(int)
            
            # 最终筛选列
            df = df[TARGET_COLUMNS]

            # 写入文件
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', index=False, header=False, encoding='utf-8')
                print(f"增量更新成功: {symbol_short}")
            else:
                df.to_csv(file_path, index=False, encoding='utf-8')
                print(f"全量下载成功: {symbol_short}")
        
        # 适当休眠，防止被接口封禁
        time.sleep(0.2)
        return True
    except Exception as e:
        print(f"下载失败 {symbol_short}: {e}")
        return False

def main():
    if not os.path.exists(FILTERED_LIST_PATH):
        print("错误: 找不到名单文件")
        return

    # 读取名单中的纯数字代码
    df_list = pd.read_csv(FILTERED_LIST_PATH)
    # 确保代码是 6 位字符串格式
    symbols = df_list['代码'].astype(str).str.zfill(6).tolist()
    
    if DEBUG_LIMIT:
        print(f"调试模式：处理前 {DEBUG_LIMIT} 只")
        symbols = symbols[:DEBUG_LIMIT]

    print(f"开始同步数据 (数据源: akshare)，目标数量: {len(symbols)}...")
    
    # 因为 akshare 访问频率限制，不建议线程开太大，建议 3-5 个
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(download_item, symbols)
    
    print("任务执行完成。")

if __name__ == "__main__":
    main()
