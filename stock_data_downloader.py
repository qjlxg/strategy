import os
import akshare as ak
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

# 创建保存目录
DATA_DIR = "stock_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

RAW_LIST_PATH = os.path.join(DATA_DIR, "raw_stock_list.csv")
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")

def get_and_filter_stocks():
    """获取实时数据、保存原始名单、执行过滤并保存清理后的名单"""
    print("正在从 akshare 获取 A 股实时名单...")
    # 获取全量 A 股实时行情数据
    df = ak.stock_zh_a_spot_em()
    
    # 保存原始名单
    df.to_csv(RAW_LIST_PATH, index=False, encoding='utf-8-sig')
    print(f"原始名单已保存至: {RAW_LIST_PATH}")

    # --- 执行过滤逻辑 ---
    # 1. 排除 ST (通过名称中是否包含 ST 判断)
    df = df[~df['名称'].str.contains("ST", na=False)]
    
    # 2. 排除 30 开头 (创业板)
    df = df[~df['代码'].str.startswith("30")]
    
    # 3. 价格过滤: 5.0 <= 最新价 <= 20.0
    # 注意：akshare 返回的列名通常包含 '最新价'
    df = df[(df['最新价'] >= 5.0) & (df['最新价'] <= 20.0)]
    
    # 4. 转换代码格式以适配 yfinance (6开头为.SS, 其他如00开头为.SZ)
    def format_code(c):
        return f"{c}.SS" if c.startswith('6') else f"{c}.SZ"
    
    df['yf_code'] = df['代码'].apply(format_code)
    
    # 保存清理后的精简名单
    df.to_csv(FILTERED_LIST_PATH, index=False, encoding='utf-8-sig')
    print(f"清理完成，符合条件股票共 {len(df)} 只。精简名单已保存。")
    return df['yf_code'].tolist()

def download_data(symbol):
    """并行下载的具体任务"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            data.to_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
            return True
    except:
        pass
    return False

def main():
    # 二次运行检查：如果精简名单已存在，直接读取
    if os.path.exists(FILTERED_LIST_PATH):
        print("检测到已存在的精简名单，直接加载进行下载...")
        df_filtered = pd.read_csv(FILTERED_LIST_PATH)
        symbols = df_filtered['yf_code'].tolist()
    else:
        # 第一次运行，执行获取和过滤
        symbols = get_and_filter_stocks()

    print(f"开始并行下载数据 (共 {len(symbols)} 只)...")
    # 使用 10 个线程并行下载
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_data, symbols)
    print("所有任务执行完毕。")

if __name__ == "__main__":
    main()
