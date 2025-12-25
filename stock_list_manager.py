import os
import akshare as ak
import pandas as pd

DATA_DIR = "stock_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

RAW_LIST_PATH = os.path.join(DATA_DIR, "raw_stock_list.csv")
FILTERED_LIST_PATH = os.path.join(DATA_DIR, "filtered_stock_list.csv")

def main():
    print("正在获取 A 股实时名单...")
    # 获取全量行情
    df = ak.stock_zh_a_spot_em()
    df.to_csv(RAW_LIST_PATH, index=False, encoding='utf-8-sig')
    
    # 过滤逻辑
    # 1. 排除 ST
    df = df[~df['名称'].str.contains("ST", na=False)]
    # 2. 排除 30 开头 (创业板)
    df = df[~df['代码'].str.startswith("30")]
    # 3. 价格过滤: 5.0 <= 最新价 <= 20.0
    df = df[(df['最新价'] >= 5.0) & (df['最新价'] <= 20.0)]
    
    # 转换代码格式适配 yfinance
    def format_code(c):
        return f"{c}.SS" if c.startswith('6') else f"{c}.SZ"
    
    df['yf_code'] = df['代码'].apply(format_code)
    
    # 保存精简名单
    df.to_csv(FILTERED_LIST_PATH, index=False, encoding='utf-8-sig')
    print(f"名单处理完成。原始股数: {len(pd.read_csv(RAW_LIST_PATH))}，精简后股数: {len(df)}")

if __name__ == "__main__":
    main()
