import os
import pandas as pd
import glob
from tqdm import tqdm

def merge_parquet_files(directory, output_file):
    """
    合并指定目录下的所有parquet文件到一个输出文件
    """
    print(f"Merging parquet files in {directory}...")
    
    # 获取目录下所有parquet文件
    parquet_files = glob.glob(os.path.join(directory, "*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {directory}")
        return False
        
    if len(parquet_files) == 1 and os.path.basename(parquet_files[0]) == os.path.basename(output_file):
        print(f"Only found {os.path.basename(output_file)}, no need to merge")
        return False
    
    # 读取所有parquet文件
    dfs = []
    for file in tqdm(parquet_files, desc=f"Reading {directory} files"):
        # 跳过目标输出文件（如果存在）
        if os.path.abspath(file) == os.path.abspath(output_file):
            continue
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if not dfs:
        print(f"No valid parquet files to merge in {directory}")
        return False
    
    # 合并数据框
    print(f"Merging {len(dfs)} parquet files...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 保存合并后的文件
    merged_df.to_parquet(output_file, index=False)
    print(f"Merged file saved to {output_file}")
    return True

def main():
    # 合并test目录下的parquet文件
    test_success = merge_parquet_files("test", "test/test.parquet")
    
    # 合并validation目录下的parquet文件
    validation_success = merge_parquet_files("validation", "validation/validation.parquet")
    
    if test_success or validation_success:
        print("Merge completed successfully!")
    else:
        print("No files were merged")

if __name__ == "__main__":
    main() 