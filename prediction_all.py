#%%
import pandas as pd
import subprocess
import multiprocessing
oas_data = pd.read_csv('./OAStest_ab.csv')
name= oas_data['name'].tolist()
namelist = list(reversed(name))
def run_command(name):
    command = f"python prediction_Abh.py {name}"
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    if process.returncode != 0:
        print(f"Command failed for {name}")
    else:
        print(f"Command succeeded for {name}")

if __name__ == "__main__":
    # 创建一个进程池
    with multiprocessing.Pool(processes=1) as pool:
        # 并行运行命令
        pool.map(run_command, namelist)
