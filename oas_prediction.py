#%%
import pandas as pd
oas_data = pd.read_csv('./OAStest_ab.csv')
namelist = oas_data['name'].tolist()
print(namelist)

for index, row in oas_data.iterrows():
    heavy = row['sequence_alignment_aa_heavy']
    light = row['sequence_alignment_aa_light']
    name = row['name']
    
    with open(f"./oas_fasta/{name}.fasta", "w") as file:
        file.write(f">H\n{heavy}\n>L\n{light}\n")

columns_to_keep = ["name", "v_call_heavy", "j_call_heavy", "v_call_light", "j_call_light"]
df_selected = oas_data[columns_to_keep].copy()
df_selected.columns = ["name", "H_Vgene", "H_Jgene", "L_Vgene", "L_Jgene"]

# 添加新的列
df_selected.loc[:, "H_Species"] = "homosapiens"
df_selected.loc[:, "L_Species"] = "homosapiens"

# 将每一行写入到单独的CSV文件中
for index, row in df_selected.iterrows():
    name = row["name"]
    row_df = row.to_frame().T
    row_df.to_csv(f"./oas_VJ/{name}_VJ.csv", index=False)


# %%
import pandas as pd

oas_data = pd.read_csv('./OAStest_ab.csv')

# 合并两个序列列并交替排列
sequences = []
for idx, row in oas_data.iterrows():
    sequences.append((f"{row['name']}_AbH", row['sequence_alignment_aa_heavy']))
    sequences.append((f"{row['name']}_AbL", row['sequence_alignment_aa_light']))

# 定义函数保存序列到FASTA文件
def save_to_fasta_file(sequences, start_idx, file_idx):
    file_name = f'sequences_part{file_idx}.fasta'
    with open(file_name, 'w') as f:
        for name, sequence in sequences[start_idx:start_idx + 500]:
            f.write(f">{name}\n{sequence}\n")
    print(f"Saved {file_name}")

# 每500条序列保存到一个FASTA文件
for i in range(0, len(sequences), 500):
    save_to_fasta_file(sequences, i, i // 500 + 1)


# %%
import os
import shutil

# 文件夹路径
folder_path = './pssm_caluate/22'

# 读取fasta文件中的序列名
fasta_file_path = './pssm_caluate/sequences_part22.fasta'
fasta_sequences = []
with open(fasta_file_path, 'r') as fasta_file:
    for line in fasta_file:
        if line.startswith('>'):
            fasta_sequences.append(line.strip()[1:])

# 获取pssm文件列表
pssm_files = [file for file in os.listdir(folder_path) if file.endswith('.pssm')]

# 遍历pssm文件，重命名为对应的fasta序列名
for pssm_file in pssm_files:
    # 获取序列编号
    index = int(pssm_file.split('_')[1].split('.')[0]) - 1
    # 获取对应的fasta序列名
    fasta_sequence_name = fasta_sequences[index].split('.')[0]
    # 构建新的文件名
    new_file_name = os.path.join(folder_path, f'{fasta_sequence_name}.pssm')
    # 重命名文件
    os.rename(os.path.join(folder_path, pssm_file), new_file_name)

print("文件重命名完成。")
# %%
