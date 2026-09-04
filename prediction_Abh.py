
from pymol import cmd
import ablang
import numpy as np
import re
from Bio import pairwise2
import shutil
from model import GraphTrans
from data import ProteinGraphDataset
from torch_geometric.loader import DataLoader
import csv
import os
from igfold import IgFoldRunner
from igfold.refine.pyrosetta_ref import init_pyrosetta
import warnings
from abnumber import Chain
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from igfold.utils.pdb import clean_pdb
import pickle
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import argparse

def is_heavy(seq):
    chain = Chain(seq, scheme='imgt')

    return chain.is_heavy_chain()


def rechain_pdb(pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure("_", pdb_file)

    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme='imgt')
        chain_id = "H" if abnum_chain.is_heavy_chain() else "L"
        try:
            chain.id = chain_id
        except ValueError:
            chain.id = chain_id + "_"
    for chain in structure.get_chains():
        if "_" in chain.id:
            chain.id = chain.id.replace("_", "")

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)


def renumber_pdb(
    in_pdb_file,
    out_pdb_file=None,
    scheme="imgt",
):
    """
    Renumber the pdb file.
    """
    if out_pdb_file is None:
        out_pdb_file = in_pdb_file

    clean_pdb(in_pdb_file)

    parser = PDBParser()
    with warnings.catch_warnings(record=True):
        structure = parser.get_structure(
            "_",
            in_pdb_file,
        )

    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
        abnum_chain = Chain(seq, scheme=scheme)
        numbering = abnum_chain.positions.items()

        chain_res = list(chain.get_residues())
        assert len(chain_res) == len(numbering)

        for pdb_r, (pos, aa) in zip(chain_res, numbering):
            if aa != seq1(pdb_r.get_resname()):
                raise Exception(f"Failed to renumber PDB file {in_pdb_file}")
            pos = str(pos)[1:]
            if not pos[-1].isnumeric():
                ins = pos[-1]
                pos = int(pos[:-1])
            else:
                pos = int(pos)
                ins = ' '

            pdb_r._id = (' ', pos, ins)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb_file)
def read_fasta(filename):
    sequences = {}
    current_sequence_id = None
    current_sequence = ''

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # 处理序列标识行
                if current_sequence_id is not None:
                    sequences[current_sequence_id] = current_sequence
                current_sequence_id = line[1:]
                current_sequence = ''
            else:
                # 处理序列行
                current_sequence += line

        # 处理最后一个序列
        if current_sequence_id is not None:
            sequences[current_sequence_id] = current_sequence

    return sequences

aa_dict = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y", "UNK": "X"
}
def get_seq(pdb_name, aa_dict):
    ag_seq = ""
    cmd.delete('all')
    cmd.load(pdb_name)
    chains_list = cmd.get_chains()
    chain_sequences = {}
    for chain in chains_list:
        cmd.select("selected_chain", f"chain {chain}")
        resn_list = []
        resi_list = []
        res_dict = {}
        seen_resi = set()
        cmd.iterate("name ca and selected_chain", "resn_list.append(resn)", space=locals())
        cmd.iterate("name ca and selected_chain", "resi_list.append(resi)", space=locals())
        for resi, resn in zip(resi_list, resn_list):
            if resi in seen_resi:
                continue

            res_dict[resi] = aa_dict.get(resn, 'X')
            seen_resi.add(resi)
        seq = ''.join(res_dict.values())
        if chain in chains_list:
            ag_seq += seq
    return ag_seq

def read_ab_sequences(pdb_path, aa_dict):
    ab_seq_all = {}

    ablist = [filename for filename in os.listdir(pdb_path) if filename.endswith('_Ab.pdb')]

    for filename in ablist:
        ab_seq = {}
        ab_filepath = os.path.join(pdb_path, filename)

        with open(ab_filepath, 'r') as ab_file:

            hchain, lchain = 'H', 'L'

            # for line in ab_file:
            #     if line.startswith("REMARK   5 PAIRED_HL"):
            #         parts = line.strip().split()
            #         if len(parts) == 7 and ('PROTEIN' in parts[6] or 'PEPTIDE' in parts[6]):
            #             hchain, lchain = parts[3].split('=')[1], parts[4].split('=')[1]
        cmd.delete('all')
        cmd.load(ab_filepath)
        cmd.select("selected_chains", f"chain {hchain}")

        H_resn_list = []
        H_resi_list = []
        H_res_dict = {}
        H_seen_resi = set()
        H_residues_list = []
        cmd.iterate("name ca and selected_chains", "H_resn_list.append((resn))", space=locals())
        cmd.iterate("name ca and selected_chains", "H_resi_list.append((resi))", space=locals())
        for resi, resn in zip(H_resi_list, H_resn_list):
            if resi in H_seen_resi:
                continue
            H_res_dict[resi] = aa_dict.get(resn, 'X')
            H_seen_resi.add(resi)
            H_residues_list = [H_res_dict[resi] for resi in H_res_dict]
        H_seq = ''.join(H_res_dict.values())
        ab_seq['H'] = H_seq

        L_resn_list = []
        L_resi_list = []
        L_res_dict = {}
        L_seen_resi = set()
        L_residues_list = []
        cmd.iterate("name ca and selected_chains", "L_resn_list.append((resn))", space=locals())
        cmd.iterate("name ca and selected_chains", "L_resi_list.append((resi))", space=locals())
        for resi, resn in zip(L_resi_list, L_resn_list):
            if resi in L_seen_resi:
                continue
            L_res_dict[resi] = aa_dict.get(resn, 'X')
            L_seen_resi.add(resi)
            L_residues_list = [L_res_dict[resi] for resi in L_res_dict]
        L_seq = ''.join(L_res_dict.values())
        ab_seq['L'] = L_seq
        ab_seq_all[filename.split('.')[0]] = ab_seq
    return ab_seq_all

def load_ablang_model(chain, model_folder):
    # model_folder_path = os.path.join(os.path.dirname(__file__), model_folder)
    ablang_model = ablang.pretrained(chain, model_folder)
    ablang_model.freeze()
    return ablang_model

def process_sequences(ablang_model, seqs):
    tokens = ablang_model.tokenizer(seqs, pad=True)
    rescodings = ablang_model.AbRep(tokens)
    return rescodings

def exact_pssm(ID,ab_raw_pssm):
    row_pattern = re.compile("\s*\d{1,4}\s[A-Z]")
    number_pattern = re.compile('[-]*\d[.]*\d*')
    if ID.endswith('Ab_H'):
        last_underscore_index = ID.rfind('_')
        modified_string = ID[:last_underscore_index] + ID[last_underscore_index + 1:]
        H_pssm_path = ab_raw_pssm + "/" + modified_string + ".pssm"
        H_pssm_matrix_para = []

        with open(H_pssm_path, 'r') as h:
            for line in h:
                match = re.match(row_pattern, line)
                if match:
                    header = match.group(0)
                    tmp = line[len(header):]
                    nums = re.findall(number_pattern, tmp)
                    pssm_20 = list(map(int, nums[:20]))
                    H_pssm_matrix_para.append(pssm_20)
        H_pssm_matrix_para = np.array(H_pssm_matrix_para)
        np.save(ab_raw_pssm + "/" + ID + '.npy',np.array(H_pssm_matrix_para))


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ACC = float(lines[i][34:38].strip())
        ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
        dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))
    return seq, dssp_feature

def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    padded_item = np.zeros(9)

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])
    return matched_dssp


def get_dssp(ID, ref_seq):
    if ID.endswith('_Ab_H'):
        dssp_list_paratope = []
        os.system(f"./mkdssp -i {ab_folder}/pdb/{ID}.pdb -o {DSSP_path}/{ID}.dssp")
        dssp_seq_paratope, dssp_matrix_paratope = process_dssp(f"{DSSP_path}/{ID}.dssp")
        if dssp_seq_paratope != ref_seq:
            dssp_matrix_paratope = match_dssp(dssp_seq_paratope, dssp_matrix_paratope, ref_seq)
        np_dssp_matrix_paratope = np.array(dssp_matrix_paratope)

        torch.save(torch.tensor(np.array(np_dssp_matrix_paratope), dtype=torch.float32),
                   "{}/{}.tensor".format(DSSP_path, ID))
        os.system(f"rm {DSSP_path}/{ID}.dssp")

def get_pdb_xyz(pdb_file):
    current_pos = '-1000'
    X = []
    current_aa = {} # N, CA, C, O, R
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and line[22:27].strip() != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                # print(current_aa)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = line[22:27].strip()

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    if current_aa != {}:
        X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
    return np.array(X)

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def copy_files(src_folder, dest_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有文件
    for file_name in os.listdir(src_folder):
        # 构造源文件的完整路径
        src_file = os.path.join(src_folder, file_name)
        # 构造目标文件的完整路径
        dest_file = os.path.join(dest_folder, file_name)
        # 复制文件
        shutil.copy2(src_file, dest_file)

class Args:
    def __init__(self, filename):
        self.dataset_path = f'./tmp/{filename}/Dataset/'
        self.feature_path = f'./tmp/{filename}/Feature/'
        self.dataset = 'AAI'
        self.output_path = './weights/model/'



# filename = 'ab2'
parser = argparse.ArgumentParser(description="Process antibody files.")
parser.add_argument("filenames", metavar="F", type=str, nargs="+",
                    help="A list of filenames to process.")
args = parser.parse_args()

for filename in args.filenames:
    ab_fasta_path = f'./oas_fasta'
    init_pyrosetta()
    sequences = read_fasta(os.path.join(ab_fasta_path,filename+'.fasta'))
    os.makedirs(f'./antibody/{filename}/pdb/', exist_ok=True)
    pred_pdb = f'./antibody/{filename}/pdb/' + filename + '_Ab.pdb'

    igfold = IgFoldRunner()
    out = igfold.fold(
        pred_pdb,  # Output PDB file
        sequences=sequences,  # Antibody sequences
        do_refine=True,  # Refine the antibody structure with PyRosetta
        do_renum=False,  # Renumber predicted antibody structure (Chothia)
    )
    rechain_pdb(pred_pdb)
    renumber_pdb(pred_pdb)


    with open('./weights/VJ_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    data = pd.read_csv(f'./oas_VJ/{filename}_VJ.csv', sep=',')
    data['H_Species'] = data['H_Species'].str.replace(' ', '')
    # data['L_Species'] = data['L_Species'].str.replace(' ', '')
    data['species_H_Vgene'] = data['H_Species'] + '-' + data['H_Vgene']
    data['species_H_Jgene'] = data['H_Species'] + '-' + data['H_Jgene']
    # data['species_L_Vgene'] = data['L_Species'] + '-' + data['L_Vgene']
    # data['species_L_Jgene'] = data['L_Species'] + '-' + data['L_Jgene']

    ids = data['name'].drop_duplicates().values.tolist()
    H_Vgene = None
    H_Jgene = None
    L_Vgene = None
    L_Jgene = None
    # Iterate through identifiers
    for id in ids:
        # Select rows based on identifier
        selected_row = data[data['name'] == id]
        H_Vgene = selected_row['species_H_Vgene'].values[0]
        H_Jgene = selected_row['species_H_Jgene'].values[0]
        # L_Vgene = selected_row['species_L_Vgene'].values[0]
        # L_Jgene = selected_row['species_L_Jgene'].values[0]

        # Tokenize and pad gene sequences
        tokenized_sequence = tokenizer.texts_to_sequences([H_Vgene, H_Jgene])
        max_sequence_length = 5
        padded_sequence = pad_sequence([torch.tensor(seq + [0] * (max_sequence_length - len(seq))) for seq in tokenized_sequence],
                                    batch_first=True, padding_value=0)
        if padded_sequence.view(1,-1).shape != torch.Size([1, 10]):
            print(id)
            print(padded_sequence.view(1,-1).shape)
        os.makedirs(f'./antibody/{filename}/VJsequence/',exist_ok=True)

        torch.save(padded_sequence.view(1,-1), f'./antibody/{filename}/VJsequence/' + f'{filename}_Ab_H.tensor')

    ab_folder = f"./antibody/{filename}/"
    LLM_path = os.path.join(ab_folder, 'LLM')
    os.makedirs(LLM_path, exist_ok=True)
    ab_raw_pssm = f'./raw_pssm'
    PSSM_path = os.path.join(ab_folder, 'PSSM')
    os.makedirs(PSSM_path, exist_ok=True)
    DSSP_path = os.path.join(ab_folder, 'DSSP')
    os.makedirs(DSSP_path, exist_ok=True)
    XYZ_path = os.path.join(ab_folder, 'XYZ')
    os.makedirs(XYZ_path, exist_ok=True)

    args = Args(filename)

    cmd.load(pred_pdb)
    cmd.select("selected_residues", "chain H and resi 1-128")
    abname = filename+"_Ab"
    paratopename = abname.replace('_Ab','_Ab_H') + '.pdb'
    cmd.save(ab_folder + 'pdb/' + paratopename, "selected_residues")
    with open(ab_folder  +'pdb/' + paratopename, 'r') as temp_output_file:
        paratope_content = temp_output_file.readlines()
    paratope = [line for line in paratope_content if line.startswith("ATOM")]
    with open(ab_folder +'pdb/' + paratopename, 'w') as temp_output_file:
        temp_output_file.writelines(paratope)
    cmd.delete('all')

    ab_seq_all = read_ab_sequences(ab_folder +'/pdb/', aa_dict)


    heavy_ablang = load_ablang_model("heavy", "./weights/ablang_weight/model-weight-heavy")
    # light_ablang = load_ablang_model("light", "./weights/ablang_weight/model-weight-light")

    for antibody_id, sequences in ab_seq_all.items():
        h_sequence = [sequences['H']]
        heavy_rescodings = process_sequences(heavy_ablang, h_sequence)
        h_tensor = heavy_rescodings.last_hidden_states.detach()
        h_tensor = torch.squeeze(h_tensor,dim=0)
        h_tensor = h_tensor[1:-1]

        torch.save(h_tensor, os.path.join(LLM_path, antibody_id + "_H.tensor"))


    ab_ID = [filename.split('.')[0] for filename in os.listdir(ab_folder+'pdb/') if filename.endswith('_Ab_H.pdb')]
    for ID in ab_ID:
        if ID.endswith('Ab_H'):
            name_ab_pssm = ID
            exact_pssm(name_ab_pssm,ab_raw_pssm)

    for ID in ab_ID:
        raw_pssm = np.load(os.path.join(ab_raw_pssm, ID + ".npy"))
        Max_pssm = np.load('./weights/PSSM_repr/AAIMax_PSSM_repr.npy')
        Min_pssm = np.load('./weights/PSSM_repr/AAIMin_PSSM_repr.npy')

        pssm = 2 * (raw_pssm - Min_pssm) / (Max_pssm - Min_pssm) -1
        torch.save(torch.tensor(pssm, dtype=torch.float32), os.path.join(PSSM_path ,ID + ".tensor"))

    for ID in ab_ID:
        if ID.endswith('Ab_H'):
            name_ab = ID
            ref_seq = get_seq(f'./antibody/{filename}/pdb/' + name_ab + ".pdb",aa_dict)
            get_dssp(name_ab, ref_seq)

    for ID in ab_ID:
        print(ab_folder + "pdb/" + ID + ".pdb")
        with open(ab_folder +"pdb/"+ ID + ".pdb", "r") as f:
            X = get_pdb_xyz(f.readlines())  # [L, 4, 3]
            torch.save(torch.tensor(X, dtype=torch.float32), os.path.join(XYZ_path, ID + '.tensor'))

    copy_files(f'./antibody/{filename}/pdb', f"./tmp/{filename}/Dataset/AAI/pdb")
    copy_files(f'./antibody/{filename}/PSSM', f"./tmp/{filename}/Feature/PSSM/AAI")
    copy_files(f'./antibody/{filename}/DSSP' , f"./tmp/{filename}/Feature/DSSP/AAI")
    copy_files(f'./antibody/{filename}/LLM' , f"./tmp/{filename}/Feature/LLM/AAI")
    copy_files(f'./antibody/{filename}/XYZ' , f"./tmp/{filename}/Dataset/AAI/pdb")
    copy_files(f'./antibody/{filename}/VJsequence' , f"./tmp/{filename}/Feature/VJsequence/AAI")
    epilib_pdb_path = './Epitope_lib/Epitopelib_pdb'
    epilib_PSSM_path = './Epitope_lib/Epitope_Feature/epitopelib_PSSM'
    epilib_DSSP_path = './Epitope_lib/Epitope_Feature/epitopelib_DSSP'
    epilib_LLM_path = './Epitope_lib/Epitope_Feature/epitopelib_LLM'
    copy_files(epilib_pdb_path, f"./tmp/{filename}/Dataset/AAI/pdb")
    copy_files(epilib_PSSM_path, f"./tmp/{filename}/Feature/PSSM/AAI")
    copy_files(epilib_DSSP_path , f"./tmp/{filename}/Feature/DSSP/AAI")
    copy_files(epilib_LLM_path , f"./tmp/{filename}/Feature/LLM/AAI")

    action_path = f'./tmp/{filename}/Dataset/AAI/actions'
    os.makedirs(action_path, exist_ok=True)
    epilib_list = [filename.split(".")[0] for filename in os.listdir(epilib_pdb_path) if filename.endswith('epitope.pdb')]
    test_data = []
    for epi_name in epilib_list:
        test_data.append([epi_name, filename+'_Ab_H'])
    with open(os.path.join(action_path,"filtered_test_cmap.actions.tsv"), 'w', newline='') as test_file:
        tsv_writer = csv.writer(test_file, delimiter=' ')
        for row in test_data:
            tsv_writer.writerow(row)

    directory = f'./tmp/{filename}/Dataset/AAI/dictionary'
    os.makedirs(directory, exist_ok=True)
    fname = [filename.split(".")[0] for filename in os.listdir(f"./tmp/{filename}/Dataset/AAI/pdb") if filename.endswith('.pdb')]
    seq_dict = {}
    for name in fname:
        pdbname = os.path.join(f"./tmp/{filename}/Dataset/AAI/pdb", name + ".pdb")
        seq = get_seq(pdbname,aa_dict)
        seq_dict[name] = seq
        with open(os.path.join(directory, 'updated.protein.dictionary.tsv'), 'a') as out_file2:
            tsv_writer = csv.writer(out_file2, delimiter=' ')
            tsv_writer.writerow([name, seq])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_input_dim_0 = 1024 + 6 + 9 + 20 # ProtTrans + 二面角 + dssp + pssm
    node_input_dim_1 = 768 + 6 + 9 + 20 # ProtTrans + 二面角 + dssp + pssm
    edge_input_dim= 32 + 7
    hidden_dim=160
    num_layers=6
    dropout=0.3
    lr=0.001
    obj_max=1  # optimization object: max is better
    epochs= 100
    patience=50
    batch_size=16
    num_samples_multiplier=3
    folds=5
    seed=42
    vocab_size=155
    v_embed_dim=8
    j_embed_dim=8
    vj_dropout_rate=0.2
    num_heads = 8
    num_workers = 8


    output_path = './weights/model/'
    models = []
    for fold in range(folds):
        state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, device)
        model = GraphTrans(node_input_dim_0,node_input_dim_1, edge_input_dim, hidden_dim, num_layers, dropout, vocab_size, v_embed_dim,
                    j_embed_dim, vj_dropout_rate, num_heads).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    print('model count:', len(models))

    dataset_path = f'./tmp/{filename}/Dataset'
    dataset = 'AAI'

    seq_dict = {}
    dict_path = os.path.join(directory, 'updated.protein.dictionary.tsv')
    for fname in os.listdir(directory):
        if(fname.split('.')[0] == "updated" and fname.split('.')[-1] == "tsv" and fname.split('.')[-2] == "dictionary"):
            dict_file = os.path.join(directory, fname)  # Corrected this line
            with open(dict_file) as f:  # Indented this block properly
                for line in f:
                    if len(line) < 10:
                        continue
                    assert(len(line.split()) == 2)
                    seq_dict[line.split()[0]] = line.split()[1]

    test_data = []
    with open(os.path.join(dataset_path, dataset, "actions/filtered_test_cmap.actions.tsv")) as f:
        for line in f:
            if len(line) < 10:
                continue
            assert(len(line.split()) == 2)
            test_data.append(line.split())

    test_dataset = ProteinGraphDataset(seq_dict, test_data, range(len(test_data)), args)
    test_dataloader = DataLoader(test_dataset, batch_size = 16, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    test_pred_dict = {} # 导出测试结果
    test_pred = []
    test_y = []
    for data in test_dataloader:
        # print("Shape of node_feat:", data[0].node_feat.shape)
        # print("Shape of node_feat:", data[1].node_feat.shape)
        data = [_data.to(device) for _data in data]
        with torch.no_grad():
            outputs = [model(data).sigmoid() for model in models]
            outputs = torch.stack(outputs, 0).mean(0)  # 5个模型预测结果求平均
        # batch_test_y = data[0].y
        batch_test_pred = outputs
        # test_y += list(batch_test_y.detach().cpu().numpy())
        test_pred += list(batch_test_pred.detach().cpu().numpy())
        IDs = [a + ' ' + b for a, b in zip(data[0].name, data[1].name)]
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = outputs[i][0]

    test_pred_dict1 = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_pred_dict.items()}
    pred_results = pd.DataFrame(list(test_pred_dict1.items()), columns=['ID', 'Prediction'])
    pred_results[['epitope_ID', 'antibody_ID']] = pred_results['ID'].str.split(expand=True)
    pred_results.drop(columns=['ID'], inplace=True)
    pred_results = pred_results[['epitope_ID', 'antibody_ID', 'Prediction']]
    df = pred_results.sort_values(by='Prediction', ascending=False)
    epitope_lib = pd.read_csv(os.path.join('./Epitope_lib/epitope_new.csv'))
    merged_df = pd.merge(df, epitope_lib[['epitope', 'Epitope - Source Organism','Epitope - Molecule Parent']], left_on='epitope_ID', right_on='epitope')
    merged_df.drop(columns=['epitope'], inplace=True)
    merged_df = merged_df[['epitope_ID', 'antibody_ID', 'Prediction', 'Epitope - Source Organism','Epitope - Molecule Parent']]
    os.makedirs('./output', exist_ok=True)
    merged_df.to_csv(os.path.join('./output', f'{filename}_pred.csv'), sep=',')

