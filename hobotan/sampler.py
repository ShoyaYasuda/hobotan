import sys
import time
import numpy as np
import numpy.random as nr


#共通後処理
"""
pool=(shots, N), score=(N, )
"""
def get_result(pool, score, index_map):
    #重複解を集計
    unique_pool, original_index, unique_counts = np.unique(pool, axis=0, return_index=True, return_counts=True)
    #print(unique_pool, original_index, unique_counts)
    
    #エネルギーもユニークに集計
    unique_energy = score[original_index]
    #print(unique_energy)
    
    #エネルギー低い順にソート
    order = np.argsort(unique_energy)
    unique_pool = unique_pool[order]
    unique_energy = unique_energy[order]
    unique_counts = unique_counts[order]
    
    #結果リスト
    result = [[dict(zip(index_map.keys(), unique_pool[i])), unique_energy[i], unique_counts[i]] for i in range(len(unique_pool))]
    #print(result)
    
    return result


#謎のglobal対応
score2 = 0

#アニーリング
class SASampler:
    def __init__(self, seed=None):
        #乱数シード
        self.seed = seed

    
    def run(self, hobomix, shots=100, T_num=2000, show=False):
        global score2
        
        #解除
        hobo, index_map = hobomix
        # print(index_map)
        
        #matrixサイズ
        N = len(hobo)
        # print(N)
        
        #次数
        ho = len(hobo.shape)
        # print(ho)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        # プール初期化
        pool_num = shots
        pool = nr.randint(0, 2, (pool_num, N)).astype(float)
        # print(pool)
        
        """
        poolの重複を解除する
        """
        # 重複は振り直し
        # パリエーションに余裕あれば確定非重複
        if pool_num < 2 ** (N - 1):
            # print('remake 1')
            for i in range(pool_num - 1):
                for j in range(i + 1, pool_num):
                    while (pool[i] == pool[j]).all():
                        pool[j] = nr.randint(0, 2, N)
        else:
            # パリエーションに余裕なければ3トライ重複可
            # print('remake 2')
            for i in range(pool_num - 1):
                for j in range(i + 1, pool_num):
                    count = 0
                    while (pool[i] == pool[j]).all():
                        pool[j] = nr.randint(0, 2, N)
                        count += 1
                        if count == 3:
                            break
        
        #スコア初期化
        score = np.zeros(pool_num)
        
        """
        #旧実装
        # einsum("ijk,i,j,k->", qmatrix, x, x, x)
        # einsum("ijk,Ni,Nj,Nk->N", qmatrix, pool, pool, pool)
        
        #スコア計算コマンド
        k = ',Na,Nb,Nc,Nd,Ne,Nf,Ng,Nh,Nj,Nk,Nl,Nm,Nn,No,Np,Nq,Nr,Ns,Nt,Nu,Nv,Nw,Nx,Ny,Nz'
        l = 'abcdefghjklmnopqrstuvwxyz'
        s = l[:ho] + k[:3*ho] + '->N'
        # print(s)
        command = 'global score2\r\n'
        command += f'score2 = np.zeros(pool_num)\r\n'
        command += f'score2 = np.einsum(\'{s}\', hobo, pool2' + ', pool2' * (ho - 1) + f')\r\n'
        # print(command)
        
        #スコア計算
        pool2 = pool
        exec(command)
        score = score2
        # print(score)
        """
        
        #スコア計算
        k = ',Na,Nb,Nc,Nd,Ne,Nf,Ng,Nh,Nj,Nk,Nl,Nm,Nn,No,Np,Nq,Nr,Ns,Nt,Nu,Nv,Nw,Nx,Ny,Nz'
        l = 'abcdefghjklmnopqrstuvwxyz'
        s = l[:ho] + k[:3*ho] + '->N'
        # print(s)
        
        operands = [hobo] + [pool] * ho
        score = np.einsum(s, *operands)
        # print(score)
        
        # フリップ数リスト（2個まで下がる）
        flip = np.sort(nr.rand(T_num) ** 2)[::-1]
        flip = (flip * max(0, N * 0.5 - 2)).astype(int) + 2
        # print(flip)
        
        # フリップマスクリスト
        flip_mask = [[1] * flip[0] + [0] * (N - flip[0])]
        if N <= 2:
            flip_mask = np.ones((T_num, N), int)
        else:
            for i in range(1, T_num):
                tmp = [1] * flip[i] + [0] * (N - flip[i])
                nr.shuffle(tmp)
                # 前と重複なら振り直し
                while tmp == flip_mask[-1]:
                    nr.shuffle(tmp)
                flip_mask.append(tmp)
            flip_mask = np.array(flip_mask, bool)
        # print(flip_mask.shape)
        
        # 局所探索フリップマスクリスト
        single_flip_mask = np.eye(N, dtype=bool)
        
        """
        アニーリング＋1フリップ
        """
        # アニーリング
        # 集団まるごと温度を下げる
        for fm in flip_mask:
            # フリップ後　pool_num, N
            # pool2 = np.where(fm, 1 - pool, pool)
            pool2 = pool.copy()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
            
            operands = [hobo] + [pool2] * ho
            score2 = np.einsum(s, *operands)
            
            # 更新マスク
            update_mask = score2 < score
            # print(update_mask)
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        
        # 最後に1フリップ局所探索
        # 集団まるごと
        for fm in single_flip_mask:
            # フリップ後
            # pool2 = np.where(fm, 1 - pool, pool)
            pool2 = pool.copy()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
            
            operands = [hobo] + [pool2] * ho
            score2 = np.einsum(s, *operands)
            
            # 更新マスク
            update_mask = score2 < score
            # print(update_mask)
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        pool = pool.astype(int)
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result



class MIKASAmpler:
    def __init__(self, seed=None, mode='GPU', device='cuda:0', verbose=1):
        #乱数シード
        self.seed = seed
        self.mode = mode
        self.device_input = device
        self.verbose = verbose

    def run(self, hobomix, shots=100, T_num=2000, use_ttd=False, show=False):
        global score2
        
        #解除
        hobo, index_map = hobomix
        # print(index_map)
        
        #pytorch確認
        attention = False
        try:
            import random
            import torch
        except:
            attention = True
        if attention:
            print()
            print('=======================\n= A T T E N T I O N ! =\n=======================')
            print('ArminSampler requires PyTorch installation.\nUse "pip install torch" (or others) and ensure\n-------\nimport torch\ntorch.cuda.is_available()\n#torch.backends.mps.is_available() #if Mac\n-------\noutputs True to set up the environment.')
            print()
            sys.exit()
        
        #matrixサイズ
        N = len(hobo)
        # print(N)
        
        #次数
        ho = len(hobo.shape)
        # print(ho)
        
        # CUDA使えるか確認
        if self.mode == 'GPU' and torch.cuda.is_available():
            if self.device_input == 'cuda:0': #指示がない場合
                self.device = 'cuda:0'
            else:
                self.device = self.device_input
        elif self.mode == 'GPU' and torch.backends.mps.is_available():
            if self.device_input == 'cuda:0': #指示がない場合
                self.device = 'mps:0'
            else:
                self.device = self.device_input
        else:
            self.mode = 'CPU'
            self.device = 'cpu'
        
        #モード表示
        if self.verbose > 0:
            print(f'MODE: {self.mode}')
            print(f'DEVICE: {self.device}')
    
        # ランダムシード
        random.seed(int(time.time()))
        nr.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        
        #シード固定
        if self.seed != None:
            random.seed(self.seed)
            nr.seed(self.seed)
            torch.manual_seed(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        # --- テンソル疑似SA ---
        #
        hobo = torch.tensor(hobo, dtype=torch.float32, device=self.device).float()
        # print(hobo.shape)
        
        #TT分解を使用する場合
        tt_cores = []
        if use_ttd:
            print(f'TTD: {use_ttd}')
            tt_cores = TT_SVD(hobo)
            # print(len(tt_cores))
            # print(tt_cores[0].shape)
            # print(tt_cores[1].shape)
        
        # プール初期化
        pool_num = shots
        pool = torch.randint(0, 2, (pool_num, N), dtype=torch.float32, device=self.device).float()
        
        # スコア初期化
        # score = torch.sum((pool @ qmatrix) * pool, dim=1, dtype=torch.float32)
        score = torch.zeros(pool_num, dtype=torch.float32)
        # print(score)
        
        """
        #旧実装
        # einsum("ijk,i,j,k->", hobo, x, x, x)
        # einsum("ijk,Ni,Nj,Nk->N", hobo, pool, pool, pool)
        
        #スコア計算コマンド
        k = ',Na,Nb,Nc,Nd,Ne,Nf,Ng,Nh,Nj,Nk,Nl,Nm,Nn,No,Np,Nq,Nr,Ns,Nt,Nu,Nv,Nw,Nx,Ny,Nz'
        l = 'abcdefghjklmnopqrstuvwxyz'
        s = l[:ho] + k[:3*ho] + '->N'
        # print(s)
        command = 'global score2\r\n'
        command += f'score2 = torch.zeros(pool_num, dtype=torch.float32)\r\n'
        command += f'score2 = torch.einsum(\'{s}\', hobo, pool2' + ', pool2' * (ho - 1) + f')\r\n'
        print(command)
        
        #スコア計算
        pool2 = pool
        exec(command)
        score = score2
        print(score)
        """
        
        #スコア計算
        k = ',Na,Nb,Nc,Nd,Ne,Nf,Ng,Nh,Nj,Nk,Nl,Nm,Nn,No,Np,Nq,Nr,Ns,Nt,Nu,Nv,Nw,Nx,Ny,Nz'
        l = 'abcdefghjklmnopqrstuvwxyz'
        if use_ttd:
            ltt = ['aA', 'AbB', 'BcC', 'CdD', 'DeE', 'EfF', 'FgG', 'GhH', 'HiJ', 'JjK', 'KkL', 'LlM', 'MmO', 'OnP', 'PoQ', 'QpR', 'RqS', 'SrT', 'TsU', 'UuV', 'VvW', 'WwX', 'XxY', 'YyZ', 'Zz']
            ltt = ltt[:ho][:]
            if len(ltt[-1]) == 3:
                ltt[-1] = ltt[-1][:2]  # 両端は 2 階のテンソルなので 2 つのインデックスのみ
            s = ','.join(ltt) + k[:3*ho] + '->N'
            operands = tt_cores + [pool] * ho
        else:
            s = l[:ho] + k[:3*ho] + '->N'
            operands = [hobo] + [pool] * ho
        # print(s)
        
        score = torch.einsum(s, *operands)
        # print(score)
        
        # フリップ数リスト（2個まで下がる）
        flip = np.sort(nr.rand(T_num) ** 2)[::-1]
        flip = (flip * max(0, N * 0.5 - 2)).astype(int) + 2
        #print(flip)
        
        # フリップマスクリスト
        flip_mask = [[1] * flip[0] + [0] * (N - flip[0])]
        if N <= 2:
            flip_mask = np.ones((T_num, N), int)
        else:
            for i in range(1, T_num):
                tmp = [1] * flip[i] + [0] * (N - flip[i])
                nr.shuffle(tmp)
                # 前と重複なら振り直し
                while tmp == flip_mask[-1]:
                    nr.shuffle(tmp)
                flip_mask.append(tmp)
            flip_mask = np.array(flip_mask, bool)
        flip_mask = torch.tensor(flip_mask).bool()
        #print(flip_mask.shape)
        
        # 局所探索フリップマスクリスト
        single_flip_mask = torch.eye(N, dtype=bool)
        #print(single_flip_mask)
        
        # スコア履歴
        score_history = []
        
        """
        アニーリング＋1フリップ
        """
        # アニーリング
        # 集団まるごと温度を下げる
        for fm in flip_mask:
            pool2 = pool.clone()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
    
            if use_ttd:
                operands = tt_cores + [pool2] * ho
            else:
                operands = [hobo] + [pool2] * ho
            score2 = torch.einsum(s, *operands)
    
            # 更新マスク
            update_mask = score2 < score
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
            
            # スコア記録
            score_history.append(torch.mean(score).item())
            
        # 最後に1フリップ局所探索
        # 集団まるごと
        for fm in single_flip_mask:
            pool2 = pool.clone()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
    
            if use_ttd:
                operands = tt_cores + [pool2] * ho
            else:
                operands = [hobo] + [pool2] * ho
            score2 = torch.einsum(s, *operands)
    
            # 更新マスク
            update_mask = score2 < score
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
            
        # スコア記録
        score_history.append(torch.mean(score).item())

        # 描画
        if show:
            import matplotlib.pyplot as plt
            plt.plot(range(T_num + 1), score_history)
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.show()
        
        pool = pool.to('cpu').detach().numpy().copy()
        score = score.to('cpu').detach().numpy().copy()
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result

def TT_SVD(C, bond_dims=None, check_bond_dims=False, return_sv=False):
    """TT_SVD algorithm
    I. V. Oseledets, Tensor-Train Decomposition, https://epubs.siam.org/doi/10.1137/090752286, Vol. 33, Iss. 5 (2011)
    Args:
        C (torch.Tensor): n-dimensional input tensor
        bond_dims (Sequence[int]): a list of bond dimensions.
                                   If `bond_dims` is None,
                                   `bond_dims` will be automatically calculated
        check_bond_dims (bool): check if `bond_dims` is valid
        return_sv (bool): return singular values
    Returns:
        list[torch.Tensor]: a list of core tensors of TT-decomposition
    """
    import torch

    dims = C.shape
    n = len(dims)  # n-dimensional tensor

    if bond_dims is None or check_bond_dims:
        # Theorem 2.1
        bond_dims_ = []
        for sep in range(1, n):
            row_dim = dims[:sep].numel()
            col_dim = dims[sep:].numel()
            rank = torch.linalg.matrix_rank(C.reshape(row_dim, col_dim))
            bond_dims_.append(rank)
        if bond_dims is None:
            bond_dims = bond_dims_

    if len(bond_dims) != n - 1:
        raise ValueError(f"{len(bond_dims)=} must be {n - 1}.")
    if check_bond_dims:
        for i, (dim1, dim2) in enumerate(zip(bond_dims, bond_dims_, strict=True)):
            if dim1 > dim2:
                raise ValueError(f"{i}th dim {dim1} must not be larger than {dim2}.")

    tt_cores = []
    SVs = []
    for i in range(n - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = bond_dims[i - 1]
        ri = bond_dims[i]
        C = C.reshape(ri_1 * dims[i], dims[i + 1 :].numel())
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        if S.shape[0] < ri:
            # already size of S is less than requested bond_dims, so update the dimension
            ri = S.shape[0]
            bond_dims[i] = ri
        # approximation
        U = U[:, :ri]
        S = S[:ri]
        if return_sv:
            SVs.append(S.detach().clone())
        Vh = Vh[:ri, :]
        tt_cores.append(U.reshape(ri_1, dims[i], ri))
        C = torch.diag(S) @ Vh
    tt_cores.append(C)
    tt_cores[0] = tt_cores[0].reshape(dims[0], bond_dims[0])
    if return_sv:
        return tt_cores, SVs
    return tt_cores


if __name__ == "__main__":
    pass