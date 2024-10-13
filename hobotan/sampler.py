import sys
import time
import numpy as np
import numpy.random as nr
from copy import deepcopy

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

class MIKASAmpler:
    def __init__(self, seed=None, mode='GPU', device='cuda:0', verbose=1):
        #乱数シード
        self.seed = seed
        self.mode = mode
        self.device_input = device
        self.verbose = verbose

    def run(self, hobomix, shots=100, T_num=2000, show=False):
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
            print('===== sampler =====')
            print(f'MODE: {self.mode}')
            print(f'DEVICE: {self.device}')
            print('===================')
    
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
        
        
        
        # インデックスと値復元
        indices, values, N = hobo
        # hobo_coalesced = hobo.coalesce()
        # indices = hobo_coalesced.indices()  # インデックスを取得
        # values = hobo_coalesced.values()    # 値を取得
        # print(indices)
        # print(values)
        
        #matrixサイズ
        # N = len(indices[0])
        # print(N)
        
        #次数
        ho = len(indices)
        # print(ho)
        
        
        # --- テンソル疑似SA ---
        #
        # hobo = torch.tensor(hobo, dtype=torch.float32, device=self.device).float()
        # print(hobo.shape)
        
        # プール初期化
        pool_num = shots
        pool = torch.randint(0, 2, (pool_num, N), dtype=torch.float32, device=self.device).float()
        
        # スコア初期化
        # score = torch.sum((pool @ qmatrix) * pool, dim=1, dtype=torch.float32)
        score = torch.zeros(pool_num, dtype=torch.float32)
        values2 = torch.clone(values)
        for i in range(ho):
            values2 = pool[:, indices[i]] * values2   # a 次元の para を掛ける
            # print(values2)
        score = torch.sum(values2, axis=1)
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
        count = 1
        for fm in flip_mask:
            
            if count % 10 == 0:
                print('.', end='')
                if count % 100 == 0:
                    print(f' {count}/{T_num} Energy={min(score)}')
            count += 1
            
            pool2 = pool.clone()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = torch.sum((pool2 @ qmatrix) * pool2, dim=1)
            
            # スコア
            values2 = torch.clone(values)
            for i in range(ho):
                values2 = pool2[:, indices[i]] * values2
            score2 = torch.sum(values2, axis=1)
    
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
    
            # スコア
            values2 = torch.clone(values)
            for i in range(ho):
                values2 = pool2[:, indices[i]] * values2
            score2 = torch.sum(values2, axis=1)
    
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
        pool = pool.astype(int)
        score = score.to('cpu').detach().numpy().copy()
        
        # ----------
        #共通後処理
        result = get_result(pool, score, index_map)
        
        return result






if __name__ == "__main__":
    pass
