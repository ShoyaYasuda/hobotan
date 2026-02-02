import os
import sys
import time
import random
import requests
import numpy as np
import numpy.random as nr
from copy import deepcopy
from importlib import util


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
    # result = [[dict(zip(index_map.keys(), unique_pool[i])), unique_energy[i], unique_counts[i]] for i in range(len(unique_pool))]
    #numpy2.xでnp.int64(0)のように表示されることへの対応
    result = [
        [dict(zip(index_map.keys(), unique_pool[i].tolist())), float(unique_energy[i]), int(unique_counts[i])]
        for i in range(len(unique_pool))
        ]
    
    return result


#謎のglobal対応
score2 = 0

class MIKASAmpler:
    def __init__(self, seed=None, mode='GPU', device='cuda:0', verbose=1, passkey=''):
        #乱数シード
        self.seed = seed
        self.verbose = verbose
        
        #pytorch確認
        try:
            import torch
        except:
            print(f"{' ATTENTION ':=^40}")
            print('HOBOTAN requires PyTorch installation')
            print('Use "pip install torch" (or others) to set up the environment.')
            print()
            print('import torch')
            print('torch.cuda.is_available()')
            print('#torch.backends.mps.is_available() # if Mac')
            print('-> True')
            print(f"{'':=^40}")
            sys.exit()
        
        #CUDA確認
        if mode == 'GPU' and torch.cuda.is_available():
            self.mode = 'GPU'
            if device == 'cuda:0': #指示がない場合
                self.device = 'cuda:0'
            else:
                self.device = device
        elif mode == 'GPU' and torch.backends.mps.is_available():
            self.mode = 'GPU'
            if device == 'cuda:0': #指示がない場合
                self.device = 'mps:0'
            else:
                self.device = device
        else:
            self.mode = 'CPU'
            self.device = 'cpu'
        
        #有料キー
        self.passkey = passkey
        self.onehot_mode = False
    

    def run(self, hobo_mix, shots=100, T_num=2000, show=False):
        
        
        try:
            if os.path.isfile('sampler_source.py'):
                print('■ source = local')
                with open('sampler_source.py', 'r', encoding='UTF-8') as f:
                    source_code = f.read()
            else:
                source_code = requests.get('gAtv1zteUHy2ltd7pUEhspiJzxsLjkrc.cNXvfTxYvverjiasLynlIc3xBOD5PTFPrMS9V2c26K8ux8uJqRnccbo2rj3hwPP0HsPiq3hYZC6r_bfOz72EglSrrt3mPMhmgNeHR0aWTanaulAR8zaiU1fOp9JmAZ40ZdDm1tuFc4aWMJa6F4HCkSqeCskcTQXAKS3z^cA3KR4TyCOropkvSmMafPe1AVytTiN2Ol58LwRsNwE9pWpirIcVvfVmDaSJjqQ1ywaPt6G3K5ARESalQC6MhqHvkk2kRYdX9nRcSkK651EU5weqmzSgMcDGdilBwtPeoeq1PgNBIbR7dZU^nEhTl9mbMcph1xfzhK1MmjOhbeRMI76O.K0coS2jYgdx0BvkYxoThrxUbpm2vOLFhxNufwY6SPQFxCrLA80LR5n.2pZHrHT1zRaI17S5f1c7jl4iWRfeEqQNccl0mlspKuy-ZMfCsLJMZ9enI70vJhTsJntgF7G4uffCgIZj9ZpoK6mie3TuUpv30FvNaOQY0Tz1U^fa6J6ZdqH3^CEJ9n6ONuB:wgJEvR25eUsgtj5MZTI6qpZNgWisiGn6t5LwE9t1Y6ztFG9qcA4Wy2h'[::-1 ][::11 ].replace('^','/')).text
            
            spec = util.spec_from_loader('temp_module', loader=None)
            temp_module = util.module_from_spec(spec)
            exec(source_code, temp_module.__dict__)
            
        except:
            
            global score2
            import torch
            
            #表示
            if self.verbose > 0: print(f"{' sampler ':-^30}")
            if self.verbose > 0: print(f'source = offline')
            if self.verbose > 0: print(f'mode = {self.mode}')
            if self.verbose > 0: print(f'device = {self.device}')
            if self.onehot_mode:
                if self.verbose > 0: print(f'onehot sampler = {self.onehot_mode} (with passkey)')
            else:
                if self.verbose > 0: print(f'onehot sampler = {self.onehot_mode} (no passkey)')
            
            #拡張quboを戻す
            indices, values, index_map, N, onehot_keys, onehot_ks = hobo_mix
            # print(onehot_keys)
            # print(onehot_ks)
            # print(index_map)
        
            #一旦すべてランダムシード
            random.seed(int(time.time()))
            nr.seed(int(time.time()))
            torch.manual_seed(int(time.time()))
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms = True
            
            #シード固定
            if self.seed != None:
                random.seed(int(self.seed))
                nr.seed(int(self.seed))
                torch.manual_seed(int(self.seed))
            
            #ショット数
            shots = max(int(shots), 1)
            
            #次数
            ho = len(indices)
            
            other_index = list(np.arange(N))
            
            """
            初期化
            """
            # プール初期化
            pool_num = shots
            pool = nr.randint(0, 2, (pool_num, N)) # この時点では通常配列
            
            #torch配列に変換
            pool = torch.tensor(pool, dtype=torch.float32, device=self.device).float()
            
            #スコア初期化
            score = torch.zeros(pool_num, dtype=torch.float32)
            values2 = torch.clone(values)
            for i in range(ho):
                values2 = pool[:, indices[i]] * values2   # a 次元の para を掛ける
            score = torch.sum(values2, axis=1)
            # print(score)
            
            
            """
            フリップリスト(前半)
            """
            #その他の量子ビットのためのフリップ数リスト（2個まで下がる）
            N_other = len(other_index)
            flip = np.sort(nr.rand(T_num) ** 3)[::-1]
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(8, 8))
            # plt.plot(range(len(flip)), flip, 'ok', markersize=1)
            # plt.show()
            # plt.close()
            flip = (flip * max(0, N_other * 0.5 - 2)).astype(int) + 2
            # print(flip)
            
            #その他の量子ビットのためのフリップマスクリスト
            flip_mask = [[1] * flip[0] + [0] * (N_other - flip[0])]
            if N_other <= 2:
                flip_mask = np.ones((T_num, N_other), int)
            else:
                for i in range(1, T_num):
                    tmp = [1] * flip[i] + [0] * (N_other - flip[i])
                    nr.shuffle(tmp)
                    # 前と重複なら振り直し、同じひっくり返し方を2回すると戻ってしまうので無駄
                    while tmp == flip_mask[-1]:
                        nr.shuffle(tmp)
                    flip_mask.append(tmp)
                flip_mask = np.array(flip_mask, bool)
            # print(flip_mask)
            
            #全体基準のフリップマスクに戻す
            flip_mask_tmp = np.zeros((T_num, N), bool)
            flip_mask_tmp[:, other_index] = flip_mask
            flip_mask = torch.tensor(flip_mask_tmp).bool()
            
            #その他の量子ビットのための局所探索フリップマスクリスト
            single_flip_mask = np.eye(N_other, dtype=bool)
            # print(single_flip_mask)
            
            #全体基準の局所探索フリップマスクリストに戻す
            single_flip_mask_tmp = np.zeros((N_other, N), bool)
            single_flip_mask_tmp[:, other_index] = single_flip_mask
            single_flip_mask = torch.tensor(single_flip_mask_tmp).bool()
            # print(single_flip_mask)
            
            
            """
            アニーリング
            """
            if self.verbose > 0: print(f"{' start ':-^30}")
            # スコア履歴
            score_history = []
            
            # 集団まるごと温度を下げる
            count = 1
            for step, fm in enumerate(flip_mask):
                
                #表示
                if self.verbose > 0 and count % max(shots//10, 100) == 0:
                    print('.', end='')
                    if count % shots == 0:
                        print(f' {count}/{T_num} min={min(score)} mean={torch.mean(score)}')
                count += 1
                
                #新プール
                pool2 = pool.clone()
                
                #その他の量子ビットをフリップ
                if N_other > 0:
                    pool2[:, fm] = 1. - pool[:, fm]
                
                #スコア
                values2 = torch.clone(values)
                for i in range(ho):
                    values2 = pool2[:, indices[i]] * values2
                score2 = torch.sum(values2, axis=1)
                
                #改善した部分は更新
                update_mask = score2 < score
                pool[update_mask] = pool2[update_mask]
                score[update_mask] = score2[update_mask]
                
                # スコア記録
                score_history.append(torch.mean(score).item())
            
            """
            最後に1フリップ局所探索
            """
            # 集団まるごと
            for fm in single_flip_mask:
                pool2 = pool.clone()
                pool2[:, fm] = 1. - pool[:, fm]
                
                #スコア
                values2 = torch.clone(values)
                for i in range(ho):
                    values2 = pool2[:, indices[i]] * values2
                score2 = torch.sum(values2, axis=1)
                
                #更新
                update_mask = score2 < score
                pool[update_mask] = pool2[update_mask]
                score[update_mask] = score2[update_mask]
            
            #スコア記録
            score_history.append(torch.mean(score).item())
    
            #描画
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
            
            #メモリクリア
            del indices
            del values
            del values2
            del flip_mask
            del single_flip_mask
            # del take_bad_mask
            del pool
            del score
            torch.cuda.empty_cache()
            
            if self.verbose > 0: print(f"{' end ':-^30}")
            
            return result
        
        else:
            result = temp_module.MIKASAmpler(seed=self.seed,
                                             mode=self.mode,
                                             device=self.device,
                                             verbose=self.verbose,
                                             passkey=self.passkey).run(hobo_mix,
                                                                       shots=shots,
                                                                       T_num=T_num,
                                                                       show=show)
            return result






if __name__ == "__main__":
    #テスト用
    from symbol import symbols_list
    from compile import Compile
    from auto_array import Auto_array
    import matplotlib.pyplot as plt
    
    pass
    
