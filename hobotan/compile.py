import re
import symengine
import numpy as np
from sympy import Rational


def replace_function(expression, function, new_function):
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (
                replace_function(arg, function,new_function)
                for arg in expression.args
            )
        if ( expression.__class__ == symengine.Pow):
            return new_function(*replaced_args)
        else:
            return expression.func(*replaced_args)



class Compile:
    def __init__(self, expr, mode='GPU', device='cuda:0', verbose=1):
        self.expr = expr
        self.mode = mode
        self.device_input = device
        self.verbose = verbose
    
    #hoboテンソル作成
    def get_hobo(self):
        """
        get hobo data
        Raises:
            TypeError: Input type is symengine.
        Returns:
            [hobo, index_map], offset. hobo is numpy tensor.
        """
        #symengine型のサブクラス
        if 'symengine.lib' in str(type(self.expr)):
            #式を展開して同類項をまとめる
            expr = symengine.expand(self.expr)
            # print('expr')
            
            #二乗項を一乗項に変換
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
            
            #最高字数を調べながらオフセットを記録
            #項に分解
            members = str(expr).split(' ')
            
            #各項をチェック
            offset = 0
            ho = 0
            for member in members:
                # print(member)
                #数字単体ならオフセット
                try:
                    offset += float(member) #エラーなければ数字
                    # print('continue')
                    continue
                except:
                    pass
                #'*'で分解
                texts = member.split('*')
                #係数を取り除く
                try:
                    texts[0] = re.sub(r'[()]', '', texts[0]) #'(5/2)'みたいなのも来る
                    # print(texts[0])
                    float(Rational(texts[0])) #分数も対応 #エラーなければ係数あり
                    texts = texts[1:]
                except:
                    # print('err')
                    pass
                
                #最高次数の計算
                # ['-']
                # ['q2']
                # ['q3', 'q4', 'q1', 'q2']
                if len(texts) > ho:
                    ho = len(texts)
            print(f'tensor order = {ho}')
            
            # #もう一度同類項をまとめる
            # expr = symengine.expand(expr)
            # print('expr')
    
            #文字と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            # print(len(coeff_dict))
            
            #定数項を消す　{1: 25} 必ずある
            del coeff_dict[1]
            # print('del')
            # print(coeff_dict)
            
            #シンボル対応表
            # 重複なしにシンボルを抽出
            tmp = []
            # count = 0
            for key in coeff_dict.keys():
                # if count % 1000 == 0:
                #     print(count)
                # count += 1
                tmp += str(key).split('*')
            # tmp = sum([str(key).split('*') for key in coeff_dict.keys()], [])
            # print(tmp)
            # print('tmp')
            keys = list(set(tmp))
            # print(len(keys))
            
            # 要素のソート（ただしアルファベットソート）
            keys.sort()
            # print(keys)
            
            # シンボルにindexを対応させる
            index_map = {key:i for i, key in enumerate(keys)}
            # print(index_map)
            
            #量子ビット数
            num = len(index_map)
            # print(num)
            
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
                print('----- compile -----')
                print(f'MODE: {self.mode}')
                print(f'DEVICE: {self.device}')
                print('-------------------')
            
            #HOBO行列生成
            
            # インデックスと値を格納
            indices_all = []
            value_all = []
            for key, value in coeff_dict.items():
                qnames = str(key).split('*')
                indices = sorted([index_map[qname] for qname in qnames])
                indices = [indices[0]] * (ho - len(indices)) + indices
                indices_all.append(indices)
                value_all.append(float(value))
            indices_all = np.array(indices_all).T
            value_all = np.array(value_all)
            # print(indices_all.shape)
            # print(value_all.shape)
            # print(np.min(indices_all), np.max(indices_all))
            # print(np.min(value_all), np.max(value_all))
            
            # インデックスと値
            indices = torch.tensor(indices_all, dtype=torch.int, device=self.device)
            values = torch.tensor(value_all, dtype=torch.float32, device=self.device)
            
            # 疎テンソル自体は作成せず、インデックスと値を格納する
            hobo = [indices, values, num]
            
            # # 疎テンソルを作成
            # print(num)
            # print(ho)
            
            # if ho >= 8:
            #     hobo1 = torch.sparse_coo_tensor(indices_all[:6], value_all, [num] * 6, dtype=torch.float32, device=self.device)
            #     hobo2 = torch.sparse_coo_tensor(indices_all[6:], value_all, [num] * (ho - 6), dtype=torch.float32, device=self.device)
            #     hobo = [hobo1, hobo2]
            # else:
            #     hobo = torch.sparse_coo_tensor(indices_all, value_all, [num] * ho, dtype=torch.float32, device=self.device)
            # # print(hobo.shape)
            
            
            # hobo = hobo.to_dense()
            # print(hobo.dtype)
            
            # hobo = np.zeros(num ** ho, dtype=float)
            # hobo = hobo.reshape([num] * ho)
            # for key, value in coeff_dict.items():
            #     qnames = str(key).split('*')
            #     indices = sorted([index_map[qname] for qname in qnames])
            #     indices = [indices[0]] * (ho - len(indices)) + indices
            #     hobo[tuple(indices)] = float(value)
            
            return [hobo, index_map], offset

        else:
            raise TypeError("Input type must be symengine.")

