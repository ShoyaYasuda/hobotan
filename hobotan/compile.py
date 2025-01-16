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
            # print(expr)
            
            '''
            ステップ1
            二乗項を一乗項に変換すると相殺される項があるので、その前に
            最高次数を調べる、オフセットを調べる、使用されているシンボルを記録しておく
            '''
            #項と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            # print(coeff_dict)
            
            #
            symbols = set() #使用されている項リスト
            offset = 0 #オフセット
            ho = 0 #最高次数
            
            #各項をチェック
            for key in coeff_dict.keys():
                # print(key)
                #数字単体ならオフセット
                if key.is_Number:
                    offset = coeff_dict[key]
                    continue
                #最高次数チェック
                ho = max(sum(1 for arg in key.args if arg.is_Symbol), ho)
                #使用されている項（ただし二乗項は一乗項に変換しておく）
                symbols.add(replace_function(key, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e))
            
            #項リストから定数を意味する1を削除
            symbols.discard(1) #存在しなくてもエラーにならない
            
            # print(symbols)
            # print(offset)
            print(f'tensor order = {ho}')
            
            '''
            ステップ2
            二乗項を一乗項に変換する
            '''
            #二乗項を一乗項に変換（ここで相殺される項がある）
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
            # print(expr)
            
            # #もう一度同類項をまとめる
            # expr = symengine.expand(expr)
            # print('expr')
            
            #定数項を消す
            expr -= offset
            # print(expr)
            
            '''
            合流
            '''
            #項と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            
            # print(len(coeff_dict))
            # print(coeff_dict)
            
            #ここで、二乗項を一乗項に変換したときに消えた項を係数0で戻す
            for key in symbols:
                if key not in coeff_dict.keys():
                    coeff_dict[key] = 0
            # print(coeff_dict)
            
            #定数項を消す、項がない場合{1: 0}が残っているため
            try:
                del coeff_dict[0] #{0: 1}が残っている場合（たぶんバグ）
            except:
                pass
            try:
                del coeff_dict[1] #{1: 0}が残っている場合
            except:
                pass
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
            # print(keys)
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

