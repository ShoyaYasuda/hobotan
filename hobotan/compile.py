import re
import sys
import symengine
import numpy as np
from sympy import Rational


"""
2026-02-02
PieckCompile, PieckSamplerを標準のCompile, Samplerに統合し、有料オプションとする
"""


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
        
        #モード表示
        if self.verbose > 0: print(f"{' HOBOTAN v1.0.0 ':=^30}")
        # if self.verbose > 0: print(f'mode = {self.mode}')
        # if self.verbose > 0: print(f'device = {self.device}')
        if self.verbose > 0: print(f"{' compile ':-^30}")
        
    
    #hoboテンソル作成
    def get_hobo(self):

        #symengine型のサブクラス
        if 'symengine.lib' in str(type(self.expr)):
            
            """
            ステップ0
            式は展開せずワンホット情報を検出する
            特定の形式のみを想定。あまり柔軟性はもたせなくてよい
            """
            #式の文字列
            expr_str = str(self.expr)
            
            #正規表現パターン
            #\(-       : 「(-」から始まる
            #[^\*\(\)] : 「*」「(」「)」以外の文字
            #+         : 1文字以上続く
            #\)\*\*2   : 「)**2」で終わる
            pattern = r"\(-[^\*\(\)]+\)\*\*2"

            #抽出
            results = re.findall(pattern, expr_str)
            
            #結果
            onehot_keys = [] #ワンホットのキーのリストリスト
            onehot_ks = [] #Nホットの値リスト
            for res in results:
                #(-1 + q0_0 + q0_1 + q0_2 + q0_3)**2
                # print(res)
                
                #前後掃除
                #-1 + q0_0 + q0_1 + q0_2 + q0_3
                inner = res[1:-4]
                # print(inner)
                
                #'+'や空白を除いて分割
                parts = [p.strip() for p in re.split(r'\+', inner) if p.strip()]
                # print(parts)
                
                #定数項が「マイナス実質整数」かチェック
                if float(parts[0]).is_integer():
                    k = -int(float(parts[0]))
                    # print(k)
                else:
                    continue
                
                #文字の項
                variables = parts[1:]
                # print(variables)
                
                #2項以上あること
                if len(variables) < 2:
                    continue
                
                #クリアなら記録
                onehot_keys.append([v for v in variables]) #symengine
                onehot_ks.append(k) #int
            # print(onehot_keys)
            # print(onehot_ks)
            
            #長いワンホットから順に採用、重複採用しない
            #ワンホット長さ
            onehot_len = [len(onehot_key) for onehot_key in onehot_keys]
            # print(onehot_len)
            #長い順インデックス
            index = np.argsort(onehot_len)[::-1]
            # print(index)
            
            seen_keys = set() #すでに検出されたキーセット
            onehot_keys2 = [] #ワンホットのキーのリストリスト
            onehot_ks2 = [] #Nホットの値リスト
            for idx in index:
                #キーがいずれも採用済みでない
                flg = all([key not in seen_keys for key in onehot_keys[idx]])
                # print(flg)
                
                #クリアなら記録
                if flg:
                    seen_keys.update(set(onehot_keys[idx]))
                    onehot_keys2.append(onehot_keys[idx])
                    onehot_ks2.append(onehot_ks[idx])
            # print(onehot_keys2)
            # print(onehot_ks2)
            
            
            """
            ステップ1
            二乗項を一乗項に変換すると相殺される項があるので、まだしない
            最高次数を調べる、オフセットを調べる、使用されているシンボルを記録しておく
            """
            #式を展開して浅い同類項をまとめる
            expr = symengine.expand(self.expr)
            # print(expr)
            
            #項と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            # print(coeff_dict)
            
            symbols = set() #使用されている項セット（1乗に下ろしたが、文字のかけ算は残ったもの）
            offset = 0 #オフセット
            ho = 0 #最高次数（二乗は一乗に戻すが、文字のかけ算は残ったもの、の次数）
            
            #各項をチェック
            for key in coeff_dict.keys():
                #数字単体ならオフセット
                if key.is_Number:
                    offset = coeff_dict[key]
                    continue
                #2乗以上を1乗に下ろす
                key2 = replace_function(key, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
                # print(key2)
                #使用されている項（1乗に下ろしたが、文字のかけ算は残ったもの）
                symbols.add(key2)
                #最高次数チェック
                ho = max(sum(1 for arg in key2.args if arg.is_Symbol), ho)
            
            #項リストから定数を意味する1を削除
            symbols.discard(1) #存在しなくてもエラーにならない
            
            # print(symbols)
            if self.verbose > 0:
                if ho <= 2:
                    print(f'tensor order = {ho} (QUBO)')
                else:
                    print(f'tensor order = {ho} (HOBO)')
            if self.verbose > 0: print(f'offset = {offset}')
            
            
            """
            ステップ2
            2乗以上を1乗に下ろす
            """
            #2乗以上を1乗に下ろす（ここで相殺される項がある） q0**2 - q0
            expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
            # print(expr)
            
            #自動的に項が整理されている
            
            #定数項を消す
            expr -= offset
            # print(expr)
            
            
            """
            合流
            """
            #項と係数の辞書
            coeff_dict = expr.as_coefficients_dict()
            # print(len(coeff_dict))
            # print(coeff_dict)
            
            #ここで、2乗以上を1乗項に変換したときに消えた項を係数0で復元
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
            
            #シンボル一覧、文字列型、重複なし
            keys = set()
            # count = 0
            for key in coeff_dict.keys():
                keys.update(set(str(key).split('*')))
            keys = list(keys)
            # print(keys)
            
            # 要素のソート（ただしアルファベットソート）
            keys.sort()
            # print(keys)
            
            # シンボルにindexを対応させる
            index_map = {key:i for i, key in enumerate(keys)}
            # print(index_map)
            
            #量子ビット数
            num = len(index_map)
            if self.verbose > 0: print(f'qbit num = {num}')
            
            #ワンホット表示
            if self.verbose > 0: print(f'onehot group = {len(onehot_keys2)}')


            """
            ステップ3
            HOBO行列生成、ただし
            """
            import torch
            # テンソルのインデックスと値
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
            hobo_mix = [indices, values, index_map, num, onehot_keys2, onehot_ks2]
            
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
            
            return hobo_mix, offset

        else:
            raise TypeError("Input type must be symengine.")




if __name__ == "__main__":
    #テスト用
    # from symbol import symbols_list
    
    pass
    
    