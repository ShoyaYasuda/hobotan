## 概要

GPU利用を前提としています。別途Pytorchをインストールしておいてください。

v0.1.0でメモリオーバー対策をしました（できてるか？）。324量子ビットの8次式での動作を確認しています。

## インストール・アップデート
```
pip install -U git+https://github.com/ShoyaYasuda/hobotan
```

## サンプルコード1

量子ビットは2次元配列で定義でき、定式化でq[0, 0]のように使用できる。（3次元もできるよ）

MIKASAmpler()を使用（別途pytorchをインストールしておくこと）。shots=10000のように増やせる。

以下のコードは、5✕5の席にできるだけ多くの生徒を座らせる（ただし縦・横に3席連続で座ってはいけない）を解いたもの。3次の項が登場する。

```python
import numpy as np
from hobotan import *
import matplotlib.pyplot as plt

#量子ビットを用意
q = symbols_list([5, 5], 'q{}_{}')

#すべての席に座りたい（できれば）
H1 = 0
for i in range(5):
    for j in range(5):
        H1 += - q[i, j]

#どの直線に並ぶ3席も連続で座ってはいけない（絶対）
H2 = 0
for i in range(5):
    for j in range(5 - 3 + 1):
        H2 += np.prod(q[i, j:j+3])
for j in range(5):
    for i in range(5 - 3 + 1):
        H2 += np.prod(q[i:i+3, j])

#式の合体
H = H1 + 10*H2

#HOBOテンソルにコンパイル
hobo, offset = Compile(H).get_hobo()
print(f'offset\n{offset}')

#サンプラー選択
solver = sampler.MIKASAmpler()

#サンプリング
result = solver.run(hobo, shots=10000)

#上位3件
for r in result[:3]:
    print(f'Energy {r[1]}, Occurrence {r[2]}')

    #さくっと配列に
    arr, subs = Auto_array(r[0]).get_ndarray('q{}_{}')
    print(arr)

    #さくっと画像に
    img, subs = Auto_array(r[0]).get_image('q{}_{}')
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.show()
```
```
offset
0
MODE: GPU
DEVICE: cuda:0
Energy -17.0, Occurrence 686
[[1 1 0 1 1]
 [1 1 0 1 1]
 [0 0 1 0 0]
 [1 1 0 1 1]
 [1 1 0 1 1]]
Energy -17.0, Occurrence 622
[[1 1 0 1 1]
 [1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]
 [1 0 1 1 0]]
Energy -17.0, Occurrence 496
[[0 1 1 0 1]
 [1 1 0 1 1]
 [1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]]
```
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img1.png" width="%">
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img2.png" width="%">
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img3.png" width="%">


## サンプルコード2

以下のコードはx^2+y^2=z^2を満たすピタゴラス数（x, y, zとも1～16）を求めたもの。4次の項が登場する。

```python
from hobotan import *

#量子ビットを2進数表現で用意
x = symbols_nbit(0, 16, 'x{}', num=4) + 1
y = symbols_nbit(0, 16, 'y{}', num=4) + 1
z = symbols_nbit(0, 16, 'z{}', num=4) + 1

#ピタゴラス条件
H = (x**2 + y**2 - z**2)**2

#HOBOテンソルにコンパイル
hobo, offset = Compile(H,).get_hobo()
print(f'offset\n{offset}')

#サンプラー選択
solver = sampler.MIKASAmpler()

#サンプリング
result = solver.run(hobo, shots=10000)

#上位10件
for r in result[:10]:
    print(f'Energy {r[1]}, Occurrence {r[2]}')
    
    #さくっと10進数に戻す
    print('x =', Auto_array(r[0]).get_nbit_value(x))
    print('y =', Auto_array(r[0]).get_nbit_value(y))
    print('z =', Auto_array(r[0]).get_nbit_value(z))
```
```
offset
1.0
MODE: GPU
DEVICE: cuda:0
Energy -1.0, Occurrence 1105
x = 8.0
y = 6.0
z = 10.0
Energy -1.0, Occurrence 643
x = 12.0
y = 9.0
z = 15.0
Energy -1.0, Occurrence 781
x = 12.0
y = 5.0
z = 13.0
Energy -1.0, Occurrence 1532
x = 3.0
y = 4.0
z = 5.0
Energy -1.0, Occurrence 1461
x = 4.0
y = 3.0
z = 5.0
Energy -1.0, Occurrence 860
x = 5.0
y = 12.0
z = 13.0
Energy -1.0, Occurrence 688
x = 6.0
y = 8.0
z = 10.0
Energy -1.0, Occurrence 1050
x = 9.0
y = 12.0
z = 15.0
Energy 0.0, Occurrence 108
x = 11.0
y = 1.0
z = 11.0
Energy 0.0, Occurrence 29
x = 10.0
y = 1.0
z = 10.0
```


## その他の使い方

論文/Paper　[HOBOTAN: Efficient Higher Order Binary Optimization Solver with Tensor Networks and PyTorch](https://blueqat.com/bqresearch/39fc4433-2907-4f43-a913-a294953b7e60)

ブログ　[量子アニーリング（HOBO対応）のパッケージ「HOBOTAN」が登場](https://vigne-cla.com/21-41/)

ブログ　[新型ソルバーの試作品、HOBOソルバー hobotan ができたようです。](https://blueqat.com/yuichiro_minato2/b562b955-0de8-4b6f-b092-15785a099c13)

blog　[A prototype of the new solver, HOBO solver "hobotan," is now available.](https://blueqat.com/yuichiro_minato2/b79a33dd-875d-4772-a11a-c6a80888a212)

ブログ　[HOBOソルバーでグラフカラーリング問題を効率化](https://blueqat.com/yuichiro_minato2/ae758ca8-27fe-43e8-8bdc-2171dfc3c01e)

blog　[Optimizing Graph Coloring Problems with the HOBO Solver](https://blueqat.com/yuichiro_minato2/de1d6041-1eb5-4fab-9776-73ab82270836)

[TYTANパッケージ](https://github.com/tytansdk/tytan) の派生形なのでそちらも参照してみてください。


## 開発㌠

derwindさん（理論）、yuminさん（マネージャー）、Shoya Yasudaさん（実装）

## 更新履歴
|日付|ver|内容|
|:---|:---|:---|
|2024/10/13|0.1.2|ランダム性を追加|
|2024/10/13|0.1.1|ソルバー実行後にGPUメモリクリア|
|2024/10/13|0.1.0|メモリオーバー対策、余計な要素を削除|
|2024/07/28|0.0.8|TT分解オプションを追加（未検証）|
|2024/07/27|0.0.7|exec(command)を解除|
|2024/07/27|0.0.6|compileのミスを修正|
|2024/07/27|0.0.5|symbols_nbitを追加|
|2024/07/26|0.0.4|MIKASAmplerに改名|
|2024/07/26|0.0.3|テンソル計算を高速化|
|2024/07/26|0.0.2|いろいろ修正|
|2024/07/26|0.0.1|初期版|

