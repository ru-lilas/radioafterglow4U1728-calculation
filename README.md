<!-- OmegaConfを使っているので以下でインストールすること。 -->
<!---->
<!-- ```bash -->
<!-- conda install -c conda-forge omegaconf -->
<!-- ``` -->

## scripts
- `integral_table.py`: Thermal synchrotron functionの積分数表を作成するスクリプト
- `table_peak_values.py`: tau_thetaからスペクトル/光度曲線のピーク値を計算し、数表を作成するスクリプト
- `find_chi2fit_parameters.py`: chi2の計算結果を読んで最小値とそのときのパラメータを得るスクリプト


## module
- `quantity_converter.py`: ある物理量を別の物理量に変換する処理は全部ここ
- `plot_utils.py`: プロット関連の処理は全部ここ

# 日記
## 2026/07/15
基本的には信頼領域が狭いほど「よい」推定といえる。
ひとつの固定パラメータに対して、
どの範囲で時間窓をとればchi2の信頼領域が最も狭くなるのかを調べる。

まずは各時間窓の場合でchi2の最小値からのずれ$\Delta \chi^2 = \chi^2 - \chi^2_{\mathrm{min}}$を計算する

