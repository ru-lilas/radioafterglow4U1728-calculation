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
