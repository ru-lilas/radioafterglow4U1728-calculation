# Chi-square testによるモデル適合度評価
モデルから計算される光度曲線が観測データを説明できるか
どうかを評価するため、chi-square testを行った。

観測データを $d=(y_i^{\mathrm{obs}},\sigma_i)_{i=1}^{N}$、
モデルから得られる値を $y_i^{\mathrm{model}}(\theta)$とする。
$\theta$は自由パラメータであり、今回は$\theta = (A_{\mathrm{w}},\beta_{\mathrm{sh}})$である。

$\chi^2(\theta)$を以下で定義する：
$$
    \chi^2(\theta) \coloneqq 
    \sum_{i=1}^{N}
    \left(
        \frac{y_i^{\mathrm{obs}}-y_i^{\mathrm{model}}(\theta)}{\sigma_i}
    \right)^2
$$

## "best-fit" パラメータ
本研究では、$\theta$をある範囲で動かしたときの$\chi^2(\theta)$を計算し、その最小値を
"best fit"と定義する。すなわち
$$
    \chi^2_{\mathrm{min}} \coloneqq \min_{\theta} \chi^2(\theta) \\
    \hat{\theta} \coloneqq
    \arg \chi^2_{\mathrm{min}}
$$
である。

ここで得られた$\hat{\theta}$はあくまでも$\chi^2$を最小とする
パラメータとして定義されたものであり、観測データと統計的に整合するかどうかについては
保証していない。

「$\hat{\theta}$が真値であると仮定したとき、
今回得られた$\chi^2_{\min}$は統計的に妥当であるかどうか」
を検証するために、chi-square testを行う。

### Chi-square testの帰無仮説
> 帰無仮説$H_0$：モデル$\hat{\theta}$は、$d$を観測誤差$(\sigma_i)_{i=1}^{N}$の範囲内で説明できる。

帰無仮説の下では、$\chi^2_{\min}$は自由度$\nu = N - n$
の$\chi^2$分布に従うことが知られている。
ここで$n$は自由パラメータの数で、今回は$n=2$である。

## p値
帰無仮説の下で、今回得られた$\chi^2_{\min,\mathrm{obs}}$の値よりも大きな$\chi^2$が得られる確率
$$
    p = P(\chi^2 \geq \chi^2_{\min,\mathrm{obs}})
$$
を考える。
