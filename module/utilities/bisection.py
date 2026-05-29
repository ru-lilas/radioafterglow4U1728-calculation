def bisection(f, a:float, b:float, tol:float=1e-8, max_iter:int=100):
    """
    2分法で f(x)=0 の解を求める。

    Parameters
    ----------
    f : callable
        関数 f(x)
    a, b : float
        初期区間 [a, b]
        f(a) と f(b) は異符号である必要がある
    tol : float
        許容誤差
    max_iter : int
        最大反復回数

    Returns
    -------
    float
        求めた近似解
    """

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("f(a) と f(b) の符号が同じです。")

    for i in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)

        # 収束判定
        if abs(fc) < tol or abs(b - a) < tol:
            return c

        # 区間更新
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise RuntimeError("最大反復回数に達しました。")
