import numpy as np


def calculate_sign(u, v, tol=1e-10):

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # 外積
    w = np.cross(u, v)

    # u または v が ±z 軸上か
    u_is_z = np.allclose(np.abs(u), [0, 0, 1], atol=tol)
    v_is_z = np.allclose(np.abs(v), [0, 0, 1], atol=tol)

    # ==================================================
    # u または v が ±z 軸の場合
    # ==================================================
    if u_is_z or v_is_z:

        # まず w[1] を見る
        if w[1] > tol:
            return 1
        elif w[1] < -tol:
            return -1

        # w[1] == 0 なら w[0] を見る
        elif w[0] > tol:
            return 1
        elif w[0] < -tol:
            return -1

    # ==================================================
    # 通常の場合
    # ==================================================
    else:

        # 基本は w[2] を見る
        if w[2] > tol:
            return 1
        elif w[2] < -tol:
            return -1

        # w[2] == 0 なら w[0] を見る
        elif w[0] > tol:
            return 1
        elif w[0] < -tol:
            return -1

        # まず w[1] を見る
        if w[1] > tol:
            return 1
        elif w[1] < -tol:
            return -1

    raise ValueError(
        f"sign を判定できません。\n"
        f"u = {u}\n"
        f"v = {v}\n"
        f"w = {w}"
    )

u = np.array([0, 0, -1])
v = np.array([0, 1, 0])

print(calculate_sign(u, v))