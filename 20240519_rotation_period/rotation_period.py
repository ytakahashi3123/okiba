import numpy as np

def calculate_rotation_period(omega_x, omega_y, omega_z):
    """
    任意のオイラー角の角速度を与えた場合の1周分の時間を計算する関数

    Parameters:
    omega_x (float): x軸周りの角速度
    omega_y (float): y軸周りの角速度
    omega_z (float): z軸周りの角速度

    Returns:
    float: 1周分の時間
    """
    # 角速度ベクトルの大きさを計算
    omega_magnitude = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    
    # 1周分の時間を計算
    rotation_period = 2 * np.pi / omega_magnitude
    
    return rotation_period

# サンプルデータ（任意の角速度）
omega_x = 10*np.pi/180
omega_y = 5*np.pi/180
omega_z = 0*np.pi/180

# 1周分の時間を計算
rotation_period = calculate_rotation_period(omega_x, omega_y, omega_z)

# 結果の表示
print(f"1周分の時間: {rotation_period} 秒")
