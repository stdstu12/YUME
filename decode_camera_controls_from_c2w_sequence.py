
def decode_camera_controls_from_c2w_sequence(cam_c2w, stride=1, translation_threshold=0.0001, rotation_threshold=0.001):
    c2w_matrices, translation_threshold, rotation_threshold = (
        cam_c2w[::stride],
        translation_threshold * stride,
        rotation_threshold * stride,
    )

    control_sequence = []

    for i in range(len(c2w_matrices) - 1):
        T_curr = c2w_matrices[i]
        T_next = c2w_matrices[i + 1]

        # 相对变换（以当前帧为原点）
        T_rel = np.linalg.inv(T_curr) @ T_next
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]  # 平移（在当前相机坐标系中）

        # —— 键盘方向判定（WASD） ——
        keys = []
        x_move, _, z_move = t_rel

        if z_move > translation_threshold:
            keys.append("W")
        if z_move < -translation_threshold:
            keys.append("S")
        if x_move > translation_threshold:
            keys.append("D")
        if x_move < -translation_threshold:
            keys.append("A")
        key_command = "+".join(keys) if keys else "None"

        # —— 鼠标方向（Yaw + Pitch） ——
        # 从旋转矩阵提取欧拉角（YXZ顺序：yaw, pitch, roll）
        roc = Rotation.from_matrix(R_rel).as_euler("xyz", degrees=False)

        # 量化方向（上下 & 左右）
        mouse_horizontal = None
        mouse_vertical = None

        if roc[1] > rotation_threshold:
            mouse_horizontal = "→"
        elif roc[1] < -rotation_threshold:
            mouse_horizontal = "←"

        if roc[0] > rotation_threshold:
            mouse_vertical = "↑"  # 抬头
        elif roc[0] < -rotation_threshold:
            mouse_vertical = "↓"  # 低头

        # 组合方向
        if mouse_horizontal and mouse_vertical:
            mouse_dir = mouse_vertical + mouse_horizontal  # 如 ↑→ 表示 ↗
        else:
            mouse_dir = mouse_horizontal or mouse_vertical or "·"  # 无旋转

        control_sequence.append({"frame": i, "keys": key_command, "mouse": mouse_dir})

    return control_sequence
