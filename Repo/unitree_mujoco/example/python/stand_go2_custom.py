import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

# 站立姿态（原始数据）
stand_up_joint_pos = np.array([
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
], dtype=float)

# 趴下姿态（原始数据）
stand_down_joint_pos = np.array([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
], dtype=float)

crc = CRC()

# 存储当前关节角度（用于调试）
current_joint_pos = np.zeros(12)

def LowStateHandler(msg: LowState_):
    """接收机器人状态，读取当前关节角度"""
    global current_joint_pos
    for i in range(12):
        current_joint_pos[i] = msg.motor_state[i].q

if __name__ == '__main__':
    print("=" * 60)
    print("Go2 站立控制 - 调试版本")
    print("=" * 60)
    
    # macOS修改：使用 lo0
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo0")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    # 创建订阅者，接收机器人状态
    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(LowStateHandler, 10)
    
    # 创建发布者，发送控制指令
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # macOS修改：快速接收初始状态（不发送零命令，避免"休眠"机器人）
    print("\n等待1秒，接收初始状态...", flush=True)
    for i in range(500):  # 等待1秒（500 * 0.002 = 1秒）
        # macOS修改：在等待期间也发布命令以激活DDS通信
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(0.002)
        if np.any(current_joint_pos != 0):
            print(f"  状态已接收（{i*0.002:.2f}秒）", flush=True)
            break
    
    print("\n初始关节角度（弧度）：")
    print("  关节 0-11:", current_joint_pos)
    
    if np.all(current_joint_pos == 0):
        print("\n⚠️  警告：状态订阅可能失败，所有关节角度都是0！")
        print("  请检查仿真器是否正常运行...")
        print("  将使用预设的趴下姿态作为起点...")
        initial_pos = stand_down_joint_pos.copy()
        real_down_pos = stand_down_joint_pos.copy()  # 备用趴下姿态
    else:
        # macOS修改：使用实际接收到的初始位置作为起点
        print("\n✅ 成功读取初始位置，将从当前位置过渡到站立姿态")
        initial_pos = current_joint_pos.copy()
        # 真正的趴下姿态 = 初始位置（机器人启动时就是趴着的）
        real_down_pos = initial_pos.copy()
    
    print("\n目标站立姿态：")
    print("  关节 0-11:", stand_up_joint_pos)
    print("\n目标趴下姿态（真正的初始位置）：")
    print("  关节 0-11:", real_down_pos)
    print("\n" + "=" * 60)
    print("开始控制！（按 Ctrl+C 停止）")
    print("=" * 60 + "\n")

    # 可调参数 - 优化：中等力量 + 高阻尼 = 平稳稳定！
    kp_stand_start = 30.0  # 站立起始刚度（30，温和启动）
    kp_stand_end = 80.0    # 站立最终刚度（80，适中驱动）
    kd_stand = 10.0        # 站立阻尼（10，高阻尼抑制振荡）
    
    kp_down = 150.0        # 趴下刚度（降到150，避免过激反应）
    kd_down = 15.0         # 趴下阻尼（15，高阻尼平稳弯曲）
    
    # 测试循环设置 - 延长时间让机器人慢慢站起！
    CONTROL_FREQ = 500  # 控制频率500Hz
    STAND_DURATION = 6.0    # 站立6秒（更长时间平稳站起）
    HOLD_STAND_DURATION = 4.0  # 保持站立4秒（充分稳定）
    DOWN_DURATION = 5.0     # 趴下5秒（慢慢弯曲）
    MAX_CYCLES = 1          # 只做1次（站立→保持→趴下）
    HOLD_TIME = 2.0         # 保持2秒观察稳定性
    
    # 保存站立保持阶段结束时的实际位置（用于趴下阶段的起点）
    stand_end_pos = np.zeros(12)
    
    total_duration = (STAND_DURATION + HOLD_STAND_DURATION + DOWN_DURATION) * MAX_CYCLES
    dt = 1.0 / CONTROL_FREQ  # macOS修改：根据控制频率计算时间步长
    print(f"测试时长：{STAND_DURATION}s站立 + {HOLD_STAND_DURATION}s保持 + {DOWN_DURATION}s趴下 + {HOLD_TIME}s观察 = 总计{total_duration+HOLD_TIME}秒")
    print(f"控制频率：{CONTROL_FREQ}Hz（降低频率减少通信延迟影响）\n", flush=True)

    last_print_time = 0.0
    cycle_count = 0
    running_time = 0.0  # macOS修改：初始化running_time

    try:
        while running_time < total_duration:
            step_start = time.perf_counter()
            running_time += dt

            # 计算当前循环和阶段（包含保持站立阶段）
            cycle_time = running_time % (STAND_DURATION + HOLD_STAND_DURATION + DOWN_DURATION)
            cycle_count = int(running_time // (STAND_DURATION + HOLD_STAND_DURATION + DOWN_DURATION)) + 1
            
            # macOS修改：每0.5秒打印一次状态
            if running_time - last_print_time >= 0.5:
                if cycle_time < STAND_DURATION:
                    phase_info = f"循环{cycle_count} - 站立中"
                elif cycle_time < STAND_DURATION + HOLD_STAND_DURATION:
                    phase_info = f"循环{cycle_count} - 保持站立"
                else:
                    phase_info = f"循环{cycle_count} - 趴下中"
                
                # macOS修改：显示前4个关节的详细数据（髋、腿）+ 实时输出
                print(f"\n[{running_time:.1f}s] {phase_info}")
                print("  关节0-3 实际: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]".format(*current_joint_pos[0:4]))
                print("  关节0-3 目标: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]".format(
                    cmd.motor_cmd[0].q, cmd.motor_cmd[1].q, cmd.motor_cmd[2].q, cmd.motor_cmd[3].q))
                print("  关节0-3 误差: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]".format(
                    current_joint_pos[0]-cmd.motor_cmd[0].q,
                    current_joint_pos[1]-cmd.motor_cmd[1].q,
                    current_joint_pos[2]-cmd.motor_cmd[2].q,
                    current_joint_pos[3]-cmd.motor_cmd[3].q))
                print("  当前Kp: {:.1f}, Kd: {:.1f}".format(cmd.motor_cmd[0].kp, cmd.motor_cmd[0].kd), flush=True)
                last_print_time = running_time

            # 三阶段控制：站立 → 保持站立 → 趴下
            if DOWN_DURATION == 0 or cycle_time < STAND_DURATION:
                # 阶段1：站立阶段 - 中等力量+高阻尼平稳站起（Kp 30→80, Kd=10）
                phase = np.tanh(cycle_time / 2.0)  # 2.0秒平滑过渡（更慢更稳）
                for i in range(12):
                    # 从stand_down_joint_pos插值到站立姿态
                    cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (1 - phase) * stand_down_joint_pos[i]
                    # Kp从30逐渐增加到80
                    cmd.motor_cmd[i].kp = phase * kp_stand_end + (1 - phase) * kp_stand_start
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kd_stand  # 固定10.0（高阻尼抑制振荡）
                    cmd.motor_cmd[i].tau = 0.0
            
            elif cycle_time < STAND_DURATION + HOLD_STAND_DURATION:
                # 阶段2：保持站立阶段 - 中等力量+高阻尼让机器人平稳稳定
                for i in range(12):
                    cmd.motor_cmd[i].q = stand_up_joint_pos[i]  # 保持站立姿态
                    cmd.motor_cmd[i].kp = kp_stand_end  # 固定80（适中力量）
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kd_stand  # 固定10.0（高阻尼抑制振荡）
                    cmd.motor_cmd[i].tau = 0.0
                
                # 在保持阶段结束时保存实际位置（作为趴下阶段的起点）
                if cycle_time >= STAND_DURATION + HOLD_STAND_DURATION - dt:
                    stand_end_pos = current_joint_pos.copy()
                    print(f"\n🔵 保存站立结束位置：{stand_end_pos[0:4]}", flush=True)
                    print(f"    目标站立位置：{stand_up_joint_pos[0:4]}", flush=True)
                    print(f"    误差：{stand_end_pos[0:4] - stand_up_joint_pos[0:4]}", flush=True)
            
            else:
                # 阶段3：趴下阶段 - 较高力量+高阻尼平稳弯曲（Kp=150, Kd=15）
                phase = np.tanh((cycle_time - STAND_DURATION - HOLD_STAND_DURATION) / 1.5)  # 1.5秒平滑过渡
                
                # 调试：第一次进入趴下阶段时打印
                if cycle_time >= STAND_DURATION + HOLD_STAND_DURATION and cycle_time < STAND_DURATION + HOLD_STAND_DURATION + dt:
                    print(f"\n🔴 进入趴下阶段！", flush=True)
                    print(f"  stand_end_pos[0:4]: {stand_end_pos[0:4]}", flush=True)
                    print(f"  real_down_pos[0:4]: {real_down_pos[0:4]}", flush=True)
                    print(f"  Kp_down: {kp_down}, Kd_down: {kd_down}", flush=True)
                
                for i in range(12):
                    # 从保持阶段结束时的实际位置插值到初始趴下姿态
                    cmd.motor_cmd[i].q = phase * real_down_pos[i] + (1 - phase) * stand_end_pos[i]
                    cmd.motor_cmd[i].kp = kp_down  # 固定150（较高力量）
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kd_down  # 固定15（高阻尼抑制振荡）
                    cmd.motor_cmd[i].tau = 0.0

            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            time_until_next_step = dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # macOS修改：保持趴下命令持续发送，让机器人完全达到趴下姿态
        print("\n" + "=" * 60)
        if DOWN_DURATION == 0:
            print(f"站立完成！保持站立姿态 {HOLD_TIME} 秒（观察稳定性）...")
        else:
            print(f"主循环完成！保持趴下命令 {HOLD_TIME} 秒（让机器人完全趴下）...")
        print("=" * 60)
        print("⏱️  观察仿真器中机器人的行为！", flush=True)
        
        # macOS修改：保持站立或真正的趴下姿态
        target_pos = stand_up_joint_pos if DOWN_DURATION == 0 else real_down_pos
        target_kp = kp_stand_end if DOWN_DURATION == 0 else kp_down
        target_kd = kd_stand if DOWN_DURATION == 0 else kd_down
        
        for i in range(12):
            cmd.motor_cmd[i].q = target_pos[i]
            cmd.motor_cmd[i].kp = target_kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = target_kd
            cmd.motor_cmd[i].tau = 0.0
        
        hold_start = time.perf_counter()
        last_hold_print = 0.0
        
        while (time.perf_counter() - hold_start) < HOLD_TIME:
            # 继续发送站立命令
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)
            
            # macOS修改：每0.5秒打印一次状态 + 实时输出
            elapsed = time.perf_counter() - hold_start
            if elapsed - last_hold_print >= 0.5:
                print(f"\n[保持+{elapsed:.1f}s]")
                print("  关节0-3 实际: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]".format(*current_joint_pos[0:4]))
                target_label = "站立" if DOWN_DURATION == 0 else "趴下(初始)"
                print("  关节0-3 目标({}): [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]".format(
                    target_label, target_pos[0], target_pos[1], target_pos[2], target_pos[3]))
                print("  关节0-3 误差: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]".format(
                    current_joint_pos[0]-target_pos[0],
                    current_joint_pos[1]-target_pos[1],
                    current_joint_pos[2]-target_pos[2],
                    current_joint_pos[3]-target_pos[3]), flush=True)
                last_hold_print = elapsed
            
            time.sleep(dt)
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        print("初始关节角度：", initial_pos)
        print("最终关节角度：", current_joint_pos)
        if DOWN_DURATION == 0:
            print("目标站立姿态：", stand_up_joint_pos)
            print("误差（最终-目标）：", current_joint_pos - stand_up_joint_pos)
        else:
            print("目标趴下姿态（初始位置）：", real_down_pos)
            print("误差（最终-目标）：", current_joint_pos - real_down_pos)
                
    except KeyboardInterrupt:
        print("\n\n程序已手动停止", flush=True)
        print("最终关节角度：", current_joint_pos, flush=True)
    
    finally:
        # macOS修改：清理DDS资源，防止消息队列积压
        print("\n正在清理DDS资源...", flush=True)
        
        # macOS修改：退出时保持站立或真正的趴下姿态
        exit_pos = stand_up_joint_pos if DOWN_DURATION == 0 else real_down_pos
        # 清理时用适中的Kp，避免剧烈动作但保持姿态
        exit_kp = kp_stand_end if DOWN_DURATION == 0 else 20.0
        exit_kd = kd_stand if DOWN_DURATION == 0 else 5.0
        
        for i in range(12):
            cmd.motor_cmd[i].q = exit_pos[i]
            cmd.motor_cmd[i].kp = exit_kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = exit_kd
            cmd.motor_cmd[i].tau = 0.0
        for i in range(12, 20):  # 其他关节清零
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        
        # 发送100次命令确保稳定
        for _ in range(100):
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)
            time.sleep(0.002)
        
        print("DDS资源已清理，程序退出。\n", flush=True)

