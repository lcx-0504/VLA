# macOS 适配修改说明

本文档记录了在 macOS (Apple Silicon) 上运行 unitree_mujoco 所做的所有修改。

## 修改概览

1. **配置文件修改** - `simulate_python/config.py`
2. **SDK 兼容性修复** - `dependencies/unitree_sdk2_python/unitree_sdk2py/utils/`
3. **运行方式变更** - 使用 `mjpython` 替代 `python3`

---

## 1. 配置文件修改

### 文件：`simulate_python/config.py`

**原始文件备份**：`simulate_python/config.py.backup`

#### 修改 1：网络接口名称

```python
# 原始值 (Linux)
INTERFACE = "lo"

# 修改为 (macOS)
INTERFACE = "lo0"
```

**原因**：macOS 的本地回环接口名称是 `lo0`，而 Linux 是 `lo`。

#### 修改 2：禁用游戏手柄

```python
# 原始值
USE_JOYSTICK = 1

# 修改为
USE_JOYSTICK = 0
```

**原因**：如果没有连接游戏手柄，保持默认值 1 会导致程序报错。

---

## 2. SDK 兼容性修复

### 文件 1：`dependencies/unitree_sdk2_python/unitree_sdk2py/utils/timerfd.py`

**原始文件备份**：`dependencies/unitree_sdk2_python/unitree_sdk2py/utils/timerfd.py.backup`

#### 问题
`timerfd_create`、`timerfd_settime`、`timerfd_gettime` 是 Linux 特有的系统调用，macOS 不支持。

#### 解决方案
添加平台检测，在 macOS 上不加载这些函数：

```python
import platform

# timerfd is Linux-specific, provide dummy functions for macOS
if platform.system() == 'Darwin':  # macOS
    # Dummy functions for macOS compatibility
    timerfd_create = None
    timerfd_settime = None
    timerfd_gettime = None
else:
    # function timerfd_create
    timerfd_create = CLIBLookup("timerfd_create", ctypes.c_int, (ctypes.c_long, ctypes.c_int))
    timerfd_settime = CLIBLookup("timerfd_settime", ctypes.c_int, (...))
    timerfd_gettime = CLIBLookup("timerfd_gettime", ctypes.c_int, (...))
```

### 文件 2：`dependencies/unitree_sdk2_python/unitree_sdk2py/utils/thread.py`

**原始文件备份**：`dependencies/unitree_sdk2_python/unitree_sdk2py/utils/thread.py.backup`

#### 问题 1
`RecurrentThread` 在 macOS 上尝试使用不存在的 `timerfd_create`。

#### 解决方案 1
修改初始化逻辑，当 `timerfd_create` 不可用时使用备用实现：

```python
# 原始代码
if interval is None or interval <= 0.0:
    super().__init__(target=self.__LoopFunc_0, name=name)
else:
    super().__init__(target=self.__LoopFunc, name=name)

# 修改为
if interval is None or interval <= 0.0 or timerfd_create is None:
    super().__init__(target=self.__LoopFunc_0, name=name)
else:
    super().__init__(target=self.__LoopFunc, name=name)
```

#### 问题 2
`__LoopFunc_0` 方法中使用了错误的变量名（**这是官方代码的 Bug**）。

**原因分析**：
- 在 `__init__` 中定义的是 `self.__loopArgs` 和 `self.__loopKwargs`
- `__LoopFunc` (timerfd 版本) 正确使用了这两个变量
- `__LoopFunc_0` (备用版本) 却错误地使用了不存在的 `self.__args` 和 `self.__kwargs`
- 在 Linux 上由于很少触发 `__LoopFunc_0`，这个 bug 没被发现
- 在 macOS 上由于强制使用 `__LoopFunc_0`，bug 立即暴露

#### 解决方案 2
修正变量名：

```python
# 原始代码（Bug）
self.__loopTarget(*self.__args, **self.__kwargs)

# 修改为（正确）
self.__loopTarget(*self.__loopArgs, **self.__loopKwargs)
```

---

## 3. 运行方式变更

### 官方文档（Linux）

```bash
cd ./simulate_python
python3 ./unitree_mujoco.py
```

### macOS 正确方式

```bash
cd ./simulate_python
mjpython ./unitree_mujoco.py
```

**原因**：
- macOS 上 MuJoCo 的图形界面 (`mujoco.viewer.launch_passive`) 需要在 `mjpython` 下运行
- `mjpython` 是 MuJoCo Python 包自带的特殊 Python 解释器
- 在安装 `mujoco` 包时会自动安装 `mjpython`

**验证 mjpython 安装**：
```bash
which mjpython
# 输出：/opt/anaconda3/envs/unitree_mujoco/bin/mjpython
```

---

## 4. 完整启动流程（macOS）

### 环境准备

```bash
# 创建 conda 环境
conda create -n unitree_mujoco python=3.10 -y
conda activate unitree_mujoco

# 安装依赖
cd dependencies/unitree_sdk2_python
pip install -e .

cd ../..
pip install mujoco
pip install pygame  # 可选，如果需要手柄支持
```

### 启动仿真器

```bash
# 确保在 conda 环境中
conda activate unitree_mujoco

# 进入仿真器目录
cd simulate_python

# 使用 mjpython 启动（重要！）
mjpython ./unitree_mujoco.py
```

### 运行测试程序（新终端）

```bash
# 激活环境
conda activate unitree_mujoco

# 运行测试
cd simulate_python
python3 ./test/test_unitree_sdk2.py
```

### 运行站立示例（新终端）

```bash
# 激活环境
conda activate unitree_mujoco

# 让机器狗站起来
cd example/python
python3 ./stand_go2.py
```

---

## 5. 恢复原始文件

如果需要恢复到原始状态：

```bash
# 恢复配置文件
cd simulate_python
cp config.py.backup config.py

# 恢复 SDK 文件
cd ../dependencies/unitree_sdk2_python/unitree_sdk2py/utils
cp timerfd.py.backup timerfd.py
cp thread.py.backup thread.py
```

---

## 6. 修改文件清单

| 文件路径                                                           | 修改类型   | 备份位置                                                                  |
| ------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------- |
| `simulate_python/config.py`                                        | 配置调整   | `simulate_python/config.py.backup`                                        |
| `simulate_python/test/test_unitree_sdk2.py`                        | 配置调整   | `simulate_python/test/test_unitree_sdk2.py.backup`                        |
| `example/python/stand_go2.py`                                      | 配置调整   | `example/python/stand_go2.py.backup`                                      |
| `dependencies/unitree_sdk2_python/unitree_sdk2py/utils/timerfd.py` | 兼容性修复 | `dependencies/unitree_sdk2_python/unitree_sdk2py/utils/timerfd.py.backup` |
| `dependencies/unitree_sdk2_python/unitree_sdk2py/utils/thread.py`  | 兼容性修复 | `dependencies/unitree_sdk2_python/unitree_sdk2py/utils/thread.py.backup`  |

---

## 7. 注意事项

1. **网络接口**：macOS 使用 `lo0`，Linux 使用 `lo`
2. **启动命令**：macOS 必须使用 `mjpython`，Linux 使用 `python3`
3. **游戏手柄**：没有手柄时必须设置 `USE_JOYSTICK = 0`
4. **性能**：M 系列芯片性能充足，无需担心
5. **兼容性修复**：修改的 SDK 文件在更新 `unitree_sdk2_python` 时会被覆盖，需要重新应用补丁

---

## 8. 已测试环境

- **操作系统**：macOS (Apple Silicon M4)
- **Python 版本**：3.10.18
- **MuJoCo 版本**：3.3.7
- **unitree_sdk2_python**：1.0.1
- **cyclonedds**：0.10.2

---

*最后更新：2025-10-17*

