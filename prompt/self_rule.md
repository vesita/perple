# AI 协作规则系统 - 快速指南

> 本文档是规则系统的快速指南，用于指导 AI 和开发者正确解析和应用规则。

## 📖 目录

- [核心机制](#核心机制)
- [执行流程](#执行流程)
- [扩展指南](#扩展指南)
- [维护指南](#维护指南)

---

## 核心机制

### 1. 优先级机制 ⭐

**规则**: `rule_order` 数字越小，优先级越高

```
基础框架 (必须):
├─ rule_order = 1 → 语言偏好 [P0, mandatory:true]
├─ rule_order = 2 → 工具链配置 [P1, mandatory:true]
└─ rule_order = 3 → 格式偏好 [P2, mandatory:false]

扩展区域 (可选):
├─ rule_order = 4 → (用户自定义)
├─ rule_order = 5 → (用户自定义)
└─ ... → (未来扩展)
```

**决策树**:

```
遇到冲突？
├─ 检查 rule_order → 小的优先 (1 > 2 > 3 > ... > N)
└─ rule_order 相同？→ mandatory:true 优先于 false
```

### 2. 强制性分级

| 级别 | 含义 | 示例 |
|------|------|------|
| `mandatory: true` | 必须遵守 | 语言选择、基础工具链 |
| `mandatory: false` | 建议遵守 (允许例外) | 命名约定、文件格式、流程规范 |

### 3. 依赖关系

```yaml
# 示例：某规则依赖语言偏好
rule_index:
  id: "new_rule_001"
  depends_on: ["lang_001"]
```

### 4. 作用域控制

```yaml
applies_to: ["*.rs", "*.py", "*.md", ...]
excludes: ["vendor/", "target/", "**/test_data/**"]
```

---

## 执行流程

### AI 应用规则的步骤

```
1. 读取 rule.yaml
   ↓
2. 检查文件是否在 excludes 中 → 是：跳过所有规则
   ↓
3. 检查文件是否匹配 applies_to → 否：跳过
   ↓
4. 按 rule_order 顺序加载子规则 (从小到大：1→2→3→...→N)
   ↓
5. 解析依赖关系 (depends_on)
   ↓
6. 建立优先级索引
   ↓
7. 应用规则时检查冲突解决机制
   ↓
8. 根据 mandatory 级别决定执行严格程度
```

### 冲突解决速查表

| 场景 | 解决方案 |
|------|----------|
| 语言偏好 vs 工具链默认 | 服从语言偏好 (rule_order:1 > 2) |
| 工具链 vs 格式偏好 | 服从工具链 (rule_order:2 > 3) |
| 全局配置 vs 局部配置 | 全局优先 (`scope_precedence: "global_first"`) |
| mandatory:true vs false | true 优先 |
| **新增规则 vs 现有规则** | **按 rule_order 判断 (支持任意数量扩展)** |

---

## 扩展指南

### 🧩 模块化设计理念

规则系统采用**插件式架构**：

- ✅ 基础框架稳定 (P0-P2)
- ✅ 扩展规则即插即用 (P3+)
- ✅ 新规则无需修改旧规则
- ✅ 只需在主配置中注册即可

### 添加新规则的步骤

```yaml
# 步骤 1: 创建规则文件 (prompt/rules/<your_rule>.yaml)
# ────────────────────────────────────────
rule_index:
  id: "unique_id_001"               # 唯一标识符
  category: "your_category"         # 类别名称
  mandatory: false                  # 强制级别
  depends_on: ["lang_001"]          # 依赖关系 (可选)

# 你的规则内容...
your_rule_content:
  key: value

# 步骤 2: 在主配置中注册 (prompt/rule.yaml)
# ────────────────────────────────────────
sub_rules:
  index:
    # ... existing rules ...
    
    # ────────────────────────────────────────
    # 新增规则 (当前最大 rule_order + 1)
    # ────────────────────────────────────────
    - rule_order: 4                 # 下一个可用序号
      file: "<your_rule>.yaml"
      name: "规则名称"
      mandatory: false

# 步骤 3: 更新元数据
# ────────────────────────────────────────
meta:
  version: "1.1.0"                  # 升级版本号

priority_table:
  P3: "规则名称 (rule_order:4, mandatory:false)"

# 步骤 4: 完成！新规则立即生效
```

### 推荐的优先级分配策略

```
基础框架 (核心规范):
├─ rule_order: 1  → 语言偏好 (沟通基础)
├─ rule_order: 2  → 工具链配置 (开发环境)
└─ rule_order: 3  → 格式偏好 (代码风格)

扩展区域 (流程/专项规范):
├─ rule_order: 4  → Git 提交、代码审查、测试规范...
├─ rule_order: 5  → 文档规范、安全规范、部署规范...
└─ rule_order: N  → 其他专项规范...
```

### 扩展示例模板

```yaml
# prompt/rules/your_new_rule.yaml
# ============================================================================
# 规则名称 - 简短描述
# ============================================================================
# 优先级：PX (rule_order: N) | mandatory: false
# 依赖：lang_001 (如有需要)
# ============================================================================

rule_index:
  id: "new_001"
  category: "your_category"
  mandatory: false
  depends_on: ["lang_001"]

# 规则内容...
basic_settings:
  key: value

examples:
  valid:
    - "示例 1"
    - "示例 2"
  invalid:
    - "错误示例 1"
    - "错误示例 2"

# 💡 应用提示
# ────────────────────────────────────────
# 使用说明和注意事项...
```

---

## 维护指南

### 添加新规则

```
1. 在 prompt/rules/ 创建新 YAML 文件
2. 定义唯一 rule_index.id
3. 设置 mandatory 和 depends_on
4. 在 rule.yaml 的 sub_rules.index 中添加引用
5. 分配 rule_order (当前最大 +1，或插入到合适位置)
6. 更新 priority_table 和 meta.version
```

### 修改现有规则

- ✅ 保持向后兼容，重大改动升级 `meta.version`
- ✅ 评估影响范围，必要时调整 `rule_order`
- ✅ 更新本文档并通知审查

### 调试冲突

```yaml
# 步骤 1: 识别冲突条款
冲突：规则 A vs 规则 B

# 步骤 2: 检查 rule_order
规则 A: rule_order = X
规则 B: rule_order = Y

# 步骤 3: 应用高优先级规则
if X < Y:
  结果：采用规则 A
else:
  结果：采用规则 B

# 通用逻辑：min(rule_order) 获胜
```

---

## 快速参考

### 规则文件结构

```yaml
rule_index:
  id: "唯一标识"
  category: "类别"
  mandatory: true/false
  depends_on: ["依赖的 rule_id"]

# 具体规则条款
category_name:
  key: value
```

### 常用命令

```bash
# Rust
cargo build          # 构建
cargo test --all     # 测试
cargo fmt --all      # 格式化
cargo clippy -- -D warnings  # 检查

# Python
uv run python        # 运行
uv pip install       # 安装依赖
uv sync             # 同步环境
black .             # 格式化

# Git
git commit -m "type(scope): description"
```

---

## 最佳实践

**推荐** ✅:

- 规则文件小而专注
- 清晰注释每个规则段
- 定期审查规则执行情况
- 保持可读性
- 为新规则预留扩展空间
- 渐进式扩展 (一次添加一个规则)

**避免** ❌:

- 混合无关规则
- 过度复杂表述
- 过度依赖工具
- 频繁更改核心规则
- 一次性添加过多规则级别

---

## 设计哲学

本规则系统采用**开放式模块化设计**：

### 核心特点

- ✅ **稳定的基础框架** (P0-P2，核心规范)
- ✅ **灵活的扩展机制** (P3+，按需添加)
- ✅ **即插即用的模块** (新规则无需修改旧规则)
- ✅ **一致的决策逻辑** (无论多少个规则)

### 扩展原则

```
1. 创建新规则文件
2. 在主配置中注册
3. 分配优先级序号
4. 立即可用，无需改动其他规则
```

**核心理念**: 简单但不简陋，灵活但有原则。  
**架构模式**: 核心稳定，边缘灵活；基础固定，扩展自由。

---

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2026-03-27 | 初始版本：基础框架 (语言、工具链、格式) |
| 1.1.0+ | 2026-03-27 | 扩展版本：支持用户自定义规则模块 |

---

**版本**: 1.1.0 (2026-03-27) | **维护者**: vesita
