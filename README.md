# Agent Trajectory Event Sequence Visualization

数据来源：mind2web，训练集共计1009个task，从中提取出agent trajectory

每个task完成的action格式如下：

```
action_reprs:
	0. [svg]   -> CLICK
	1. [svg]   -> CLICK
	2. [searchbox]  Find a location -> TYPE: NAPA VALLEY
	3. [span]  Napa Valley -> CLICK
	4. [combobox]  Reservation type -> SELECT: Wineries
	5. [svg]   -> CLICK
	6. [svg]   -> CLICK
	7. [button]  15 -> CLICK
	8. [combobox]  Time -> SELECT: 10:00 AM
	9. [combobox]  Party size -> SELECT: 4 guests
	10. [svg]   -> CLICK
	11. [button]  Edit cuisine type filter -> CLICK
	12. [checkbox]  Mediterranean -> CLICK
	13. [button]  Submit -> CLICK
	14. [button]  Open additional search filters -> CLICK
	15. [checkbox]  Outdoors -> CLICK
	16. [checkbox]  Wine tasting -> CLICK
	17. [button]  Update search -> CLICK
	18. [span]  10:00 AM -> CLICK
```

将action_reprs序列变为对应csv格式。

详细代码见visual_traj.ipynb

```python
import pandas as pd
from datetime import datetime, timedelta
import os
import glob

# 获取data/train目录下的所有json文件
train_dir = "./data/train/"
json_files = glob.glob(os.path.join(train_dir, "train_*.json"))
json_files.sort()  # 排序以确保顺序

print(f"找到 {len(json_files)} 个JSON文件:")
for f in json_files:
    print(f"  - {os.path.basename(f)}")
print("\n" + "="*60)

# 遍历每个json文件
for json_file in json_files:
    filename = os.path.basename(json_file)
    file_prefix = filename.replace('.json', '')
    
    print(f"\n正在处理: {filename}")
    
    # 读取当前json文件
    with open(json_file, "r") as f:
        all_samples = json.load(f)
    
    print(f"  包含 {len(all_samples)} 个 samples")
    
    # 创建空列表来存储所有行
    rows = []
    
    # 遍历每个 sample
    for sample_idx, sample in enumerate(all_samples):
        sequence_id = f"seq_{sample_idx + 1}"
        action_reprs = sample.get("action_reprs", [])
        
        # 为每个 action 创建一行数据
        for action_idx, action in enumerate(action_reprs):
            # 生成时间戳（从某个起点开始，每个action递增1秒）
            base_time = datetime(2024, 1, 1) + timedelta(days=sample_idx)
            timestamp = base_time + timedelta(seconds=action_idx + 1)
            
            # 解析 action_repr，提取 event_type
            parts = action.split('-') if '-' in action else action.split()
            
            if len(parts) >= 1:
                event_type = parts[-1]  # 取最后一个部分作为event_type
                details = action  # 完整的 action_repr 作为 details
            else:
                event_type = action
                details = action
            
            # 添加到行列表
            rows.append({
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'event_type': event_type,
                'details': action,
                'sequence_id': sequence_id
            })
    
    # 转换为 DataFrame
    df = pd.DataFrame(rows)
    
    # 保存为 CSV 文件
    output_csv = f"{file_prefix}_formatted.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"  ✓ 已保存: {output_csv}")
    print(f"  ✓ 共 {len(df)} 行数据，来自 {len(all_samples)} 个序列")

print("\n" + "="*60)
print("✓ 所有文件转换完成！")
print(f"\n生成的CSV文件:")
csv_files = glob.glob("train_*_formatted.csv")
csv_files.sort()
for csv_file in csv_files:
    size = os.path.getsize(csv_file) / 1024  # KB
    print(f"  - {csv_file} ({size:.1f} KB)")

```

## mind2web数据格式

### 原始数据

原始的数据是真实网页的快照，包含：trace.zip，session.har.zip，{action_id}_before/after.mhtm，dom_content.json，screenshot.json，storage.json

```
raw_dump
├── session
│   ├── {session_id}
│   │   ├── session.har.zip
│   │   └── videos
│   │       ├── {page_id}.webm
│   │       └── ...
|   └── ...
├── task
│   ├── {annotation_id}
│   │   ├── processed
│   │   │   ├── dom_content.json
│   │   │   ├── screenshot.json
│   │   │   └── snapshots
│   │   │       ├── {action_id}_before.mhtml
│   │   │       ├── {action_id}_after.mhtml
│   │   │       └── ...
│   │   ├── storage.json
│   │   └── trace.zip
│   └── ...
└── task_meta.json
```

### 训练集json文件

提取后的json文件：

- "annotation_id" (str): unique id for each task
- "website" (str): website name
- "domain" (str): website domain
- "subdomain" (str): website subdomain
- "confirmed_task" (str): task description
- "action_reprs" (list[str]): human readable string representation of the action sequence
- "actions" (list[dict]): list of actions (steps) to complete the task
  - "action_uid" (str): unique id for each action (step)
  - "raw_html" (str): raw html of the page before the action is performed
  - "cleaned_html" (str): cleaned html of the page before the action is performed
  - "operation" (dict): operation to perform
    - "op" (str): operation type, one of CLICK, TYPE, SELECT
    - "original_op" (str): original operation type, contain additional HOVER and ENTER that are mapped to CLICK, not used
    - "value" (str): optional value for the operation, e.g., text to type, option to select
  - "pos_candidates" (list[dict]): ground truth elements. Here we only include positive elements that exist in "cleaned_html" after our preprocessing, so "pos_candidates" might be empty. The original labeled element can always be found in the "raw_html".
    - "tag" (str): tag of the element
    - "is_original_target" (bool): whether the element is the original target labeled by the annotator
    - "is_top_level_target" (bool): whether the element is a top level target find by our algorithm. please see the paper for more details.
    - "backend_node_id" (str): unique id for the element
    - "attributes" (str): serialized attributes of the element, use `json.loads` to convert back to dict
  - "neg_candidates" (list[dict]): other candidate elements in the page after preprocessing, has similar structure as "pos_candidates"

每个action_reprs对应一个actions，正确的element标记为pos_candidates，错误的标记为neg_candidates。正样本数量较少个位数，负样本为网页中的其余元素几百个。

·





