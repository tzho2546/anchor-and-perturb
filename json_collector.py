import json
import os
from typing import Optional

from datetime import datetime



class JSONCollector:
    def __init__(self, model_name: str = "qmix"):
        now = datetime.now()
        time_str = now.strftime("%Y%m%d-%H%M%S")
        self.save_path = f"{model_name}-{time_str}.json"
        self.logs = []

        # 确保保存目录存在
        os.makedirs(os.path.dirname(os.path.abspath(model_name)), exist_ok=True)
        # 初始化清空旧文件
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write("[\n")
        self.first_entry = True

    def collect(self,
                episode: int,
                total_episodes: int,
                status: str,
                avg_train: float,
                eval_score: float,
                win_rate: float,
                epsilon: float,
                is_pulse: bool,
                extra_metadata: Optional[dict] = None):
        """
        同步打印终端 log，并写入结构化 JSON 数据
        """

        entry = {
            "episode": episode + 1,
            "total_episodes": total_episodes,
            "status": status.strip(),
            "train_return": round(float(avg_train), 4),
            "eval_reward": round(float(eval_score), 4),
            "eval_win_rate": round(float(win_rate), 4),
            "epsilon": round(float(epsilon), 4),
            "is_pulse": is_pulse
        }

        if extra_metadata:
            entry.update(extra_metadata)

        self.logs.append(entry)

        # 实时追加写入，防止程序意外崩溃丢数据
        with open(self.save_path, "a", encoding="utf-8") as f:
            if not self.first_entry:
                f.write(",\n")
            json.dump(entry, f, ensure_ascii=False)
            self.first_entry = False

    def close(self):
        """训练彻底结束时调用，封死合法 JSON 尾部"""
        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write("\n]")