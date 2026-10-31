# 学習ランの再現情報

各サブディレクトリは 1 回の学習ランに対応し、学習時に自動保存されたスナップショットを収める。

| ディレクトリ | タスク | 学習日 | 結果 |
|---|---|---|---|
| `stairs_v1/` | `Unitree-Go2-Velocity-Stairs` | 2026-08-04 | 10,000 iter / 7.6h、reward 22.36、段差 11cm まで転倒 ≤3% |
| `stairs_v2/` | `Unitree-Go2-Velocity-StairsV2` | 2026-08-05 | 10,000 iter / 7.7h、reward 29.28、段差 13cm まで転倒 0%、15cm 15.8% |
| `obstacle/` | `Unitree-Go2-Velocity-Obstacle` | 2026-08-12 | 10,000 iter / 7.8h、reward 28.77、障害物衝突 0%、階段 13cm 3.9% |
| `obstacle_hm/` | `Unitree-Go2-Velocity-ObstacleHM` | 2026-08-28〜31 | 10,000 iter(3,908 で中断→resume)、reward 27.95、障害物 0/400、長階段スタール解消。階段 15cm+ は劣化(登り厳密 53%) |
| `obstacle_hmdiverse/` | `Unitree-Go2-Velocity-ObstacleHMDiverse` | 2026-08-31 | 10,000 iter、reward 26.84。壁クラスからの負の転移で長階段スタール再発(対照実験として保存) |
| `obstacle_ext/` | `Unitree-Go2-Velocity-ObstacleExt` | 2026-09-01 | 10,000 iter / 7.5h、reward 28.7。**採用版(回避+階段v3)**: 障害物 0/400、階段 厳密到達 登り 17cm 83%・19cm 63%(全ポリシー最良)、長階段クリア |

各ディレクトリの内容:

- `RUN_DIR` — `logs/rsl_rl/` 以下の元のランディレクトリ名(チェックポイント `model_9999.pt` と TensorBoard イベントはそこにある。`logs/` は git 管理外)
- `env_cfg.yaml` / `agent_cfg.yaml` — 学習時に解決された環境・PPO 設定の全量(再現の正)
- `unitree_rl_lab.diff` — 学習時の unitree_rl_lab フォーク上での upstream コミット(`2522f89`)からの差分(歴史的記録)。
  **rsl_rl の自動保存は untracked ファイルを含まない**ため、新規ファイル(`velocity_stairs_env_cfg.py`、`go2_obstacle/`、`scripts/eval_*.py`)はこの diff に現れない。
  学習は当時 unitree_rl_lab フォーク内で実施しており、その追加ファイル群を**本リポジトリ(go2-rl-stairs-avoid)に移設**した。
  移設はパッケージパスの書き換えのみでロジックは同一(学習時スナップショット `params/*_env_cfg.py` との一致は 2026-08-28 / 2026-09-02 に確認済み)

## 再現手順

1. 環境(下記)を揃え、本リポジトリを README.md の手順でインストールする(タスク実装は移設後も学習時と同一パラメータ)
2. 学習:
   ```bash
   cd <this repo>
   ~/work/IsaacLab/isaaclab.sh -p src/train/train.py --task Unitree-Go2-Velocity-StairsV2 --viz none --max_iterations 10000
   ~/work/IsaacLab/isaaclab.sh -p src/train/train.py --task Unitree-Go2-Velocity-Obstacle --viz none --max_iterations 10000
   ~/work/IsaacLab/isaaclab.sh -p src/train/train.py --task Unitree-Go2-Velocity-ObstacleHM --viz none --max_iterations 10000
   ```
   `obstacle_ext`(回避+階段v3)は階段特化 v2 からのウォームスタートが前提:
   ```bash
   ~/work/IsaacLab/isaaclab.sh -p src/train/ext_warmstart_surgery.py \
       --spec logs/rsl_rl/unitree_go2_velocity_stairsv2/<stairs_v2 run>/model_9999.pt \
       --out logs/rsl_rl/unitree_go2_velocity_obstacleext/warmstart/model_0.pt
   ~/work/IsaacLab/isaaclab.sh -p src/train/train.py --task Unitree-Go2-Velocity-ObstacleExt \
       --viz none --max_iterations 10000 --resume --load_run warmstart
   ```
   PPO・環境パラメータは `env_cfg.yaml` / `agent_cfg.yaml` と一致するはず(乱数シードは非固定なので数値は完全一致しない)
3. 評価:
   ```bash
   ~/work/IsaacLab/isaaclab.sh -p src/eval/eval_stairs.py   --viz none --task Unitree-Go2-Velocity-StairsV2 --num_envs 200 --steps 2000
   ~/work/IsaacLab/isaaclab.sh -p src/eval/eval_stairs.py   --viz none --task Unitree-Go2-Velocity-Obstacle --num_envs 200 --steps 2000
   ~/work/IsaacLab/isaaclab.sh -p src/eval/eval_obstacle.py --viz none --task Unitree-Go2-Velocity-Obstacle --num_envs 200 --steps 2000
   ~/work/IsaacLab/isaaclab.sh -p src/eval/eval_obstacle.py --viz none --task Unitree-Go2-Velocity-StairsV2 --num_envs 200 --steps 2000  # 盲走対照
   ```
   注意: ウォームスタート用ディレクトリが `logs/rsl_rl/` にあると「最新 run」自動選択がそれを拾うため、
   **評価は必ず `--checkpoint` で model_9999.pt を明示**する。階段の成功指標は `summit_ok` 列
   (階段帯を渡り切り、かつエピソード中無転倒)を使う — `summit_rate` は渡り切り後の転倒も成功に数える緩い判定。

## 環境(2026-08-28 時点で検証済み)

| 要素 | バージョン |
|---|---|
| Isaac Sim | 6.0.1-rc.7(`/opt/IsaacSim`、release.42383) |
| Isaac Lab | `v3.0.0-beta2.patch1`(commit `ffff603e`) |
| unitree_rl_lab upstream | `isaaclab3.0` ブランチ commit `2522f89` |
| rsl-rl-lib | 5.0.1 |
| torch | 2.10.0+cu128 |
| GPU | NVIDIA L40S 46GB |
| Python | Isaac Sim 同梱 3.12、conda 不使用(`isaaclab.sh -p` 経由) |

注意: `unitree_rl_lab.sh` は conda 前提のため使わない。インストール手順はリポジトリルートの README.md を参照
(unitree_rl_lab を editable install した上で、本リポジトリを `isaaclab.sh -p -m pip install -e .`)。
