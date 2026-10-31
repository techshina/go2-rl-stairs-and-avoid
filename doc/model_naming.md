# Go2 ポリシーの命名規則(正準対応表)

2026-09-02 制定。以後、資料・会話・ドキュメントではここで定める正準名を使う。
gym タスク ID・log ディレクトリ・S3 パスは**改名しない**(コミット済み文書・学習ログ・S3 バックアップ・artifact が旧 ID を参照しているため)。

## 命名規則

- 系列は 2 つ: **Stairs**(階段のみ、回避なし)/ **StairsAvoid**(階段+障害物回避)
- 世代は時系列で `-v1, -v2, ...`。実装キーの括弧併記(例: v2(HM))は付けない
- **日本語正式名は「回避+階段vN」**(StairsAvoid-vN と 1:1 対応。日本語資料はこちら、英語名・コード関連は StairsAvoid-vN)
- 派生・対照ランは世代を増やさず接尾辞: 例 `StairsAvoid-v2d` / 回避+階段v2d(diverse 対照)
- gym タスク ID 内では gymnasium が `-v2` サフィックスをバージョンとして解釈するため `StairsV2` のような連結表記になっている(既知の制約)

## 対応表

| 正準名(英語) | 日本語正式名 | 旧称 | gym タスク ID | env cfg | 学習 run(checkpoint) | 状態 |
|---|---|---|---|---|---|---|
| Stairs-v1 | 階段v1 | Stairs v1 | `Unitree-Go2-Velocity-Stairs` | `stairs/velocity_stairs_env_cfg.py` | `unitree_go2_velocity_stairs/2026-08-04_19-22-39` | 旧世代(S3 非保存) |
| **Stairs-v2** | 階段特化 | 階段特化(v2) | `Unitree-Go2-Velocity-StairsV2` | `stairs/velocity_stairs_env_cfg.py`(StairsV2) | `unitree_go2_velocity_stairsv2/2026-08-05_14-37-58/model_9999.pt` | 階段ベースライン |
| StairsAvoid-v1 | 回避+階段v1 | lidar 版 / 階段+回避 | `Unitree-Go2-Velocity-Obstacle` | `stairs_avoid/velocity_obstacle_env_cfg.py` | `unitree_go2_velocity_obstacle/2026-08-12_10-11-55/model_9999.pt` | 参考(長階段スタール) |
| StairsAvoid-v2 | 回避+階段v2 | HM 版 | `Unitree-Go2-Velocity-ObstacleHM` | `stairs_avoid/velocity_heightmap_env_cfg.py` | `unitree_go2_velocity_obstaclehm/2026-08-31_16-22-58/model_9999.pt` | 参考(高段差劣化) |
| StairsAvoid-v2d | 回避+階段v2d | HMDiverse | `Unitree-Go2-Velocity-ObstacleHMDiverse` | `stairs_avoid/velocity_heightmap_env_cfg.py`(Diverse) | `unitree_go2_velocity_obstaclehmdiverse/2026-08-31_18-19-01/model_9999.pt` | v2 の対照実験 |
| **StairsAvoid-v3** | 回避+階段v3 | Ext 版 | `Unitree-Go2-Velocity-ObstacleExt` | `stairs_avoid/velocity_extgrid_env_cfg.py` | `unitree_go2_velocity_obstacleext/2026-09-01_17-03-39/model_9999.pt` | **採用版** |
| (名称なし) | — | 案①(ObstacleNear) | `Unitree-Go2-Velocity-ObstacleNear` | `stairs_avoid/velocity_nearft_env_cfg.py` | 未学習(warmstart ckpt 配置のみ) | 削除済み(2026-10-30、git 履歴に残置) |

checkpoint パスの前置きはすべて `logs/rsl_rl/`。

## 注意

- 過去資料に出る「one policy」「階段+回避」「lidar 版 / HM 版 / Ext 版」は旧称(「回避+階段vN」は日本語正式名として継続使用)。artifact・CLAUDE.md は順次この表に揃える
- ObstacleHM / ObstacleExt には warmstart 用ディレクトリ(`*_warmstart`)が同居しており「最新 run」自動選択を shadow する。**評価は必ず `--checkpoint` で上表の run を明示**すること
- 各 run の再現情報(cfg スナップショット・diff・手順)は `doc/train_runs/` を参照
- **logs/ のレイアウト(2026-09-02 再編)**: 学習・評価ログとデモ動画はポリシー別フォルダ `logs/<正準名の小文字>/`(例 `logs/stairsavoid-v3/`、生クリップは各配下の `videos/`)に置く。`logs/rsl_rl/` のみ学習成果物としてタスク別のまま(resume・checkpoint 自動探索が依存するため移動しない)。`record_demo.py` の出力は `logs/videos/<out>/` に落ちるので該当ポリシーフォルダへ随時移す
