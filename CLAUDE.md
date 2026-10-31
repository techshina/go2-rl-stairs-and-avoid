# go2-rl-stairs-avoid

Unitree Go2 の階段昇降+障害物回避ポリシーを Isaac Lab で学習・評価する拡張パッケージ。
unitree_rl_lab の上に載り、タスク登録・環境設定・学習/評価/デモのスクリプトを提供する。
セットアップ手順・バージョン要件・使い方は README.md を参照。

## タスク

| gym タスク ID | 内容 |
|---|---|
| `Unitree-Go2-Velocity-ObstacleExt` | 本タスク。階段+障害物回避(胴体まわりの高さマップを前方に拡張) |
| `Unitree-Go2-Velocity-StairsV2` | 階段のみ。本タスクの学習初期値 |
| `Unitree-Go2-Velocity-Stairs` / `-Obstacle` / `-ObstacleHM` / `-ObstacleHMDiverse` | 比較用 |

`walk_demo.py` の `--policy` は、タスク ID ではなく `stairs-v2` や `stairsavoid-v3` のような短縮名で指定する
(対応は `src/demo/walk_demo.py` の `POLICIES`)。

## コード構成

- `src/tools/` — タスク定義のパッケージ(import 名は `tools`)。`import tools` するだけで全タスクが gym に登録される
  - `stairs/` — 階段のみのタスク
  - `stairs_avoid/` — 階段+障害物回避のタスク。`velocity_*_env_cfg.py` が環境設定、`*_mdp.py` が観測・報酬の関数
  - `agents/` — PPO の設定(actor と critic で観測を分ける非対称構成)
- `src/train/`、`src/eval/`、`src/demo/`、`src/utils/` — 実行スクリプト

## 環境の作法

- **conda は使わない。** Python の実行はすべて `<IsaacLab>/isaaclab.sh -p <script>` 経由(Isaac Sim 同梱の Python 3.12)
- GUI なしは `--viz none`(`--headless` ではない)。GUI は `--viz kit`
- バージョンは README の組み合わせに固定する。Isaac Lab は beta API のため、ずれると観測次元エラーで壊れる
- Go2 の USD は環境変数 `UNITREE_MODEL_DIR` で参照する
- gymnasium はタスク ID 末尾の `-v2` をバージョンとして解釈するため、ID は `StairsV2` のように連結して書く

## 主要スクリプト

| スクリプト | 用途 |
|---|---|
| `src/train/train.py` / `play.py` | 学習・再生(rsl_rl の標準フロー) |
| `src/train/ext_warmstart_surgery.py` | StairsV2 の重みを本タスクの入力形状に移植し、学習初期値を作る |
| `src/eval/eval_stairs.py` | 階段評価(段差 10 段階 × 登り/降り、転倒率・到達率) |
| `src/eval/eval_obstacle.py` | 障害物評価(柱密度 10 段階、衝突率・クリアランス・前進速度) |
| `src/demo/walk_demo.py` | 学習済みポリシーを歩かせるデモ(GUI/録画、`--vx` で直進に固定) |
| `src/demo/record_demo.py` | デモ動画の録画(追従カメラ、地形指定 `--step_h` / `--pillars` / `--terrain_seed`) |
| `src/utils/export_policy_jit.py` | checkpoint を TorchScript に書き出す(rsl_rl なしで推論できる) |
| `src/utils/list_envs.py` | 登録済みタスクの一覧(インストール確認用) |

## 評価で間違えやすい点

- **checkpoint は必ず `--checkpoint` で明示する。** 省略時は `logs/rsl_rl/` の最新ディレクトリを自動で選ぶが、
  学習初期値用の `warmstart` ディレクトリを拾うことがある
- 階段の成功指標は **`summit_ok`**(階段帯を渡り切り、かつエピソード中に無転倒)。`summit_rate` は渡り切り後の
  転倒も成功に数える緩い判定。転倒率だけを見ると、登ろうとしないポリシーが良く見えるので単独では使わない
- 障害物評価の「衝突」は胴体の接触だけを数える。柱に足を取られた転倒は `bad_orientation` に計上されるため、
  転倒数とセットで見る
- 段差 15〜17cm の階段成績は試行数が少なく、学習ごとのばらつきが大きい(±10pt 程度は振れる)
- GUI なしの録画では ViewerCfg の追従カメラが効かない。`record_demo.py` は録画カメラを毎ステップ動かすので、
  録画にはこちらを使う

## 学習の再現

- 学習時に解決された全パラメータは、`train.py` が `logs/rsl_rl/<タスク名小文字>/<run>/params/` に保存する
- 本タスクは StairsV2 の checkpoint を初期値にして学習する(手順は README の「学習」)
- 10,000 iteration で NVIDIA L40S 約 7.5 時間。乱数シードは固定していないため、数値は完全には一致しない
- checkpoint はリポジトリに含めない。`model/` に置くか、`logs/rsl_rl/<タスク名小文字>/<run>/` に生成される
