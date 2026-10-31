# go2-rl-stairs-avoid

Unitree Go2 が**階段の昇り降りと、平地での障害物回避を1つのポリシーでこなす**ための強化学習タスク集です。
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) と [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) の上に載る拡張パッケージとして、
タスク定義・学習・評価・デモのスクリプトを提供します。

ロボットは胴体まわりの高さマップ(前方 2.0m・左右 ±1.0m、0.1m 間隔)を観測し、
階段は登る対象、柱や箱は避ける対象として区別するよう学習します。単一のポリシーで両方をこなし、
上位の経路計画は使いません。

## 性能

シミュレーション上の評価結果(前進指令 0.5m/s):

| 項目 | 結果 |
|---|---|
| 障害物(柱)への衝突 | 0 / 400 |
| 階段の登り切り(段差 17cm) | 83% |
| 階段の登り切り(段差 19cm) | 63% |

階段の「登り切り」は、階段帯を渡り切り、かつエピソード中に一度も転倒しなかった割合です。

## 実行環境

| 要素 | バージョン |
|---|---|
| GPU | NVIDIA L40S(学習は VRAM 40GB 級、デモ・評価は数 GB で可) |
| Isaac Sim | 6.0.1-rc.7 |
| Isaac Lab | `v3.0.0-beta2.patch1` |
| unitree_rl_lab | `isaaclab3.0` ブランチ、コミット `2522f89` |
| Python | Isaac Sim 同梱の 3.12 |

Isaac Lab は beta API のため、別バージョンでは観測次元エラーで動きません。

## セットアップ

Isaac Lab と unitree_rl_lab を上記のバージョンで導入し、Go2 の USD モデル
([unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model))の場所を
環境変数 `UNITREE_MODEL_DIR` に設定したうえで、本リポジトリをインストールします。

```bash
<IsaacLab>/isaaclab.sh -p -m pip install -e .
```

## 使い方

以下、`isaaclab.sh` は `<IsaacLab>/isaaclab.sh` の略です。GUI なしで実行するときは `--viz none`、
GUI で見るときは `--viz kit` を指定します(`--headless` ではありません)。

### 1. 学習

本タスクは、階段のみを学習したポリシーを初期値にして学習します。そのため 2 段階で進めます。
1 回の学習(10,000 iteration)は NVIDIA L40S で約 7.5 時間です。

```bash
# (1) 階段のみのポリシーを学習
isaaclab.sh -p src/train/train.py --task Unitree-Go2-Velocity-StairsV2 --viz none --max_iterations 10000

# (2) (1) の重みを本タスクの入力形状に移植して初期値を作る
isaaclab.sh -p src/train/ext_warmstart_surgery.py \
    --spec logs/rsl_rl/unitree_go2_velocity_stairsv2/<run>/model_9999.pt \
    --out logs/rsl_rl/unitree_go2_velocity_obstacleext/warmstart/model_0.pt

# (3) 階段+障害物回避を学習
isaaclab.sh -p src/train/train.py --task Unitree-Go2-Velocity-ObstacleExt \
    --viz none --max_iterations 10000 --resume --load_run warmstart
```

checkpoint は `logs/rsl_rl/unitree_go2_velocity_obstacleext/<run>/model_9999.pt` に、
学習時に解決された全パラメータは同じディレクトリの `params/` に保存されます。
乱数シードは固定していないため、学習結果は実行ごとに多少ばらつきます。

### 2. デモ

```bash
# GUI で歩かせる
isaaclab.sh -p src/demo/walk_demo.py --policy stairsavoid-v3 --checkpoint <model_9999.pt> --viz kit
# GUI なしで動画に記録(logs/videos/walk_<policy>/ に出力)
isaaclab.sh -p src/demo/walk_demo.py --policy stairsavoid-v3 --checkpoint <model_9999.pt> --viz none --video
```

地形やカメラを指定して録画するときは `src/demo/record_demo.py` を使います。

### 3. 評価

```bash
# 階段(段差 5〜23cm の 10 段階 × 登り/降り)
isaaclab.sh -p src/eval/eval_stairs.py --viz none --task Unitree-Go2-Velocity-ObstacleExt \
    --num_envs 400 --steps 2000 --checkpoint <model_9999.pt>
# 障害物(柱の密度 10 段階。衝突率・クリアランス)
isaaclab.sh -p src/eval/eval_obstacle.py --viz none --task Unitree-Go2-Velocity-ObstacleExt \
    --num_envs 200 --steps 2000 --checkpoint <model_9999.pt>
```

NVIDIA L40S でそれぞれ 2〜3 分です。結果の読み方:

- **`--checkpoint` は必ず指定してください。** 省略すると最新の学習ディレクトリを自動で選びますが、
  学習手順 (2) で作ったウォームスタート用ディレクトリを拾うことがあります
- 階段の成功指標は **`summit_ok`**(階段帯を渡り切り、かつ無転倒)です。`summit_rate` は
  渡り切った後の転倒も成功に数えます
- 障害物評価の「衝突」は胴体の接触だけを数えます。柱に足を取られた転倒は `bad_orientation` に
  計上されるので、転倒数とあわせて見てください

## 同梱タスク

本タスクのほかに、学習の初期値や比較に使うタスクも登録しています。

| タスク ID | 内容 |
|---|---|
| `Unitree-Go2-Velocity-ObstacleExt` | **本タスク。** 階段+障害物回避(胴体まわりの高さマップを前方に拡張) |
| `Unitree-Go2-Velocity-StairsV2` | 階段のみ(階段向けに報酬を調整し、観測履歴 5 フレーム)。本タスクの学習初期値 |
| `Unitree-Go2-Velocity-Stairs` | 階段のみ(unitree_rl_lab 標準の報酬設定) |
| `Unitree-Go2-Velocity-Obstacle` | 階段+障害物回避(水平 2D lidar で障害物を検出する比較用) |
| `Unitree-Go2-Velocity-ObstacleHM` | 階段+障害物回避(前方の粗い高さマップを追加する比較用) |
| `Unitree-Go2-Velocity-ObstacleHMDiverse` | ObstacleHM の障害物の種類を増やした比較用 |

## リポジトリ構成

```
go2-rl-stairs-avoid/
├── src/
│   ├── tools/           タスク定義(import 名は tools)
│   │   ├── stairs/        階段のみのタスク
│   │   ├── stairs_avoid/  階段+障害物回避のタスク(環境設定と観測・報酬の関数)
│   │   └── agents/        PPO の設定
│   ├── train/           学習・再生・ウォームスタート用の重み移植
│   ├── eval/            階段と障害物の定量評価
│   ├── demo/            歩行デモと動画録画
│   └── utils/           TorchScript への書き出し、タスク一覧
├── model/               checkpoint の置き場(git 管理外)
└── setup.py, pyproject.toml, config/   パッケージ定義
```

TorchScript への書き出し(`src/utils/export_policy_jit.py`)を使うと、rsl_rl なしで推論できます。
