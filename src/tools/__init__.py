"""Go2 階段昇降+障害物回避タスク(unitree_rl_lab の上に載る拡張パッケージ)。

import するだけで全 gym タスクが登録される。
"""

from . import stairs  # noqa: F401  (Stairs / StairsV2 の登録)
from . import stairs_avoid  # noqa: F401  (Obstacle 系 4 タスクの登録)
