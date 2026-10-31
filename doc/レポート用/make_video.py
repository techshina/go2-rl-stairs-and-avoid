"""成果発表用の紹介動画(約80秒)を Blender の動画編集機能(VSE)で組み立てて書き出す。

実行(ローカル Mac):
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python doc/レポート用/make_video.py -- [--still 秒,秒,...]

--still を付けると動画は書き出さず、指定秒のコマを PNG で out/ に書き出す(確認用)。
素材はこのファイルと同じフォルダの rl-video-step-0.mp4(階段) / rl-video-step-1.mp4(障害物) / Picture1.png(学習中)。
"""

import glob
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
OUT_MP4 = os.path.join(OUT_DIR, "go2_stairs_avoid.mp4")

FPS = 50  # 素材動画が 50fps なので合わせる(ずらすと再生速度が変わる)
W, H = 1920, 1080

FONT = next(f for f in sorted(glob.glob("/System/Library/Fonts/*W6.ttc")) if "丸" not in f)
FONT_LIGHT = next(f for f in sorted(glob.glob("/System/Library/Fonts/*W3.ttc")) if "丸" not in f)

BG = (0.957, 0.965, 0.973)  # 背景(明るいグレー)
INK = (0.12, 0.16, 0.22, 1.0)  # 本文
SUB = (0.36, 0.40, 0.47, 1.0)  # 補足
ACCENT = (0.13, 0.39, 0.80, 1.0)  # 見出しのアクセント
FADE = 15  # フェード長(フレーム)

# ---------------------------------------------------------------------------
# 構成(秒)。尺を変えるときはここだけ触ればよい
# ---------------------------------------------------------------------------
T_TITLE = (0, 6)
T_AIM = (6, 15)
T_TRAIN = (15, 29)
T_STAIRS = (29, 39)  # 階段(等速)
T_STAIRS_SLOW = (39, 49)  # 階段(スロー)
T_OBST = (49, 69)  # 障害物
T_END = (69, 77)

STAIRS_SKIP = 2  # 素材の1コマ目が黒いので飛ばす
SLOW_SRC = (150, 400)  # スロー再生する素材上の区間(フレーム)。250f → 0.5倍で 500f


def f(sec):
    return int(round(sec * FPS)) + 1


scene = bpy.context.scene
ed = scene.sequence_editor_create()
strips = ed.strips


def fade(strip, start, end, fin=FADE, fout=FADE, peak=1.0):
    """blend_alpha を keyframe してフェードイン/アウト。"""
    keys = []
    if fin:
        keys += [(start, 0.0), (start + fin, peak)]
    else:
        keys += [(start, peak)]
    if fout:
        keys += [(end - fout, peak), (end, 0.0)]
    for fr, a in keys:
        strip.blend_alpha = a
        strip.keyframe_insert("blend_alpha", frame=fr)


def color(ch, start, end, rgb, alpha=1.0, fin=0, fout=0):
    s = strips.new_effect(name=f"color{ch}_{start}", type="COLOR", channel=ch, frame_start=start, length=end - start)
    s.color = rgb
    s.width, s.height = W, H  # Python から作ると 1x1 のままで描画されない
    s.blend_type = "ALPHA_OVER"
    fade(s, start, end, fin, fout, peak=alpha)
    return s


def text(ch, start, end, body, size, y, x=0.5, col=INK, font=FONT, align="CENTER", box=None, fin=FADE, fout=FADE):
    s = strips.new_effect(name=f"text{ch}_{start}", type="TEXT", channel=ch, frame_start=start, length=end - start)
    s.text = body
    s.font = bpy.data.fonts.load(font, check_existing=True)
    s.font_size = size
    s.color = col
    s.location = (x, y)
    s.alignment_x = align
    s.anchor_x = align
    s.anchor_y = "CENTER"
    s.wrap_width = 0.9
    if box:
        s.use_box = True
        s.box_color = box
        s.box_margin = 0.012
    s.blend_type = "ALPHA_OVER"
    fade(s, start, end, fin, fout)
    return s


def movie(ch, start, path, src_start=1, src_end=None):
    s = strips.new_movie(os.path.basename(path), path, ch, start - (src_start - 1), fit_method="FIT")
    s.frame_offset_start = src_start - 1
    if src_end is not None:
        s.frame_offset_end = s.frame_duration - src_end
    s.blend_type = "ALPHA_OVER"
    return s


def caption(ch, start, end, title, sub=None):
    """動画に重ねる左下の字幕。"""
    box = (0.08, 0.10, 0.14, 0.72)
    text(ch, start, end, title, 56, 0.185 if sub else 0.12, x=0.05, col=(1, 1, 1, 1), align="LEFT", box=box)
    if sub:
        text(ch + 1, start, end, sub, 34, 0.085, x=0.05, col=(0.88, 0.91, 0.95, 1), font=FONT_LIGHT, align="LEFT", box=box)


# 全体の下地
color(1, 1, f(T_END[1]), BG).blend_type = "REPLACE"

# 1. タイトル -----------------------------------------------------------------
a, b = f(T_TITLE[0]), f(T_TITLE[1])
img = strips.new_image("title_bg", os.path.join(HERE, "Picture1.png"), 2, a, fit_method="FILL")
img.frame_final_end = b
img.blend_type = "ALPHA_OVER"
fade(img, a, b, fin=0)
color(3, a, b, BG, alpha=0.78, fin=0)
text(4, a, b, "Unitree Go2 × 強化学習", 44, 0.64, col=ACCENT, fin=FADE * 2)
text(5, a, b, "階段昇降と障害物回避を\n1つのポリシーで", 104, 0.47, fin=FADE * 2)
text(6, a, b, "Isaac Lab によるシミュレーション学習", 40, 0.27, col=SUB, font=FONT_LIGHT, fin=FADE * 2)

# 2. ねらい -------------------------------------------------------------------
a, b = f(T_AIM[0]), f(T_AIM[1])
text(2, a, b, "ねらい", 48, 0.76, col=ACCENT)
text(3, a, b, "階段の昇り降りと、平地での障害物回避を\n1つのポリシーで実現する", 76, 0.56)
text(4, a + FPS, b, "入力:胴体まわりの高さマップ(前方に拡張)", 44, 0.33, col=SUB, font=FONT_LIGHT)
text(5, a + FPS, b, "学習:PPO(強化学習)/ Isaac Lab", 44, 0.24, col=SUB, font=FONT_LIGHT)

# 3. 学習中の様子 -------------------------------------------------------------
a, b = f(T_TRAIN[0]), f(T_TRAIN[1])
img = strips.new_image("train", os.path.join(HERE, "Picture1.png"), 2, a, fit_method="FILL")
img.frame_final_end = b
img.blend_type = "ALPHA_OVER"
fade(img, a, b)
# ゆっくり寄りながら右(柱の区画)へ流す。scale は読み込み時の FILL 倍率に掛ける
base = img.transform.scale_x
for fr, sc, ox in [(a, 1.0, 0), (b, 1.22, -140)]:
    img.transform.scale_x = img.transform.scale_y = base * sc
    img.transform.offset_x = ox
    for p in ("scale_x", "scale_y", "offset_x"):
        img.transform.keyframe_insert(p, frame=fr)
text(5, a, b, "学習中の様子", 40, 0.92, x=0.05, col=(1, 1, 1, 1), align="LEFT", box=(0.13, 0.39, 0.80, 0.9))
caption(3, a + FPS, b, "シミュレーション上で多数の個体を同時に学習", "階段・柱の地形をランダムに生成し、上達に応じて難易度を上げる")

# 4. 階段 ---------------------------------------------------------------------
stairs = os.path.join(HERE, "rl-video-step-0.mp4")
a, b = f(T_STAIRS[0]), f(T_STAIRS[1])
m = movie(2, a, stairs, src_start=STAIRS_SKIP)
m.frame_final_end = b
fade(m, a, b, fout=0)
text(5, a, b, "階段昇降", 40, 0.92, x=0.05, col=(1, 1, 1, 1), align="LEFT", box=(0.13, 0.39, 0.80, 0.9), fout=0)
caption(3, a + FPS // 2, b, "高さマップで段差を捉えて昇る", None)

# 4'. 階段スロー
a, b = f(T_STAIRS_SLOW[0]), f(T_STAIRS_SLOW[1])
# リタイミング: SLOW_SRC 区間の終端キーを後ろへずらして 0.5 倍に引き延ばす。
# (SPEED エフェクトは入力ストリップの尺の外を読めず、後半が空になる)
sp = strips.new_movie("slow", stairs, 8, a - (SLOW_SRC[0] - 1), fit_method="FIT")
sp.retiming_keys.add(timeline_frame=a)
k_end = sp.retiming_keys.add(timeline_frame=a + SLOW_SRC[1] - SLOW_SRC[0])
k_end.timeline_frame = b
sp.frame_final_start, sp.frame_final_end = a, b
sp.blend_type = "ALPHA_OVER"
fade(sp, a, b, fin=0)
text(10, a, b, "階段昇降", 40, 0.92, x=0.05, col=(1, 1, 1, 1), align="LEFT", box=(0.13, 0.39, 0.80, 0.9), fin=0)
text(11, a, b, "スロー再生 ×0.5", 36, 0.92, x=0.95, col=(1, 1, 1, 1), align="RIGHT", box=(0.08, 0.10, 0.14, 0.72), fin=0)

# 5. 障害物 -------------------------------------------------------------------
a, b = f(T_OBST[0]), f(T_OBST[1])
m = movie(2, a, os.path.join(HERE, "rl-video-step-1.mp4"), src_start=STAIRS_SKIP)
m.frame_final_end = b
fade(m, a, b)
text(5, a, b, "障害物回避", 40, 0.92, x=0.05, col=(1, 1, 1, 1), align="LEFT", box=(0.13, 0.39, 0.80, 0.9))
caption(3, a + FPS // 2, b, "柱の間を、ぶつからずに抜けていく", None)

# 6. まとめ -------------------------------------------------------------------
a, b = f(T_END[0]), f(T_END[1])
text(2, a, b, "まとめ", 48, 0.72, col=ACCENT, fout=0)
text(3, a, b, "1つのポリシーで\n階段昇降 + 障害物回避 を獲得", 84, 0.50, fout=0)
text(4, a + FPS, b, "Unitree Go2 / Isaac Lab", 40, 0.28, col=SUB, font=FONT_LIGHT, fout=0)

# ---------------------------------------------------------------------------
# 出力設定
# ---------------------------------------------------------------------------
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = f(T_END[1]) - 1
scene.render.use_sequencer = True
scene.view_settings.view_transform = "Standard"  # 既定の AgX だと素材がくすむ
os.makedirs(OUT_DIR, exist_ok=True)

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
if "--still" in argv:
    scene.render.image_settings.file_format = "PNG"
    for sec in argv[argv.index("--still") + 1].split(","):
        scene.frame_set(f(float(sec)))
        scene.render.filepath = os.path.join(OUT_DIR, f"still_{float(sec):05.1f}s.png")
        bpy.ops.render.render(write_still=True)
else:
    scene.render.image_settings.media_type = "VIDEO"  # Blender 5 以降、FFMPEG はこれを先に指定する
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = OUT_MP4
    bpy.ops.render.render(animation=True)
    print("WROTE", OUT_MP4)
