import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------
# 설정
# -------------------------------------------------------
FRAMES_DIR = Path("/home/radar01/sdb1/jw/glm-4.6v/tmp/20260209_151825_accident_2")   # 프레임 이미지들이 있는 폴더
OUT_DIR = Path("/home/radar01/sdb1/jw/glm-4.6v/tmp/20260209_151825_accident_2_frame_diff_viz")  # 결과 저장 폴더
OUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Frame diff 계산 함수
# -------------------------------------------------------
def compute_diff(prev_img, curr_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    score = float(diff.mean())   # frame diff 정도 (평균 차이값)

    return diff, score


# -------------------------------------------------------
# 메인 실행
# -------------------------------------------------------
def main():
    paths = sorted(FRAMES_DIR.glob("*.png"))

    if len(paths) < 2:
        print("[ERROR] 프레임 이미지가 2장 이상 필요합니다.")
        return

    scores = []
    prev_img = None

    for i, p in enumerate(paths):
        curr_img = cv2.imread(str(p))
        if curr_img is None:
            print(f"[WARN] 이미지를 읽지 못했습니다: {p}")
            continue

        if prev_img is None:
            prev_img = curr_img
            continue

        diff, score = compute_diff(prev_img, curr_img)
        scores.append(score)

        # diff heatmap 만들기
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_heat = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

        # prev / curr / diff_heat 나란히 붙이기
        concat = np.concatenate([prev_img, curr_img, diff_heat], axis=1)

        label = f"{paths[i-1].name} -> {p.name} | mean_abs_diff={score:.3f}"
        cv2.putText(concat, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)

        out_path = OUT_DIR / f"diff_{i:03d}.png"
        cv2.imwrite(str(out_path), concat)

        print(f"[INFO] Saved: {out_path}  (score={score:.3f})")

        prev_img = curr_img

    # diff score 그래프 저장
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(scores) + 1), scores, marker="o")
    plt.title("Frame-to-frame Difference Score")
    plt.xlabel("Frame index (starting from 1)")
    plt.ylabel("Mean abs diff (gray)")
    plt.grid(True)

    graph_path = OUT_DIR / "diff_score_plot.png"
    plt.savefig(str(graph_path), dpi=200)
    plt.show()

    print(f"\n[INFO] Diff score plot saved: {graph_path}")
    print(f"[INFO] All diff visualization images saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
