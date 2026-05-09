"""Dashboard trực quan kết quả JAMPR VRPTW.

Hiển thị:
  - Bảng số liệu: cost, số xe, feasibility
  - Biểu đồ learning curve (từ checkpoints)
  - Sơ đồ routes của instance tốt nhất
  - Biểu đồ phân phối cost

Usage:
    python scripts/visualize.py --checkpoint outputs/checkpoints/cvrptw_tw1_20_mcon3_epoch010.pt
    python scripts/visualize.py --all_checkpoints  (so sánh tất cả epoch)
    python scripts/visualize.py --checkpoint ... --n_samples 5
"""

import os
import sys
import glob
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import torch

# Windows console UTF-8 fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm

from src.models.jampr import JAMPRModel
from src.data.generator import VRPTWDataGenerator
from src.evaluation.metrics import compute_cost, check_feasibility

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────── Màu sắc đẹp ────────────────────────────
ROUTE_COLORS = [
    '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6',
    '#1ABC9C', '#E67E22', '#2980B9', '#8E44AD', '#27AE60',
    '#D35400', '#16A085', '#C0392B', '#7D3C98', '#1F618D',
]
BG_COLOR   = '#0F1117'
CARD_COLOR = '#1A1D27'
TEXT_COLOR = '#E8EAF0'
ACCENT     = '#4FC3F7'
GREEN      = '#69F0AE'
RED        = '#FF5252'
YELLOW     = '#FFD740'


# ══════════════════════════════════════════════════════════════════════
# 1.  Tải model + sinh batch
# ══════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str):
    """Tải model từ checkpoint, trả về (model, config, epoch)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    epoch  = ckpt.get('epoch', '?')
    model  = JAMPRModel(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config, epoch


def run_inference(model, config, n_samples: int = 8):
    """Chạy greedy inference trên n_samples instances ngẫu nhiên."""
    tc      = config.get('training', config)
    problem = tc.get('problem', 'cvrptw_tw1')
    n_cust  = tc.get('n_customers', 20)

    gen   = VRPTWDataGenerator()
    batch = gen.generate_batch(problem, n_cust, n_samples, seed=42)

    with torch.no_grad():
        solutions, _ = model(batch, greedy=True)

    costs = compute_cost(solutions, batch, problem)
    return solutions, batch, costs, problem


# ══════════════════════════════════════════════════════════════════════
# 2.  Bảng số liệu (console + subplot)
# ══════════════════════════════════════════════════════════════════════

def print_metrics_table(solutions, costs, batch):
    """In bảng kết quả ra console."""
    header = f"{'Instance':>8} | {'Cost':>8} | {'#Tours':>6} | {'#Nodes':>7} | {'Feasible':>8}"
    sep    = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for i, (sol, cost) in enumerate(zip(solutions, costs.tolist())):
        n_tours = len(sol)
        n_nodes = sum(len(t) for t in sol)
        inst = {k: v[i] for k, v in batch.items() if isinstance(v, torch.Tensor) and v.ndim > 0}
        feasible, _ = check_feasibility(sol, inst)
        flag = 'OK' if feasible else 'NO'
        print(f"{i+1:>8} | {cost:>8.4f} | {n_tours:>6} | {n_nodes:>7} | {flag:>8}")
    print(sep)
    finite_costs = [c for c in costs.tolist() if c != float('inf') and c == c]  # exclude inf and NaN
    if finite_costs:
        mean_c = float(np.mean(finite_costs))
        std_c  = float(np.std(finite_costs))
    else:
        mean_c, std_c = float('inf'), 0.0
    print(f"{'Average':>8} | {mean_c:>8.4f} | {'':>6} | {'':>7} | std={std_c:.4f}")
    print(sep + '\n')
    return mean_c, std_c


def draw_metrics_table(ax, solutions, costs, batch):
    """Vẽ bảng số liệu vào matplotlib axis."""
    rows = []
    for i, (sol, cost) in enumerate(zip(solutions, costs.tolist())):
        n_tours = len(sol)
        n_nodes = sum(len(t) for t in sol)
        inst = {k: v[i] for k, v in batch.items() if isinstance(v, torch.Tensor) and v.ndim > 0}
        feasible, _ = check_feasibility(sol, inst)
        rows.append([f'{i+1}', f'{cost:.3f}', str(n_tours), str(n_nodes),
                     '✓' if feasible else '✗'])

    col_labels = ['#', 'Cost', 'Vehicles', 'Customers\nServed', 'Feasible']
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Màu header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#263238')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Màu rows xen kẽ
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor('#1E2329' if i % 2 == 0 else '#232A33')
            tbl[(i, j)].set_text_props(color='#CFD8DC')
            # Feasible column
            if j == 4:
                val = rows[i - 1][4]
                tbl[(i, j)].set_text_props(
                    color='#69F0AE' if val == '✓' else '#FF5252',
                    fontweight='bold',
                )

    ax.set_title('📊 Kết quả Inference (Greedy)', color=TEXT_COLOR,
                 fontsize=12, pad=12, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════
# 3.  Sơ đồ routes
# ══════════════════════════════════════════════════════════════════════

def draw_routes(ax, instance_idx: int, solutions, batch, problem: str, costs):
    """Vẽ sơ đồ route cho 1 instance."""
    sol    = solutions[instance_idx]
    coords = batch['coords'][instance_idx].numpy()      # (N+1, 2)
    tw     = batch.get('time_windows')
    cost   = costs[instance_idx].item()

    ax.set_facecolor('#0D1117')

    # Vẽ depot
    ax.scatter(coords[0, 0], coords[0, 1], c='#FFD740', s=280,
               marker='*', zorder=6, linewidths=1.5,
               edgecolors='#FFF8E1', label='Depot')

    # Vẽ tất cả customers (nền)
    ax.scatter(coords[1:, 0], coords[1:, 1], c='#546E7A', s=45,
               alpha=0.5, zorder=3)

    # Màu node theo tour
    node_color = {}
    for k, tour in enumerate(sol):
        for n in tour:
            node_color[n] = ROUTE_COLORS[k % len(ROUTE_COLORS)]

    # Vẽ routes
    for k, tour in enumerate(sol):
        if not tour:
            continue
        color = ROUTE_COLORS[k % len(ROUTE_COLORS)]

        # Highlight nodes trong tour này
        tc_coords = coords[tour]
        ax.scatter(tc_coords[:, 0], tc_coords[:, 1], c=color,
                   s=70, zorder=5, edgecolors='white', linewidths=0.6)

        # Arrows: depot → node1 → ... → nodeN → depot
        route = [0] + list(tour) + [0]
        for a, b_idx in zip(route[:-1], route[1:]):
            x0, y0 = coords[a]
            x1, y1 = coords[b_idx]
            dx, dy = x1 - x0, y1 - y0
            ax.annotate('',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=1.4, alpha=0.85,
                    connectionstyle='arc3,rad=0.05',
                ),
                zorder=4,
            )

    # Label node indices
    for i in range(len(coords)):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]),
                    fontsize=6.5, ha='center', va='bottom',
                    color='#ECEFF1', fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

    # Legend (xe)
    legend_handles = []
    for k, tour in enumerate(sol):
        if tour:
            import matplotlib.patches as mpatches
            legend_handles.append(
                mpatches.Patch(color=ROUTE_COLORS[k % len(ROUTE_COLORS)],
                               label=f'Tour {k+1}: {tour}')
            )

    ax.legend(handles=legend_handles, loc='upper right', fontsize=6.5,
              facecolor='#1A1D27', edgecolor='#37474F', labelcolor='#CFD8DC',
              framealpha=0.85)

    ax.set_title(f'🗺️  Instance #{instance_idx+1} — Cost: {cost:.3f}  |  {len(sol)} routes',
                 color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_xlabel('x', color='#78909C')
    ax.set_ylabel('y', color='#78909C')
    ax.tick_params(colors='#546E7A')
    for spine in ax.spines.values():
        spine.set_edgecolor('#263238')


# ══════════════════════════════════════════════════════════════════════
# 4.  Learning curve (từ nhiều checkpoints)
# ══════════════════════════════════════════════════════════════════════

def draw_learning_curve(ax, checkpoint_dir: str, problem: str, n_cust: int):
    """Vẽ learning curve bằng cách load từng checkpoint và eval cost."""
    pattern = os.path.join(checkpoint_dir, f'{problem}_{n_cust}_*.pt')
    files   = sorted(glob.glob(pattern))
    if not files:
        ax.text(0.5, 0.5, 'Chưa có checkpoint để\nvẽ learning curve',
                ha='center', va='center', transform=ax.transAxes,
                color='#78909C', fontsize=11)
        ax.set_facecolor('#0D1117')
        return

    epochs, train_costs = [], []
    gen = VRPTWDataGenerator()

    for fpath in files:
        try:
            ckpt   = torch.load(fpath, map_location='cpu', weights_only=False)
            epoch  = ckpt.get('epoch', 0)
            metrics = ckpt.get('metrics', {})

            if 'cost_mean' in metrics:
                train_costs.append(metrics['cost_mean'])
                epochs.append(epoch)
        except Exception as e:
            logger.warning('Bỏ qua checkpoint %s: %s', fpath, e)

    if not epochs:
        ax.text(0.5, 0.5, 'Không đọc được\nmetrics từ checkpoint',
                ha='center', va='center', transform=ax.transAxes,
                color='#78909C', fontsize=11)
        return

    ax.set_facecolor('#0D1117')
    ax.plot(epochs, train_costs, 'o-', color=ACCENT, lw=2, ms=6,
            label='Train cost', zorder=4)
    ax.fill_between(epochs, train_costs, alpha=0.15, color=ACCENT)

    best_epoch = epochs[int(np.argmin(train_costs))]
    best_cost  = min(train_costs)
    ax.axhline(best_cost, color=GREEN, lw=1, ls='--', alpha=0.6,
               label=f'Best: {best_cost:.4f} (ep {best_epoch})')

    ax.set_title('📈 Learning Curve (Train Cost)', color=TEXT_COLOR,
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='#78909C')
    ax.set_ylabel('Avg Cost', color='#78909C')
    ax.legend(facecolor='#1A1D27', edgecolor='#37474F', labelcolor='#CFD8DC')
    ax.tick_params(colors='#546E7A')
    ax.grid(True, alpha=0.15, color='#37474F')
    for spine in ax.spines.values():
        spine.set_edgecolor('#263238')


# ══════════════════════════════════════════════════════════════════════
# 5.  Histogram phân phối cost
# ══════════════════════════════════════════════════════════════════════

def draw_cost_hist(ax, costs):
    """Histogram phân phối cost (bỏ qua inf/NaN)."""
    raw  = costs.numpy()
    vals = raw[np.isfinite(raw)]   # lọc bỏ inf / NaN
    n_inf = int((~np.isfinite(raw)).sum())

    ax.set_facecolor('#0D1117')
    if len(vals) == 0:
        ax.text(0.5, 0.5, 'Tất cả costs = inf\n(Model chưa hội tụ)',
                ha='center', va='center', transform=ax.transAxes,
                color=RED, fontsize=12, fontweight='bold')
        ax.set_title('Phan phoi Cost', color=TEXT_COLOR, fontsize=11, fontweight='bold')
        return

    n, bins, patches = ax.hist(vals, bins=min(len(vals), 15),
                                color=ACCENT, alpha=0.75, edgecolor='#263238')

    # Gradient màu
    for patch, val in zip(patches, bins[:-1]):
        patch.set_facecolor(cm.cool(val / (bins[-1] + 1e-8)))

    ax.axvline(vals.mean(), color=YELLOW, lw=1.8, ls='--',
               label=f'Mean: {vals.mean():.3f}')
    ax.axvline(vals.min(),  color=GREEN,  lw=1.4, ls=':',
               label=f'Min:  {vals.min():.3f}')
    if n_inf > 0:
        ax.text(0.98, 0.95, f'{n_inf} inf (TW vi pham)',
                ha='right', va='top', transform=ax.transAxes,
                color=RED, fontsize=9)

    ax.set_title('📦 Phân phối Cost', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_xlabel('Cost', color='#78909C')
    ax.set_ylabel('Count', color='#78909C')
    ax.legend(facecolor='#1A1D27', edgecolor='#37474F', labelcolor='#CFD8DC')
    ax.tick_params(colors='#546E7A')
    ax.grid(True, alpha=0.15, axis='y', color='#37474F')
    for spine in ax.spines.values():
        spine.set_edgecolor('#263238')


# ══════════════════════════════════════════════════════════════════════
# 6.  Dashboard tổng hợp
# ══════════════════════════════════════════════════════════════════════

def build_dashboard(checkpoint_path: str, n_samples: int = 8,
                    show_instance: int = 0, save_path: str = 'outputs/dashboard.png'):
    """Tạo dashboard 2×3 panels và lưu ảnh."""
    logger.info('Đang tải checkpoint: %s', checkpoint_path)
    model, config, epoch = load_model(checkpoint_path)

    tc      = config.get('training', config)
    problem = tc.get('problem', 'cvrptw_tw1')
    n_cust  = tc.get('n_customers', 20)

    logger.info('Chạy inference %d instances (problem=%s, n=%d)...', n_samples, problem, n_cust)
    solutions, batch, costs, problem = run_inference(model, config, n_samples)

    # Console table
    mean_c, std_c = print_metrics_table(solutions, costs, batch)

    # ─── Layout ───
    fig = plt.figure(figsize=(20, 13), facecolor=BG_COLOR)
    fig.suptitle(
        f'JAMPR VRPTW — Dashboard  |  {problem.upper()}  N={n_cust}  '
        f'Epoch {epoch}  |  Mean Cost: {mean_c:.4f} ± {std_c:.4f}',
        color=TEXT_COLOR, fontsize=14, fontweight='bold', y=0.99,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.38, wspace=0.30,
                           left=0.05, right=0.97, top=0.93, bottom=0.04)

    # Panel 1 (top-left): Bảng metrics
    ax_tbl = fig.add_subplot(gs[0, 0])
    draw_metrics_table(ax_tbl, solutions, costs, batch)

    # Panel 2 (top-mid): Route instance tốt nhất
    best_idx = int(costs.argmin().item())
    ax_best = fig.add_subplot(gs[0, 1])
    draw_routes(ax_best, best_idx, solutions, batch, problem, costs)

    # Panel 3 (top-right): Route instance chọn theo show_instance
    ax_sel = fig.add_subplot(gs[0, 2])
    draw_routes(ax_sel, min(show_instance, n_samples - 1), solutions, batch, problem, costs)

    # Panel 4 (bot-left): Learning curve
    ckpt_dir = os.path.dirname(checkpoint_path)
    ax_lc = fig.add_subplot(gs[1, 0])
    draw_learning_curve(ax_lc, ckpt_dir, problem, n_cust)

    # Panel 5 (bot-mid): Cost histogram
    ax_hist = fig.add_subplot(gs[1, 1])
    draw_cost_hist(ax_hist, costs)

    # Panel 6 (bot-right): #vehicles bar chart
    ax_bar = fig.add_subplot(gs[1, 2])
    n_vehicles = [len(sol) for sol in solutions]
    bars = ax_bar.bar(range(1, n_samples + 1), n_vehicles,
                      color=[ROUTE_COLORS[v % len(ROUTE_COLORS)] for v in n_vehicles],
                      edgecolor='#263238', linewidth=0.8, alpha=0.85)
    ax_bar.axhline(np.mean(n_vehicles), color=YELLOW, lw=1.5, ls='--',
                   label=f'Avg: {np.mean(n_vehicles):.1f} xe')
    ax_bar.set_facecolor('#0D1117')
    ax_bar.set_title('🚗 Số Xe Sử Dụng / Instance', color=TEXT_COLOR,
                     fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('Instance', color='#78909C')
    ax_bar.set_ylabel('# Vehicles', color='#78909C')
    ax_bar.tick_params(colors='#546E7A')
    ax_bar.legend(facecolor='#1A1D27', edgecolor='#37474F', labelcolor='#CFD8DC')
    ax_bar.grid(True, alpha=0.15, axis='y', color='#37474F')
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#263238')
    for bar, v in zip(bars, n_vehicles):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(v), ha='center', va='bottom', color=TEXT_COLOR, fontsize=9)

    # Lưu
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    logger.info('✅ Dashboard đã lưu: %s', save_path)
    return save_path


# ══════════════════════════════════════════════════════════════════════
# 7.  So sánh tất cả checkpoints
# ══════════════════════════════════════════════════════════════════════

def compare_all_checkpoints(checkpoint_dir: str, problem: str, n_cust: int,
                             save_path: str = 'outputs/compare_epochs.png'):
    """Vẽ biểu đồ so sánh cost qua từng epoch."""
    pattern = os.path.join(checkpoint_dir, f'{problem}_{n_cust}_*.pt')
    files   = sorted(glob.glob(pattern))
    if not files:
        logger.error('Không tìm thấy checkpoint nào trong %s', checkpoint_dir)
        return

    epochs, means, stds, n_veh = [], [], [], []
    gen = VRPTWDataGenerator()

    for fpath in files:
        try:
            model, config, epoch = load_model(fpath)
            solutions, batch, costs, prob = run_inference(model, config, n_samples=16)
            means.append(costs.mean().item())
            stds.append(costs.std().item())
            n_veh.append(np.mean([len(s) for s in solutions]))
            epochs.append(epoch)
            logger.info('Epoch %s: cost=%.4f ± %.4f', epoch, means[-1], stds[-1])
        except Exception as e:
            logger.warning('Lỗi khi load %s: %s', fpath, e)

    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
    fig.suptitle(f'So sánh Epochs — {problem.upper()} N={n_cust}',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    # Cost qua epoch
    ax = axes[0]
    ax.set_facecolor('#0D1117')
    means_arr = np.array(means)
    stds_arr  = np.array(stds)
    ax.plot(epochs, means_arr, 'o-', color=ACCENT, lw=2, ms=7, label='Mean cost')
    ax.fill_between(epochs, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.2, color=ACCENT, label='±1 std')
    best_e = epochs[int(np.argmin(means_arr))]
    ax.axvline(best_e, color=GREEN, lw=1.2, ls='--', label=f'Best epoch: {best_e}')
    ax.set_title('Cost qua các Epoch', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='#78909C')
    ax.set_ylabel('Cost', color='#78909C')
    ax.tick_params(colors='#546E7A')
    ax.legend(facecolor='#1A1D27', edgecolor='#37474F', labelcolor='#CFD8DC')
    ax.grid(True, alpha=0.15, color='#37474F')

    # Số xe qua epoch
    ax2 = axes[1]
    ax2.set_facecolor('#0D1117')
    ax2.plot(epochs, n_veh, 's-', color=YELLOW, lw=2, ms=7, label='Avg vehicles')
    ax2.set_title('Số Xe TB qua các Epoch', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Epoch', color='#78909C')
    ax2.set_ylabel('Avg # Vehicles', color='#78909C')
    ax2.tick_params(colors='#546E7A')
    ax2.legend(facecolor='#1A1D27', edgecolor='#37474F', labelcolor='#CFD8DC')
    ax2.grid(True, alpha=0.15, color='#37474F')

    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_edgecolor('#263238')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    logger.info('✅ So sánh epochs đã lưu: %s', save_path)


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='JAMPR Visualization Dashboard')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='outputs/checkpoints',
                        help='Directory chứa checkpoints (dùng với --all_checkpoints)')
    parser.add_argument('--all_checkpoints', action='store_true',
                        help='So sánh tất cả checkpoints trong checkpoint_dir')
    parser.add_argument('--problem', type=str, default='cvrptw_tw1')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Số instance để inference (tối đa 15 để bảng đẹp)')
    parser.add_argument('--show_instance', type=int, default=1,
                        help='Index instance để hiện thêm 1 sơ đồ route (0-based)')
    parser.add_argument('--output', type=str, default='outputs/dashboard.png')
    args = parser.parse_args()

    if args.all_checkpoints:
        compare_all_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            problem=args.problem,
            n_cust=args.n,
            save_path=args.output.replace('.png', '_compare.png'),
        )
    else:
        # Auto-detect latest checkpoint nếu không chỉ định
        ckpt = args.checkpoint
        if ckpt is None:
            pattern = os.path.join(args.checkpoint_dir, f'{args.problem}_{args.n}_*.pt')
            files   = sorted(glob.glob(pattern))
            if not files:
                logger.error('Không tìm thấy checkpoint. Hãy chạy training trước!')
                return
            ckpt = files[-1]
            logger.info('Auto-detect checkpoint mới nhất: %s', ckpt)

        build_dashboard(
            checkpoint_path=ckpt,
            n_samples=min(args.n_samples, 15),
            show_instance=args.show_instance,
            save_path=args.output,
        )


if __name__ == '__main__':
    main()
