import torch

@torch.no_grad()
def usage_effective_stats(p: torch.Tensor, eps: float = 1e-12):
    """
    p: (n_feats,) probability per selection slot, sum ~ 1
    Returns:
      entropy (nats), N_eff=exp(entropy), n90, n95, top10sum, top100sum, maxp
    """
    p = p.float()
    p = p / (p.sum() + eps)

    # entropy in nats
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum().item()
    n_eff = float(torch.exp(torch.tensor(entropy)).item())

    # top-n coverage
    vals, _ = torch.sort(p, descending=True)
    csum = torch.cumsum(vals, dim=0)

    def n_for(thr):
        # smallest n such that cumulative sum >= thr
        return int((csum < thr).sum().item() + 1)

    n90 = n_for(0.90)
    n95 = n_for(0.95)

    top10sum = vals[:10].sum().item()
    top100sum = vals[:100].sum().item()
    maxp = vals[0].item()

    return {
        "entropy_nats": entropy,
        "N_eff": n_eff,
        "n90": n90,
        "n95": n95,
        "maxp": maxp,
        "top10sum": top10sum,
        "top100sum": top100sum,
    }



@torch.no_grad()
def corr(a, b, eps=1e-8):
    # a,b: same shape, flatten
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + eps)

@torch.no_grad()
def dim_vs_frame_stats_corr(x, dim=951):
    """
    x: (B,T,D) raw
    计算指定维度与每帧 norm / std / mean 的相关
    """
    assert x.dim() == 3
    # 每帧统计
    frame_norm = torch.linalg.norm(x.float(), dim=-1)      # (B,T)
    frame_std  = x.float().std(dim=-1)                     # (B,T)
    frame_mean = x.float().mean(dim=-1)                    # (B,T)

    xd = x[..., dim].float()                               # (B,T)

    print(f"=== Correlations for dim={dim} on raw x ===")
    print("corr(x_dim, frame_norm):", corr(xd, frame_norm).item())
    print("corr(x_dim, frame_std ):", corr(xd, frame_std ).item())
    print("corr(x_dim, frame_mean):", corr(xd, frame_mean).item())




@torch.no_grad()
def per_dim_scale_ranking(x, topk=20):
    """
    x: (B,T,D)
    打印每维的 |mean|、std、abs max、以及 p99/p99.9（用采样近似）
    """
    assert x.dim() == 3
    B, T, D = x.shape
    N = B * T
    X = x.reshape(N, D).float()          # (N, D)
    AX = X.abs()

    mean = X.mean(dim=0)                 # (D,)
    std  = X.std(dim=0)                  # (D,)
    amax = AX.max(dim=0).values          # (D,)

    # 采样估计每维 p99 / p99.9（避免 N 很大时慢）
    m = min(N, 20000)
    idx = torch.randint(0, N, (m,), device=x.device)
    AXs = AX[idx]                        # (m, D)
    qs = torch.tensor([0.99, 0.999], device=x.device)
    qv = torch.quantile(AXs, qs, dim=0)  # (2, D)
    p99, p999 = qv[0], qv[1]

    def show_top(title, v):
        tv, ti = torch.topk(v, topk)
        print(f"\n=== Top-{topk} by {title} ===")
        for i in range(topk):
            d = int(ti[i].item())
            print(f"dim {d:4d}  {title}={tv[i].item():8.3f}  mean={mean[d].item():8.3f}  std={std[d].item():8.3f}  max={amax[d].item():8.1f}  p99={p99[d].item():8.1f}  p99.9={p999[d].item():8.1f}")

    show_top("|mean|", mean.abs())
    show_top("std", std)
    show_top("max|x|", amax)
    show_top("p99.9|x|", p999)

    # 给一个全局摘要：mean/std 的分布
    print("\n=== Summary across dims ===")
    print("mean:  min/median/p90/p99/max =",
          mean.min().item(),
          mean.median().item(),
          torch.quantile(mean, torch.tensor(0.90, device=x.device)).item(),
          torch.quantile(mean, torch.tensor(0.99, device=x.device)).item(),
          mean.max().item())
    print("|mean|: median/p99/max =",
          mean.abs().median().item(),
          torch.quantile(mean.abs(), torch.tensor(0.99, device=x.device)).item(),
          mean.abs().max().item())
    print("std:   median/p99/max =",
          std.median().item(),
          torch.quantile(std, torch.tensor(0.99, device=x.device)).item(),
          std.max().item())

# 用法
# per_dim_scale_ranking(x, topk=20)

@torch.no_grad()
def top_dims_spike_table(x, dims, thresholds=(50, 100, 200)):
    """
    x: (B,T,D)
    dims: list[int] 需要分析的维度索引
    thresholds: 统计 P(|x| > thr)
    """
    assert x.dim() == 3
    B, T, D = x.shape
    N = B * T
    X = x.reshape(N, D).abs().float()  # (N, D)

    dims = torch.tensor(dims, device=x.device, dtype=torch.long)
    Xd = X[:, dims]  # (N, len(dims))

    # 基本统计
    mean = Xd.mean(dim=0)
    std  = Xd.std(dim=0)
    mx   = Xd.max(dim=0).values

    # 分位数（对每个维度）
    qs = torch.tensor([0.99, 0.999, 0.9999], device=x.device)
    qv = torch.quantile(Xd, qs, dim=0)  # (3, len(dims))

    # 阈值超过率
    rates = {}
    for thr in thresholds:
        rates[thr] = (Xd > thr).float().mean(dim=0)

    print(f"=== Top dims spike table on raw |x|  (N={N}, B={B}, T={T}) ===")
    header = "dim | mean  std   max   p99   p99.9 p99.99 " + " ".join([f"P>{thr}" for thr in thresholds])
    print(header)
    for i, d in enumerate(dims.tolist()):
        row = [
            f"{d:4d}",
            f"{mean[i].item():6.2f}",
            f"{std[i].item():6.2f}",
            f"{mx[i].item():6.1f}",
            f"{qv[0,i].item():6.1f}",
            f"{qv[1,i].item():6.1f}",
            f"{qv[2,i].item():6.1f}",
        ]
        for thr in thresholds:
            row.append(f"{rates[thr][i].item():6.4f}")
        print(" ".join(row))



@torch.no_grad()
def per_dim_tail_summary(x, q_list=(0.99, 0.999), sample_per_dim=20000, topk=20):
    """
    x: (B,T,D)
    返回每个维度的 abs(x) 分位数估计（采样），并打印 topk 维度排行。
    """
    assert x.dim() == 3
    B, T, D = x.shape
    N = B * T

    # (N, D)
    X = x.reshape(N, D).abs()

    # 每个维度采样 sample_per_dim 个点（带放回）
    m = min(sample_per_dim, N)
    idx = torch.randint(0, N, (m,), device=x.device)
    Xs = X[idx].float()  # (m, D)

    qs = torch.tensor(q_list, device=x.device)
    # 对 dim=0（样本维）算分位数 => (len(q_list), D)
    qv = torch.quantile(Xs, qs, dim=0)

    # 打印整体统计
    for qi, q in enumerate(q_list):
        v = qv[qi]
        print(f"\n=== per-dim |x| quantile q={q} ===")
        print("min/median/mean/p90/p99/max:",
              v.min().item(),
              v.median().item(),
              v.mean().item(),
              torch.quantile(v, torch.tensor(0.90, device=x.device)).item(),
              torch.quantile(v, torch.tensor(0.99, device=x.device)).item(),
              v.max().item())

        # top-k 维度
        top_vals, top_idx = torch.topk(v, topk)
        print(f"top-{topk} dims (idx: value):")
        for i in range(topk):
            print(int(top_idx[i].item()), float(top_vals[i].item()))

    return qv  # (len(q_list), D)

# 用法：
# qv = per_dim_tail_summary(x, q_list=(0.99, 0.999), sample_per_dim=20000, topk=20)


class ActivationStats:
    """
    Track feature firing rate with EMA.
    For Top-k SAE, "firing" should mean: selected by top-k mask.
    firing_rate[i] ~ P(feature i is selected in top-k for a token)
    """
    def __init__(self, n_feats: int, ema_beta: float = 0.99, eps: float = 1e-6, device="cpu"):
        self.n_feats = n_feats
        self.ema_beta = ema_beta
        self.eps = eps
        self.firing_rate = None

    @torch.no_grad()
    def _init_if_needed(self, device):
        if self.firing_rate is None:
            self.firing_rate = torch.zeros(self.n_feats, device=device)

    @torch.no_grad()
    def update_from_topk(self, topk_idx: torch.Tensor):
        """
        topk_idx: [B,T,K] or [BT,K]
        Updates firing rate by counting selected indices.
        """
        device = topk_idx.device
        self._init_if_needed(device)

        idx = topk_idx.reshape(-1)  # [B*T*K]
        counts = torch.bincount(idx, minlength=self.n_feats).float()  # [E]

        if topk_idx.dim() == 3:
            BT = topk_idx.shape[0] * topk_idx.shape[1]
        else:
            BT = topk_idx.shape[0]

        fired_rate_batch = counts / max(BT, 1)
        self.firing_rate.mul_(self.ema_beta).add_(fired_rate_batch * (1 - self.ema_beta))

    @torch.no_grad()
    def dead_mask(self, device, threshold: float = 1e-4):
        self._init_if_needed(device)
        return self.firing_rate < threshold

    @torch.no_grad()
    def alive_count(self, threshold: float = 1e-4):
        if self.firing_rate is None:
            return 0
        return int((self.firing_rate >= threshold).sum().item())



class RunningMeanStd:
    def __init__(self, D: int, eps: float = 1e-5, device="cpu"):
        self.eps = eps
        self.n = torch.tensor(0.0, device=device)
        self.mean = torch.zeros(D, device=device)
        self.M2 = torch.zeros(D, device=device)  # sum of squared diffs

    @torch.no_grad()
    def update(self, x_btd: torch.Tensor):
        # x: [B,T,D] or [N,D]
        if x_btd.dim() == 2:
            x = x_btd
        else:
            x = x_btd.reshape(-1, x_btd.size(-1))

        x = x.float()
        n2 = x.size(0)
        if n2 == 0:
            return

        mean2 = x.mean(dim=0)
        # var * n (unbiased not needed here)
        M2_2 = ((x - mean2) ** 2).sum(dim=0)

        n1 = self.n
        n2t = torch.tensor(float(n2), device=x.device)
        n = n1 + n2t
        delta = mean2 - self.mean

        self.mean = self.mean + delta * (n2t / n)
        print(self.mean.shape, self.mean[951], self.mean[:10])
        self.M2 = self.M2 + M2_2 + (delta ** 2) * (n1 * n2t / n)
        self.n = n
        

    @torch.no_grad()
    def finalize(self):
        var = self.M2 / torch.clamp(self.n, min=1.0)
        std = torch.sqrt(var + self.eps)
        return self.mean, std


class DimScalerEMA:
    def __init__(self, D, beta=0.99, eps=1e-6, device="cuda"):
        self.beta = beta
        self.eps = eps
        self.sigma = torch.ones(D, device=device)

    @torch.no_grad()
    def update(self, x):  # x: (..., D)
        # 这里用 std，也可以用 abs 的 p90/p95 之类鲁棒尺度
        batch_std = x.float().std(dim=tuple(range(x.dim()-1)))  # (D,)
        self.sigma.mul_(self.beta).add_(batch_std * (1 - self.beta))

