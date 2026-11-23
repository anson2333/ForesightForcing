from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


logger = logging.getLogger(__name__)


# Supported projector widths.
_CHECKPOINT_PROJECTOR_WIDTHS = frozenset({256, 1024})


@dataclass
class GeometryTransitionConfig:
    """Geometry Transition: learnable spatial queries + layerwise Q-Former updates.

    Token layout (FLUX.2 image stream):
        img = [h_ref (N) | h_noisy_tgt (N) | z_Δ (N)]

    z_Δ starts as learnable queries and is refined after every double block by
    cross-attending to [h_ref, txt].  The transition is supervised in frozen
    teacher feature space.
    """

    enabled: bool = False
    weight: float = 1.0
    scale_weight: float = 0.1
    mode: str = "bottleneck"  # "bottleneck" | "aux"
    placement: str = "video"  # video | action
    teacher_supervision_enabled: bool = True
    action_condition: str = "dropout"
    action_full_keep_prob: float = 0.5  # P(action sees ref+z) under dropout
    inference_action_condition: str = "transition_and_ref"
    action_residual_enabled: bool = False
    action_residual_num_heads: int = 8
    action_residual_gate_init: float = 0.0
    # Eval-only multiplier for the learned ReZero gate. Training always uses
    # scale 1.0 so this can sweep deployment strength without changing weights.
    action_residual_inference_scale: float = 1.0
    # Training-only safety switch: freeze the full DiT/action backbone and
    # optimize geometry_transition (including the residual adapter) only.
    freeze_dit: bool = False
    vggt_backend: str = "vggt"  # vggt | vggt_omega
    # Optional override for the teacher source tree. Leave None to use the
    # backend-specific vendored checkout under third_party/.
    vggt_repo_path: Optional[str] = None
    vggt_model_id: str = "facebook/VGGT-1B"
    vggt_drop_heads: bool = True
    vggt_layers: tuple[int, ...] = (-1,)
    source_layer: int = -1
    source_channels: Optional[int] = None
    # --- supervision objective ---
    # "relation": align pairwise token-distance structure against frozen VGGT
    #   features. Correspondence-free and scale-invariant, so it sidesteps both
    #   the pixel-correspondence requirement and VGGT's per-forward up-to-scale
    #   normalization.
    # "geometry_forcing": per-patch feature differencing with decoupled Angular +
    #   Scale alignment (Geometry Forcing, ICLR 2026, Sec 4.2), adapted to the
    #   transition setting: the teacher target is y_Δ,p = y_tgt,p - y_ref,p taken
    #   on VGGT *features*, because VGGT's cross-frame
    #   attention already places both frames in one shared space. Requires patch
    #   correspondence, hence `warp_ref_to_target` (see below).
    # Accepts a single name or a list.
    objective: str | Sequence[str] = "geometry_forcing"
    # Mix coefficients used when several objectives are active. Each is applied on
    # top of the shared `weight`, so total = weight * (rel_w * relation_terms
    # + forc_w * forcing_terms).
    relation_weight: float = 1.0
    forcing_weight: float = 1.0
    # Weight on relation's "static pairs stay flat" term. None falls back to
    # `scale_weight` for backwards compatibility -- but note `scale_weight` is
    # lambda_Scale for geometry_forcing, so set this explicitly whenever both
    # objectives are on, otherwise one number silently controls two unrelated terms.
    background_weight: Optional[float] = None
    # Run the frozen teacher once and feed both objectives from it. Only sound when
    # `warp_ref_to_target` is False: with warping on, relation would be reading a
    # motion-compensated reference frame, which changes what its r_ref means.
    # Leaving this False costs a second VGGT forward when both objectives are on.
    share_teacher_forward: bool = False
    # Number of camera views tiled into each frame by the dataset (LIBERO packs
    # agentview+wrist side by side; RoboTwin uses a 3-view mosaic).
    num_camera_views: int = 1
    camera_tiling: str = "horizontal"  # horizontal | vertical | robotwin | none
    vggt_multiview: bool = False
    vggt_view_mode: str = "auto"  # auto | joint | per_view

    # --- geometry_forcing objective only ---------------------------------
    # Motion-compensate the reference frame onto the target's pixel grid before
    # the teacher runs. Per-patch differencing is only meaningful when patch p of
    # both frames observes the same surface; without this, y_Δ,p at a patch the
    # object moved through subtracts two unrelated surfaces.
    # NOTE: a good warp removes everything a 2D flow field can explain, so the
    # surviving signal is occlusion/disocclusion, true depth change, non-rigid
    # deformation and flow error.
    warp_ref_to_target: bool = True
    # Dense optical flow only. A global homography is useless here: with a fixed
    # camera (LIBERO agentview) RANSAC locks onto the static background and
    # returns ~identity, compensating nothing.
    warp_method: str = "flow"  # flow | none
    # Patches whose warp sampled outside the source frame are dropped: they have
    # no valid correspondence at all.
    warp_drop_invalid: bool = True
    # Weight on the Angular term (lambda_Angular in the paper; 0.5 there).
    angular_weight: float = 0.5
    forcing_norm_weight: float = 0.01
    # Hidden width of g_phi, the Scale-alignment prediction head. None -> teacher_channels.
    scale_head_hidden: Optional[int] = None
    # Token subsampling for the O(N^2) relation matrix. 0 disables subsampling.
    relation_num_tokens: int = 256
    # Blend of learned action relevance with a uniform prior (eta in Eq.14 of
    # MECo-WAM); prevents the weight map collapsing onto a single location.
    action_weight_eta: float = 0.25
    action_weight_temp: float = 0.07
    use_action_aware_weight: bool = True
    # Width of the action expert's hidden state; set from the model so the
    # relevance projection exists before the optimizer is built.
    action_channels: Optional[int] = None
    # Width of the frozen VGGT teacher features (VGGT-1B aggregator output).
    teacher_channels: int = 2048
    # Geometry-loss warmup: ramp from start_scale to 1.0 so the auxiliary term
    # cannot destabilise flow matching before the trunk has settled.
    loss_warmup_steps: int = 2000
    loss_warmup_start_scale: float = 0.1
    # The teacher target is the longest image dimension.
    image_size: int = 518
    query_grid: Optional[tuple[int, int]] = None
    projector_mid_channels: int = 256
    num_heads: int = 8
    max_queries: int = 4096
    transition_time_value: float = 5.0
    eps: float = 1e-6


class CrossAttentionBlock(nn.Module):
    """Pre-LN multi-head cross-attention + MLP (Q-Former style)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads != 0:
            # Fall back to a divisor of dim.
            for h in range(min(num_heads, dim), 0, -1):
                if dim % h == 0:
                    num_heads = h
                    break
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query: [B, N, D], context: [B, M, D]
        b, n, d = query.shape
        q = self.q_proj(self.norm_q(query))
        kv = self.norm_kv(context)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        query = query + self.out_proj(out)
        query = query + self.mlp(self.norm_mlp(query))
        return query


class GeometryActionResidualAdapter(nn.Module):
    """Cross-attend action tokens to the geometry transition."""

    def __init__(
        self,
        *,
        action_dim: int,
        geometry_dim: int,
        num_heads: int,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or geometry_dim <= 0:
            raise ValueError(
                f"Adapter dimensions must be positive, got action={action_dim} geometry={geometry_dim}."
            )
        if num_heads <= 0 or action_dim % num_heads != 0:
            raise ValueError(
                f"action_dim={action_dim} must be divisible by num_heads={num_heads}."
            )
        self.action_dim = int(action_dim)
        self.geometry_dim = int(geometry_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.action_dim // self.num_heads
        self.action_norm = nn.LayerNorm(self.action_dim, elementwise_affine=False, eps=1e-6)
        self.geometry_norm = nn.LayerNorm(self.geometry_dim, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(self.action_dim, self.action_dim, bias=False)
        self.k_proj = nn.Linear(self.geometry_dim, self.action_dim, bias=False)
        self.v_proj = nn.Linear(self.geometry_dim, self.action_dim, bias=False)
        self.out_proj = nn.Linear(self.action_dim, self.action_dim, bias=False)
        # ReZero: the untrained adapter is exactly the identity while this
        # scalar receives a gradient on the first optimizer step.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def prepare_geometry(
        self,
        z_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_delta.ndim != 3:
            raise ValueError(f"z_delta must be [B,N,C], got {tuple(z_delta.shape)}.")
        if z_delta.shape[-1] != self.geometry_dim:
            raise ValueError(
                f"Expected geometry width {self.geometry_dim}, got {z_delta.shape[-1]}."
            )
        batch_size, token_count, _ = z_delta.shape
        # Keep projection inputs aligned with the adapter parameter dtype.
        projection_dtype = self.k_proj.weight.dtype
        geometry = self.geometry_norm(z_delta.to(dtype=projection_dtype))
        key = self.k_proj(geometry).view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(geometry).view(
            batch_size, token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        return key, value, geometry.mean(dim=1)

    def _compute_residual(
        self,
        action_tokens: torch.Tensor,
        z_delta: Optional[torch.Tensor] = None,
        prepared_geometry: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"action_tokens must be [B,N,C], got {tuple(action_tokens.shape)}."
            )
        if action_tokens.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action width {self.action_dim}, got {action_tokens.shape[-1]}."
            )
        if (z_delta is None) == (prepared_geometry is None):
            raise ValueError(
                "Provide exactly one of z_delta or prepared_geometry to the action residual adapter."
            )
        key, value, geometry_pool = (
            self.prepare_geometry(z_delta)
            if prepared_geometry is None
            else prepared_geometry
        )
        batch_size, action_count, _ = action_tokens.shape
        if key.shape[0] != batch_size or value.shape != key.shape:
            raise ValueError(
                "Prepared geometry K/V must match the action batch and each other, got "
                f"key={tuple(key.shape)} value={tuple(value.shape)} action_batch={batch_size}."
            )
        if geometry_pool.shape != (batch_size, self.geometry_dim):
            raise ValueError(
                "Prepared geometry pool has the wrong shape: "
                f"expected {(batch_size, self.geometry_dim)}, got {tuple(geometry_pool.shape)}."
            )
        projection_dtype = self.q_proj.weight.dtype
        query_input = self.action_norm(action_tokens.to(dtype=projection_dtype))
        query = self.q_proj(query_input).view(
            batch_size, action_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        residual = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous().view(batch_size, action_count, self.action_dim)
        residual = self.out_proj(residual).to(dtype=action_tokens.dtype)
        return residual, query_input, geometry_pool

    def forward(
        self,
        action_tokens: torch.Tensor,
        z_delta: Optional[torch.Tensor] = None,
        prepared_geometry: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        residual, _, _ = self._compute_residual(
            action_tokens,
            z_delta=z_delta,
            prepared_geometry=prepared_geometry,
        )
        effective_gate = self.gate.to(dtype=residual.dtype) * float(residual_scale)
        return action_tokens + effective_gate * residual


class AlignmentProjector(nn.Module):
    """Project spatial z_Δ maps to 3D displacement fields (XYZ)."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Compatibility alias.
TransitionProjector = AlignmentProjector


class ScaleHead(nn.Module):
    """g_phi: recover full-scale teacher features from a unit-norm direction.

    Deliberately operates on the L2-normalised f_phi(z) so the only way to fit the
    target magnitude is to *predict* it from the direction. Regressing y_delta
    directly with an MSE instead lets the student satisfy the loss by inflating
    its own activations, which is the collapse the paper reports (Table 3: plain
    MSE scores 1648 FVD vs 243 for Angular+Scale).
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeometryTransitionAligner(nn.Module):
    """Learnable transition queries with layerwise causal updates."""

    def __init__(self, cfg: GeometryTransitionConfig):
        super().__init__()
        self.cfg = cfg
        # Spatial query grid for the current forward pass.
        self._active_query_grid: Optional[tuple[int, int]] = None
        placement = str(cfg.placement)
        if placement not in {"video", "action"}:
            raise ValueError(
                f"Unknown geometry query placement={cfg.placement!r}; expected video or action."
            )
        if str(cfg.vggt_view_mode) not in {"auto", "joint", "per_view"}:
            raise ValueError(
                f"Unknown vggt_view_mode={cfg.vggt_view_mode!r}; "
                "expected auto, joint, or per_view."
            )
        if placement == "action":
            if str(cfg.mode) != "bottleneck":
                raise ValueError("placement='action' requires mode='bottleneck'.")
            if str(cfg.action_condition) != "ref_only":
                raise ValueError(
                    "placement='action' requires action_condition='ref_only': the query "
                    "and action rows jointly attend text + reference + their own stream."
                )
            if str(cfg.inference_action_condition) != "ref_only":
                raise ValueError(
                    "placement='action' requires inference_action_condition='ref_only'."
                )
            if bool(cfg.action_residual_enabled):
                raise ValueError(
                    "placement='action' is already inside the action stream and is incompatible "
                    "with action_residual_enabled."
                )
        if str(cfg.inference_action_condition) not in {
            "transition_only",
            "transition_and_ref",
            "ref_only",
        }:
            raise ValueError(
                f"Unknown inference_action_condition={cfg.inference_action_condition!r}; "
                "expected transition_only, transition_and_ref, or ref_only."
            )
        if bool(cfg.action_residual_enabled):
            if str(cfg.mode) != "bottleneck":
                raise ValueError("action_residual_enabled requires mode='bottleneck'.")
            if str(cfg.action_condition) != "ref_only":
                raise ValueError(
                    "action_residual_enabled requires action_condition='ref_only' so the "
                    "baseline reference path remains intact."
                )
            if str(cfg.inference_action_condition) != "ref_only":
                raise ValueError(
                    "action_residual_enabled requires inference_action_condition='ref_only'."
                )
        if cfg.source_channels is None:
            raise ValueError(
                "GeometryTransitionConfig.source_channels must be set so transition params "
                "exist before optimizer creation."
            )
        channels = int(cfg.source_channels)
        heads = int(cfg.num_heads)
        # `objective` may be one name or several; several are summed.
        # Accepts "relation", ["relation", "geometry_forcing"], a Hydra ListConfig
        # (iterable of str), or "relation,geometry_forcing" so a bare CLI override
        # like objective=relation,geometry_forcing also works.
        raw_objective = cfg.objective
        if isinstance(raw_objective, str):
            names = [p for p in raw_objective.replace("+", ",").split(",") if p.strip()]
        else:
            names = [str(x) for x in raw_objective]
        known = {"relation", "geometry_forcing"}
        objectives: list[str] = []
        for name in names:
            name = str(name).strip()
            if name not in known:
                raise ValueError(
                    f"Unknown geometry objective {name!r}; "
                    "expected relation or geometry_forcing (or a list of them)."
                )
            if name not in objectives:
                objectives.append(name)
        if not objectives:
            raise ValueError("GeometryTransitionConfig.objective is empty.")
        self.objectives: tuple[str, ...] = tuple(objectives)
        self.objective = self.objectives[0] if len(self.objectives) == 1 else "+".join(self.objectives)
        self.query_tokens = nn.Parameter(torch.randn(int(cfg.max_queries), channels) * 0.02)
        self.updater = CrossAttentionBlock(channels, heads)
        # Each objective gets its OWN student projection: relation compares pairwise
        # distances at token width (so structure, not coordinates, is aligned --
        # this is what keeps it coordinate-frame free), while geometry_forcing must
        # land in the teacher's 2048-d space to regress y_delta. One shared head
        # cannot serve both output widths, and forcing them to would also couple
        # two independent objectives through one set of weights.
        self.alignment_projector = AlignmentProjector(
            in_channels=channels,
            out_channels=channels,
            mid_channels=int(cfg.projector_mid_channels),
        )
        if self.uses_forcing:
            self.forcing_projector = AlignmentProjector(
                in_channels=channels,
                out_channels=int(cfg.teacher_channels),
                mid_channels=int(cfg.projector_mid_channels),
            )
            hidden = cfg.scale_head_hidden or int(cfg.teacher_channels)
            self.scale_head = ScaleHead(int(cfg.teacher_channels), int(hidden))
        else:
            self.forcing_projector = None
            self.scale_head = None
        # Action-aware relevance: which visual tokens are coupled to the current
        # robot motion. Lazily sized on first use since the action width is not
        # known here.
        if bool(cfg.use_action_aware_weight):
            if cfg.action_channels is None:
                raise ValueError(
                    "GeometryTransitionConfig.action_channels must be set when "
                    "use_action_aware_weight=True so the relevance projections exist "
                    "before optimizer creation."
                )
            self.action_weight_video_proj = nn.Linear(channels, channels, bias=False)
            self.action_weight_action_proj = nn.Linear(int(cfg.action_channels), channels, bias=False)
        else:
            self.action_weight_video_proj = None
            self.action_weight_action_proj = None
        if bool(cfg.action_residual_enabled):
            if cfg.action_channels is None:
                raise ValueError(
                    "GeometryTransitionConfig.action_channels is required when "
                    "action_residual_enabled=True."
                )
            self.action_residual_adapter = GeometryActionResidualAdapter(
                action_dim=int(cfg.action_channels),
                geometry_dim=channels,
                num_heads=int(cfg.action_residual_num_heads),
                gate_init=float(cfg.action_residual_gate_init),
            )
        else:
            self.action_residual_adapter = None
        self.projector = self.alignment_projector
        self.transition_head = nn.Identity()
        # Lazy: VGGT is only needed to compute the training loss, never by
        # `update_delta`/inference. Loading it unconditionally here would force
        # every eval/inference process to have the `vggt` package importable and
        # would waste ~2.6GB of GPU memory that inference never touches.
        self.vggt: Optional[nn.Module] = None
        self._n_double_layers: Optional[int] = None
        # Keep the optimization step in the sidecar so loss warmup resumes exactly.
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long), persistent=True)

    @property
    def uses_relation(self) -> bool:
        return "relation" in self.objectives

    @property
    def uses_forcing(self) -> bool:
        return "geometry_forcing" in self.objectives

    @property
    def placement(self) -> str:
        return str(self.cfg.placement)

    @property
    def uses_teacher_supervision(self) -> bool:
        return bool(self.cfg.teacher_supervision_enabled)

    def apply_trainable_policy(self) -> None:
        """Enable only geometry parameters that participate in this experiment."""
        self.requires_grad_(False)
        self.query_tokens.requires_grad_(True)
        if self.placement == "video":
            self.updater.requires_grad_(True)
        if self.uses_teacher_supervision:
            self.alignment_projector.requires_grad_(True)
            if self.forcing_projector is not None:
                self.forcing_projector.requires_grad_(True)
            if self.scale_head is not None:
                self.scale_head.requires_grad_(True)
            if self.action_weight_video_proj is not None:
                self.action_weight_video_proj.requires_grad_(True)
                self.action_weight_action_proj.requires_grad_(True)
        if self.action_residual_adapter is not None:
            self.action_residual_adapter.requires_grad_(True)

    def set_num_double_layers(self, n: int) -> None:
        n = int(n)
        if n <= 0:
            raise ValueError(f"Number of double layers must be positive, got {n}.")
        self._n_double_layers = n

    def apply_action_residual(
        self,
        action_tokens: torch.Tensor,
        z_delta: Optional[torch.Tensor] = None,
        prepared_geometry: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.action_residual_adapter is None:
            return action_tokens
        residual_scale = (
            1.0
            if self.training
            else float(self.cfg.action_residual_inference_scale)
        )
        return self.action_residual_adapter(
            action_tokens,
            z_delta=z_delta,
            prepared_geometry=prepared_geometry,
            residual_scale=residual_scale,
        )

    def should_run_action_residual_at_inference(self) -> bool:
        adapter = self.action_residual_adapter
        if adapter is None:
            return False
        return bool(adapter.gate.detach().ne(0).any().item())

    def prepare_action_residual_geometry(
        self,
        z_delta: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.action_residual_adapter is None:
            return None
        return self.action_residual_adapter.prepare_geometry(z_delta)

    @property
    def train_step(self) -> int:
        return int(self._train_step.item())

    def loss_warmup_scale(self) -> float:
        """Ramp the geometry loss in so it cannot destabilise early flow matching."""
        warmup = int(self.cfg.loss_warmup_steps)
        if warmup <= 0:
            return 1.0
        start = float(self.cfg.loss_warmup_start_scale)
        frac = min(1.0, float(self.train_step) / float(warmup))
        return start + (1.0 - start) * frac

    def advance_step(self, *, training: bool) -> bool:
        """Advance the optimization counter before a transformer forward."""
        if training:
            self._train_step += 1
        return False

    def _ensure_vggt(self) -> nn.Module:
        """Load the frozen VGGT teacher on first use (see `self.vggt` in __init__).

        Placed on the same device as the rest of this module so a lazy load
        after `.to(device)` still lands correctly.
        """
        if not self.uses_teacher_supervision:
            raise RuntimeError(
                "VGGT was requested while teacher_supervision_enabled=false."
            )
        if self.vggt is None:
            vggt = self._load_vggt(self.cfg)
            device = self.query_tokens.device
            dtype = next(vggt.parameters()).dtype
            self.vggt = vggt.to(device=device, dtype=dtype)
        return self.vggt

    @staticmethod
    def _load_vggt(cfg: GeometryTransitionConfig) -> nn.Module:
        """Load the frozen VGGT teacher, keeping only the aggregator trunk.

        Resolution goes through `ensure_vggt_importable`, which prefers an
        explicitly configured checkout and otherwise adds the pinned submodule
        at third_party/vggt to sys.path.

        The prediction heads (camera / depth / point / track) are dropped after
        loading: relation supervision reads aggregator patch features only, and the
        two DPT heads plus the track head are a large block of frozen weights that
        would otherwise sit in GPU memory for every training step.
        """
        backend = str(getattr(cfg, "vggt_backend", "vggt")).strip().lower()
        if backend == "vggt_omega":
            return GeometryTransitionAligner._load_vggt_omega(cfg)
        if backend != "vggt":
            raise ValueError(
                f"Unknown geometry_transition.vggt_backend={backend!r}; "
                "expected 'vggt' or 'vggt_omega'."
            )

        from .vggt_imports import ensure_vggt_importable

        ensure_vggt_importable(cfg.vggt_repo_path)
        try:
            from vggt.models.vggt import VGGT
        except Exception as exc:  # noqa: BLE001 - surface the resolution attempt
            searched = cfg.vggt_repo_path or "third_party/vggt (pinned submodule)"
            raise ImportError(
                "Could not import VGGT for geometry-transition supervision "
                f"(searched: {searched}). Initialize the pinned source with "
                "`git submodule update --init --recursive`, or set "
                f"geometry_transition.vggt_repo_path to a checkout. Cause: {exc}"
            ) from exc

        model = VGGT.from_pretrained(cfg.vggt_model_id)
        if bool(cfg.vggt_drop_heads):
            for head in ("camera_head", "depth_head", "point_head", "track_head"):
                if getattr(model, head, None) is not None:
                    setattr(model, head, None)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        # `from_pretrained` reads embed_dim from the hub config, which this local
        # checkout cannot see ahead of time. Fail at construction, not on the
        # first training step, if teacher_channels was set from a stale
        # assumption (aggregator emits 2 * embed_dim per cached layer -- frame
        # and global intermediates concatenated).
        aggregator = getattr(model, "aggregator", None)
        patch_embed = getattr(aggregator, "patch_embed", None) if aggregator is not None else None
        embed_dim = getattr(patch_embed, "embed_dim", None)
        if embed_dim is not None and int(cfg.teacher_channels) != 2 * int(embed_dim):
            raise ValueError(
                f"geometry_transition.teacher_channels={cfg.teacher_channels} does not match "
                f"the loaded VGGT aggregator (embed_dim={embed_dim} -> expected "
                f"{2 * int(embed_dim)}). Update the config to match the checkpoint."
            )
        return model

    @staticmethod
    def _load_vggt_omega(cfg: GeometryTransitionConfig) -> nn.Module:
        """Load the official VGGT-Omega aggregator from a local checkpoint.

        Omega exposes the same aggregator return contract as the original VGGT,
        but uses the package name ``vggt_omega`` and a direct state-dict load.
        Only relation/geometry-forcing supervision is supported: Omega predicts
        depth, not a dense coordinate head.
        """
        from .vggt_imports import ensure_vggt_omega_importable

        ensure_vggt_omega_importable(cfg.vggt_repo_path)
        try:
            from vggt_omega.models import VGGTOmega
        except Exception as exc:  # noqa: BLE001
            searched = cfg.vggt_repo_path or "third_party/vggt-omega (vendored)"
            raise ImportError(
                "Could not import VGGT-Omega for geometry-transition supervision "
                f"(searched: {searched}). Clone "
                "https://github.com/facebookresearch/vggt-omega or set "
                f"geometry_transition.vggt_repo_path to a checkout. Cause: {exc}"
            ) from exc

        if str(cfg.vggt_model_id).startswith("facebook/"):
            raise ValueError(
                "VGGT-Omega requires a local checkpoint path in "
                "geometry_transition.vggt_model_id; download the official model.pt first."
            )
        checkpoint = Path(cfg.vggt_model_id).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint}")

        # Load the state dict before discarding unused prediction heads.
        model = VGGTOmega(
            enable_camera=True,
            enable_depth=True,
            enable_alignment=False,
            autocast=True,
        )
        if checkpoint.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "VGGT-Omega .safetensors checkpoints require safetensors."
                ) from exc
            state = load_file(str(checkpoint), device="cpu")
        else:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                for key in ("state_dict", "model", "model_state_dict"):
                    candidate = payload.get(key)
                    if isinstance(candidate, dict) and candidate and all(
                        isinstance(value, torch.Tensor) for value in candidate.values()
                    ):
                        payload = candidate
                        break
            state = payload
        if not isinstance(state, dict) or not state or not all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise ValueError(
                f"Unsupported VGGT-Omega checkpoint payload in {checkpoint}; "
                "expected a tensor state dict."
            )
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys in VGGT-Omega checkpoint: "
                + ", ".join(incompatible.unexpected_keys[:8])
            )
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(("camera_head.", "dense_head.", "text_alignment_head."))
        ]
        if missing:
            raise RuntimeError(
                "VGGT-Omega checkpoint is missing aggregator parameters: "
                + ", ".join(missing[:8])
            )
        if bool(cfg.vggt_drop_heads):
            for head in ("camera_head", "dense_head", "text_alignment_head"):
                if getattr(model, head, None) is not None:
                    setattr(model, head, None)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        aggregator = getattr(model, "aggregator", None)
        patch_embed = getattr(aggregator, "patch_embed", None) if aggregator is not None else None
        embed_dim = getattr(patch_embed, "embed_dim", None)
        if embed_dim is not None and int(cfg.teacher_channels) != 2 * int(embed_dim):
            raise ValueError(
                f"geometry_transition.teacher_channels={cfg.teacher_channels} does not match "
                f"the loaded vggt_omega aggregator (embed_dim={embed_dim} -> expected "
                f"{2 * int(embed_dim)})."
            )
        return model

    def init_delta_queries(
        self,
        batch_size: int,
        num_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Learnable spatial query tokens z_Δ_init [B, N, D]."""
        max_q = int(self.query_tokens.shape[0])
        if num_tokens <= max_q:
            base = self.query_tokens[:num_tokens]
        else:
            # Linear upsample along token axis when N exceeds the table.
            base = F.interpolate(
                self.query_tokens.T.unsqueeze(0),
                size=num_tokens,
                mode="linear",
                align_corners=False,
            ).squeeze(0).T
        return base.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    def use_joint_vggt_views(self, num_views: int) -> bool:
        """Resolve the configured joint/per-view mode."""
        if int(num_views) <= 1:
            return False
        mode = str(self.cfg.vggt_view_mode)
        if mode == "joint":
            return True
        if mode == "per_view":
            return False
        return bool(self.cfg.vggt_multiview)

    def build_transition_img_ids(
        self,
        ref_img_ids: torch.Tensor,
        *,
        time_value: Optional[float] = None,
    ) -> torch.Tensor:
        ids = ref_img_ids.clone()
        ids[..., 0] = float(self.cfg.transition_time_value if time_value is None else time_value)
        return ids

    def update_delta(
        self,
        layer_idx: int,
        z_delta: torch.Tensor,
        h_ref: torch.Tensor,
        h_tgt: torch.Tensor,
        txt: torch.Tensor,
    ) -> torch.Tensor:
        """Causal Q-Former update from the reference stream and text."""
        del layer_idx, h_tgt
        if h_ref.shape[:2] != z_delta.shape[:2]:
            raise ValueError(
                f"z/h_ref batch/token mismatch: "
                f"z={tuple(z_delta.shape)} ref={tuple(h_ref.shape)}"
            )
        # This module is built in fp32 while the MoT stack runs in bf16. Training
        # hides the mismatch behind autocast, but the inference path has no
        # autocast, so align the inputs with our own parameter dtype here and
        # cast the result back to the stream dtype.
        out_dtype = z_delta.dtype
        param_dtype = next(self.updater.parameters()).dtype
        if param_dtype != out_dtype:
            z_delta = z_delta.to(param_dtype)
            h_ref = h_ref.to(param_dtype)
            txt = txt.to(param_dtype)
        context = torch.cat([h_ref, txt], dim=1)
        return self.updater(z_delta, context).to(out_dtype)

    def update_delta_causal(
        self,
        layer_idx: int,
        z_delta: torch.Tensor,
        h_ref: torch.Tensor,
        txt: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility alias for the causal query update."""
        del layer_idx  # Shared Q-Former parameters are intentionally reused at all layers.
        if h_ref.shape[:2] != z_delta.shape[:2]:
            raise ValueError(
                f"z/h_ref batch/token mismatch: z={tuple(z_delta.shape)} ref={tuple(h_ref.shape)}"
            )
        if txt.ndim != 3 or txt.shape[0] != z_delta.shape[0]:
            raise ValueError(
                f"Text context must be [B,N,C] with matching batch, got {tuple(txt.shape)}."
            )
        out_dtype = z_delta.dtype
        param_dtype = next(self.updater.parameters()).dtype
        query = z_delta.to(param_dtype)
        context = torch.cat([h_ref.to(param_dtype), txt.to(param_dtype)], dim=1)
        return self.updater(query, context).to(out_dtype)

    def make_layer_hook(self, *, cond_len: int, target_len: int, transition_len: int):
        """Hook for MoT double blocks: refine z_Δ slots in-place after each layer."""

        def hook(layer_idx: int, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
            if transition_len <= 0:
                return img
            expected = cond_len + target_len + transition_len
            if img.shape[1] < expected:
                raise ValueError(
                    f"img tokens {img.shape[1]} shorter than expected layout "
                    f"ref({cond_len})+tgt({target_len})+z({transition_len})"
                )
            h_ref_full = img[:, :cond_len]
            h_tgt_full = img[:, cond_len : cond_len + target_len]
            z_old = img[:, cond_len + target_len : cond_len + target_len + transition_len]
            h_ref = self._resize_spatial_tokens(h_ref_full, transition_len)
            z_new = self.update_delta(layer_idx, z_old, h_ref, h_tgt_full, txt)
            img_new = img.clone()
            img_new[:, cond_len + target_len : cond_len + target_len + transition_len] = z_new
            return img_new

        return hook

    def _resize_spatial_tokens(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resize [B,N,C] spatial tokens by deterministic 2-D average pooling."""
        if int(tokens.shape[1]) == int(target_len):
            return tokens
        src_grid = GeometryTransitionAligner._infer_square_grid(int(tokens.shape[1]))
        dst_grid = self._active_query_grid or GeometryTransitionAligner._infer_square_grid(
            int(target_len)
        )
        if int(dst_grid[0] * dst_grid[1]) != int(target_len):
            raise ValueError(
                f"Active query grid {dst_grid} does not match {target_len} tokens."
            )
        b, _, c = tokens.shape
        x = rearrange(tokens, "b (h w) c -> b c h w", h=src_grid[0], w=src_grid[1])
        x = F.adaptive_avg_pool2d(x, output_size=dst_grid)
        return rearrange(x, "b c h w -> b (h w) c")

    @staticmethod
    def _as_btc_hw(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected a 5D video tensor, got {tuple(images.shape)}")
        if images.shape[1] in (1, 3):
            return rearrange(images, "b c t h w -> b t c h w")
        if images.shape[2] in (1, 3):
            return images
        raise ValueError(f"Cannot infer video layout from shape {tuple(images.shape)}")

    @staticmethod
    def _infer_square_grid(num_tokens: int) -> tuple[int, int]:
        h = int(num_tokens ** 0.5)
        if h * h != int(num_tokens):
            raise ValueError(f"Cannot infer square grid from {num_tokens} tokens; pass source_grid explicitly.")
        return h, h

    @staticmethod
    def _camera_view_rects(views: int, tiling: str) -> list[tuple[float, float, float, float]]:
        """Per-view (top, left, height, width) as fractions of the tiled frame.

        Single source of truth for the dataset's layout: `split_camera_views`
        slices pixels with it and `compute_teacher_features` places teacher
        features with it, so the two can no longer disagree about where a view
        lives.

        Fractions rather than pixels because the same rects must address both the
        image (e.g. 288x256) and the token grid (e.g. 18x16). For the RoboTwin
        mosaic both supported layouts (`compact_288x256` 192/96 and
        `384x320` 256/128, see `robot_video_dataset._robotwin_camera_sizes`)
        share the same 2/3 : 1/3 split, so the layout name is not needed here.
        """
        if views <= 1:
            return [(0.0, 0.0, 1.0, 1.0)]
        if tiling == "horizontal":
            return [(0.0, i / views, 1.0, 1.0 / views) for i in range(views)]
        if tiling == "vertical":
            return [(i / views, 0.0, 1.0 / views, 1.0) for i in range(views)]
        if tiling == "robotwin":
            # [cam_high ; (left_wrist | right_wrist)] stacked vertically.
            if views != 3:
                raise ValueError(f"camera_tiling='robotwin' requires num_camera_views=3, got {views}.")
            return [
                (0.0, 0.0, 2.0 / 3.0, 1.0),
                (2.0 / 3.0, 0.0, 1.0 / 3.0, 0.5),
                (2.0 / 3.0, 0.5, 1.0 / 3.0, 0.5),
            ]
        raise ValueError(f"Unknown camera_tiling={tiling!r}.")

    @staticmethod
    def _rect_to_bounds(
        rect: tuple[float, float, float, float],
        height: int,
        width: int,
        *,
        what: str,
        strict: bool = True,
    ) -> tuple[int, int, int, int]:
        """Resolve a fractional rect to integer [r0, r1, c0, c1] bounds.

        Refuses to round: a rect that does not land on an integer boundary would
        silently shift a view's features off the tokens that observe it, which is
        exactly the failure this indirection exists to prevent.

        strict=False (coarse query grids only): the layout fractions rarely
        divide a small query grid evenly, so edges round to the nearest token.
        The teacher feature map is interpolated into whatever region this
        returns, so a +-1 coarse-token shift is an acceptable approximation.
        Pixel-level splitting always uses strict=True.
        """
        top, left, rect_h, rect_w = rect

        def edge(frac: float, total: int, axis: str) -> int:
            value = frac * total
            rounded = int(round(value))
            if strict and abs(value - rounded) > 1e-6:
                raise ValueError(
                    f"{what} {axis} extent {total} is incompatible with the camera "
                    f"layout: fraction {frac:g} maps to {value:g}, not an integer. "
                    "Choose a resolution/token grid that divides the layout evenly."
                )
            return rounded

        r0 = edge(top, height, "height")
        r1 = edge(top + rect_h, height, "height")
        c0 = edge(left, width, "width")
        c1 = edge(left + rect_w, width, "width")
        if r1 <= r0 or c1 <= c0:
            raise ValueError(f"{what} rect {rect} resolved to an empty region.")
        return r0, r1, c0, c1

    def split_camera_views(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Undo the dataset's multi-camera tiling into a list of single-view clips.

        The dataset concatenates camera views into one frame (LIBERO: agentview
        and wrist side by side; RoboTwin: a 3-view mosaic). Feeding that collage
        to a 3D reconstructor is meaningless — it is not a physically realizable
        image — so each view is reconstructed separately.

        Args:
            images: [B, T, 3, H, W]
        Returns:
            list of [B, T, 3, h, w], one per camera view.
        """
        views = int(self.cfg.num_camera_views)
        tiling = str(self.cfg.camera_tiling)
        if views <= 1 or tiling == "none":
            return [images]
        h, w = int(images.shape[-2]), int(images.shape[-1])
        out: list[torch.Tensor] = []
        for rect in self._camera_view_rects(views, tiling):
            r0, r1, c0, c1 = self._rect_to_bounds(rect, h, w, what="Image")
            out.append(images[..., r0:r1, c0:c1])
        return out

    def _geometry_view_clips(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return endpoint clips per camera as [B,2,3,H,W].

        A 5D input is a tiled model frame. A 6D input carries the
        original camera views independently as [B,V,C,T,H,W], avoiding the
        resolution loss inherent in the compact RoboTwin mosaic.
        """
        if images.ndim == 5:
            frames = self._as_btc_hw(images)
            endpoints = torch.stack([frames[:, 0], frames[:, -1]], dim=1)
            return self.split_camera_views(endpoints)
        if images.ndim != 6:
            raise ValueError(
                "Geometry teacher input must be a tiled 5D video or separate-view "
                f"[B,V,C,T,H,W] tensor, got {tuple(images.shape)}."
            )
        expected_views = int(self.cfg.num_camera_views)
        if int(images.shape[1]) != expected_views:
            raise ValueError(
                f"Geometry teacher input has {images.shape[1]} views; expected "
                f"num_camera_views={expected_views}."
            )
        if int(images.shape[2]) not in (1, 3):
            raise ValueError(
                "Separate-view geometry teacher input must be [B,V,C,T,H,W] with "
                f"C=1 or 3, got {tuple(images.shape)}."
            )
        return [
            torch.stack([images[:, view_idx, :, 0], images[:, view_idx, :, -1]], dim=1)
            for view_idx in range(expected_views)
        ]

    @staticmethod
    def _flow_worker_count(batch: int) -> int:
        """Flow threads for THIS rank, budgeted against the whole node.

        Every rank on the node runs its own copy of this, on top of its dataloader
        workers, so a per-rank pool sized to the batch oversubscribes the CPU by
        roughly (ranks x workers). Divide the cores by the local world size and
        leave the dataloader room.
        """
        import os

        cores = os.cpu_count() or 8
        local_ranks = max(
            1,
            int(
                os.environ.get("LOCAL_WORLD_SIZE")
                or os.environ.get("NPROC_PER_NODE")
                or os.environ.get("WORLD_SIZE")
                or 1
            ),
        )
        budget = max(1, cores // (local_ranks * 2))
        return max(1, min(batch, budget, 4))

    @classmethod
    def _flow_pool(cls, workers: int):
        """Process-wide thread pool, created once and reused across steps."""
        from concurrent.futures import ThreadPoolExecutor

        pool = getattr(cls, "_FLOW_POOL", None)
        if pool is None or getattr(cls, "_FLOW_POOL_SIZE", 0) != workers:
            if pool is not None:
                pool.shutdown(wait=True)
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="geomflow")
            cls._FLOW_POOL = pool
            cls._FLOW_POOL_SIZE = workers
        return pool

    @staticmethod
    @torch.no_grad()
    def _dense_flow_tgt_to_ref(ref_u8: "np.ndarray", tgt_u8: "np.ndarray") -> "np.ndarray":
        """Farneback flow from target to reference → [H, W, 2] in pixels.

        Direction matters: to resample the reference onto the target's grid we need,
        for each *target* pixel, where it came from in the reference. That is
        exactly flow(tgt -> ref), so the backward-warp below has one source sample
        per destination pixel and cannot leave holes. Computing flow(ref -> tgt)
        and forward-scattering would leave gaps wherever the motion diverges.
        """
        import cv2
        import numpy as np

        # Farneback internally forks over every core by default. With 8 ranks on one
        # node, each already running 24 dataloader workers and calling this from a
        # thread pool, that nests three levels of parallelism and oversubscribes the
        # CPU badly enough to look like a hang. One thread per call; the outer pool
        # provides the parallelism.
        if cv2.getNumThreads() != 1:
            cv2.setNumThreads(1)

        g_ref = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2GRAY)
        g_tgt = cv2.cvtColor(tgt_u8, cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(
            g_tgt, g_ref, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        ).astype(np.float32)

    @torch.no_grad()
    def warp_ref_onto_target(
        self,
        ref: torch.Tensor,
        tgt: torch.Tensor,
        *,
        input_range: str = "tanh",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Motion-compensate `ref` onto `tgt`'s pixel grid.

        Establishes the patch correspondence that per-patch differencing needs:
        after this, patch p of the warped reference and patch p of the target
        observe the same surface wherever the flow is correct, so
        y_tgt,p - y_warped_ref,p is a difference of *the same* surface rather than
        of two unrelated ones.

        What survives the subtraction is therefore residual geometric change that
        2D motion cannot explain -- occlusion/disocclusion, genuine depth change,
        non-rigid deformation, and flow error -- not raw appearance change.

        Args:
            ref, tgt: [B, 3, H, W], same range (`input_range`).
        Returns:
            warped_ref: [B, 3, H, W] in the same range as the input.
            valid:      [B, 1, H, W] float, 1 where the sample fell inside `ref`.
        """
        import numpy as np

        if ref.shape != tgt.shape:
            raise ValueError(f"ref/tgt shape mismatch: {tuple(ref.shape)} vs {tuple(tgt.shape)}")
        if str(self.cfg.warp_method) == "none":
            return ref, ref.new_ones(ref.shape[0], 1, *ref.shape[-2:])

        b, _, h, w = ref.shape
        # Flow needs uint8 RGB on the CPU; keep the originals for the resample so
        # the warp itself stays in the caller's dtype and precision.
        def _to_u8(x: torch.Tensor) -> "np.ndarray":
            x = x.detach().float()
            x = (x + 1.0) * 0.5 if input_range == "tanh" else x
            x = x.clamp(0, 1).mul(255.0).round()
            return x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        ref_u8, tgt_u8 = _to_u8(ref), _to_u8(tgt)
        workers = self._flow_worker_count(b)
        if workers > 1:
            # cv2 releases the GIL, so threads give real parallelism here. The pool
            # is cached: building and tearing one down per view per step, across
            # every rank, costs more than the flow on a busy node.
            flow_list = list(
                self._flow_pool(workers).map(
                    lambda i: self._dense_flow_tgt_to_ref(ref_u8[i], tgt_u8[i]),
                    range(b),
                )
            )
        else:
            flow_list = [self._dense_flow_tgt_to_ref(ref_u8[i], tgt_u8[i]) for i in range(b)]
        flows = np.stack(flow_list, axis=0)
        flow = torch.from_numpy(flows).to(device=ref.device, dtype=torch.float32)  # [B,H,W,2]

        # Absolute source coords, then to grid_sample's [-1, 1] convention.
        yy, xx = torch.meshgrid(
            torch.arange(h, device=ref.device, dtype=torch.float32),
            torch.arange(w, device=ref.device, dtype=torch.float32),
            indexing="ij",
        )
        src_x = xx[None] + flow[..., 0]
        src_y = yy[None] + flow[..., 1]
        # align_corners=False maps pixel centre i to (2i+1)/size - 1.
        gx = (2.0 * src_x + 1.0) / float(w) - 1.0
        gy = (2.0 * src_y + 1.0) / float(h) - 1.0
        grid = torch.stack([gx, gy], dim=-1)

        warped = F.grid_sample(
            ref.float(), grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        ).to(dtype=ref.dtype)
        # `border` padding keeps the warp finite, but a sample taken outside the
        # source has no real correspondence -- flag it so the loss can drop it.
        inside = (
            (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
        ).unsqueeze(1).to(dtype=ref.dtype)
        return warped, inside

    def _vggt_patch_size(self) -> tuple[int, int]:
        """Return VGGT's (height, width) patch size from the loaded backbone."""
        aggregator = getattr(self.vggt, "aggregator", None)
        patch_embed = getattr(aggregator, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 14)
        if isinstance(patch_size, (tuple, list)):
            return int(patch_size[0]), int(patch_size[1])
        patch_size = int(patch_size)
        return patch_size, patch_size

    def _vggt_patch_grid(self, images: torch.Tensor) -> tuple[int, int]:
        """Return the rectangular VGGT patch grid for a prepared image batch."""
        patch_h, patch_w = self._vggt_patch_size()
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if height % patch_h != 0 or width % patch_w != 0:
            raise RuntimeError(
                f"VGGT image size {(height, width)} is not divisible by patch size "
                f"{(patch_h, patch_w)}."
            )
        return height // patch_h, width // patch_w

    def _reshape_vggt_tokens(
        self,
        tokens: torch.Tensor,
        prepared: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Reshape VGGT tokens using the actual rectangular patch grid."""
        if tokens.ndim != 4:
            raise RuntimeError(f"Expected VGGT tokens [B,T,P,D], got {tuple(tokens.shape)}.")
        b, t, num_patches, dim = tokens.shape
        patch_grid_h, patch_grid_w = self._vggt_patch_grid(prepared)
        expected = patch_grid_h * patch_grid_w
        if num_patches != expected:
            raise RuntimeError(
                f"VGGT returned {num_patches} spatial patches for grid "
                f"{(patch_grid_h, patch_grid_w)} (expected {expected})."
            )
        maps = tokens.reshape(b * t, patch_grid_h, patch_grid_w, dim).permute(0, 3, 1, 2)
        return maps, patch_grid_h, patch_grid_w

    def _prepare_vggt_images(self, images: torch.Tensor, *, input_range: str = "tanh") -> torch.Tensor:
        """Normalize + aspect-preserving resize for VGGT.

        The teacher preprocessing scales the longest dimension to ``image_size``
        and rounds both dimensions to its patch size (14 for original VGGT, 16
        for VGGT-Omega). Keeping the view's aspect ratio is important for
        geometry; the downstream feature scatter handles a rectangular patch
        grid explicitly.
        """
        images = self._as_btc_hw(images)
        b, t, c, _, _ = images.shape
        if c != 3:
            raise ValueError(f"VGGT expects RGB images, got C={c}.")
        if t < 2:
            raise ValueError("Need at least two frames for geometry transition.")
        vggt = self._ensure_vggt()
        vggt_param = next(vggt.parameters())
        vggt_device = vggt_param.device
        vggt_dtype = vggt_param.dtype
        image_size = int(self.cfg.image_size)
        patch_embed = getattr(getattr(vggt, "aggregator", None), "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = (int(patch_size[0]), int(patch_size[1]))
        else:
            default_patch = 16 if str(self.cfg.vggt_backend) == "vggt_omega" else 14
            patch_h = patch_w = int(patch_size or default_patch)
        if image_size % patch_h != 0 or image_size % patch_w != 0:
            backend = str(self.cfg.vggt_backend).strip().lower()
            raise ValueError(
                f"geometry_transition.image_size={image_size} is not divisible by "
                f"{backend} patch size {(patch_h, patch_w)}; use e.g. 512 for VGGT-Omega."
            )
        # Normalize/resize in float32, then match frozen teacher dtype (often bf16).
        images = images.to(device=vggt_device, dtype=torch.float32, non_blocking=True)
        # Explicit range declaration. Probing `images.min() < 0` guesses wrong on
        # any batch whose pixels happen to be all-positive.
        if input_range == "tanh":
            images = (images + 1.0) * 0.5
        elif input_range != "unit":
            raise ValueError(f"input_range must be 'tanh' or 'unit', got {input_range!r}.")
        images = images.clamp(0, 1)
        images = rearrange(images, "b t c h w -> (b t) c h w")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        target = int(self.cfg.image_size)
        patch_h, patch_w = self._vggt_patch_size()
        scale = target / float(max(height, width))
        resized_h = max(patch_h, int(round(height * scale / patch_h)) * patch_h)
        resized_w = max(patch_w, int(round(width * scale / patch_w)) * patch_w)
        images = F.interpolate(
            images,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        images = rearrange(images, "(b t) c h w -> b t c h w", b=b, t=t)
        return images.to(dtype=vggt_dtype)

    @torch.no_grad()
    def compute_teacher_features(
        self,
        images: torch.Tensor,
        *,
        grid: tuple[int, int],
        input_range: str = "tanh",
    ) -> torch.Tensor:
        """Frozen-VGGT patch features for the reference and target frames.

        Features, not `world_points`, are the supervision source. Pairwise feature
        distances are correspondence-free and survive VGGT's per-forward
        up-to-scale normalization, whereas a same-pixel-index coordinate difference
        is neither a displacement field nor comparable across samples.

        Both frames go through a single VGGT forward per camera view: VGGT is
        frame-1-canonical and re-normalizes scale per call, so features from
        separate calls do not share a frame of reference.

        Args:
            images: A tiled [B,C,T,H,W]/[B,T,C,H,W] clip, or separate camera
                views [B,V,C,T,H,W]; frame 0 is reference and frame -1 target.
            grid:   (H_l, W_l) token grid of the student z_Δ map, spanning all
                camera views as laid out by the dataset.
        Returns:
            [B, 2, H_l, W_l, D] teacher features (index 0 = ref, 1 = tgt).
        """
        views = self._geometry_view_clips(images)
        n_views = len(views)
        grid_h, grid_w = int(grid[0]), int(grid[1])
        rects = self._camera_view_rects(int(self.cfg.num_camera_views), str(self.cfg.camera_tiling))
        if len(rects) != n_views:
            raise RuntimeError(f"Got {n_views} views but {len(rects)} layout rects.")

        if self.use_joint_vggt_views(n_views):
            return self._teacher_features_multiview(
                views,
                rects=rects,
                grid_h=grid_h,
                grid_w=grid_w,
                input_range=input_range,
            )

        # Each view is scattered back into the exact token region that observes it,
        # using the same rects that sliced the pixels. An equal-width split would
        # misplace any layout that is not a plain horizontal strip (the RoboTwin
        # mosaic puts cam_high across the full width of the top 2/3), which shifts
        # every teacher token off its student counterpart.
        out: Optional[torch.Tensor] = None
        for view, rect in zip(views, rects):
            r0, r1, c0, c1 = self._rect_to_bounds(
                rect, grid_h, grid_w, what="Token grid", strict=False
            )
            prepared = self._prepare_vggt_images(view, input_range=input_range)
            tokens = self._vggt_patch_tokens(prepared)  # [B, 2, P, D]
            maps, _, _ = self._reshape_vggt_tokens(tokens, prepared)
            b, t, _, dim = tokens.shape
            maps = F.interpolate(
                maps.float(), size=(r1 - r0, c1 - c0), mode="bilinear", align_corners=False
            )
            maps = maps.permute(0, 2, 3, 1).reshape(b, t, r1 - r0, c1 - c0, dim)
            if out is None:
                out = maps.new_zeros((b, t, grid_h, grid_w, dim))
            out[:, :, r0:r1, c0:c1] = maps
        if out is None:
            raise RuntimeError("No camera views produced teacher features.")
        return out

    @torch.no_grad()
    def compute_teacher_delta(
        self,
        images: torch.Tensor,
        *,
        grid: tuple[int, int],
        input_range: str = "tanh",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Warped per-patch teacher difference y_Δ = y_tgt - y_ref → [B, H_l, W_l, D].

        The reference frame is motion-compensated onto the target grid first (see
        `warp_ref_onto_target`), so patch p of both frames observes the same
        surface and the difference is well-posed. Feature-space differencing (not
        feature differencing is what makes this valid: VGGT's aggregator applies
        global attention across all S inputs, so both frames' features already live
        in one shared space, whereas independent coordinate regressions do not.

        Returns:
            y_delta: [B, H_l, W_l, D]
            valid:   [B, H_l, W_l] float in [0, 1], or None when warping is off.
                     Fractional values mean the patch only partly had valid
                     correspondence after pooling to the token grid.
        """
        grid_h, grid_w = int(grid[0]), int(grid[1])

        do_warp = bool(self.cfg.warp_ref_to_target) and str(self.cfg.warp_method) != "none"
        views = self._geometry_view_clips(images)
        n_views = len(views)
        rects = self._camera_view_rects(int(self.cfg.num_camera_views), str(self.cfg.camera_tiling))
        if len(rects) != n_views:
            raise RuntimeError(f"Got {n_views} views but {len(rects)} layout rects.")

        # Warp per view: each camera has its own motion field, so a single flow
        # over the collage would blend unrelated viewpoints across the tile seam.
        # Validity is scattered into each view's own token rect for the same reason
        # the features are: an equal-width split misplaces the RoboTwin mosaic.
        pairs: list[torch.Tensor] = []
        valid: Optional[torch.Tensor] = None
        for view, rect in zip(views, rects):
            r, t = view[:, 0], view[:, 1]
            if do_warp:
                r, inside = self.warp_ref_onto_target(r, t, input_range=input_range)
                r0, r1, c0, c1 = self._rect_to_bounds(rect, grid_h, grid_w, what="Token grid")
                # Average-pool occupancy to the token grid: a patch is partially
                # valid if only some of its pixels had correspondence.
                pooled = F.adaptive_avg_pool2d(inside.float(), (r1 - r0, c1 - c0))[:, 0]
                if valid is None:
                    valid = pooled.new_zeros((pooled.shape[0], grid_h, grid_w))
                valid[:, r0:r1, c0:c1] = pooled
            pairs.append(torch.stack([r, t], dim=1))  # [B,2,3,h,w]

        if self.use_joint_vggt_views(n_views):
            teacher = self._teacher_features_multiview(
                pairs, rects=rects, grid_h=grid_h, grid_w=grid_w, input_range=input_range
            )
        else:
            teacher = None
            for pair, rect in zip(pairs, rects):
                r0, r1, c0, c1 = self._rect_to_bounds(rect, grid_h, grid_w, what="Token grid")
                prepared = self._prepare_vggt_images(pair, input_range=input_range)
                tokens = self._vggt_patch_tokens(prepared)  # [B,2,P,D]
                maps, _, _ = self._reshape_vggt_tokens(tokens, prepared)
                b, t, _, dim = tokens.shape
                maps = F.interpolate(
                    maps.float(), size=(r1 - r0, c1 - c0), mode="bilinear", align_corners=False
                )
                maps = maps.permute(0, 2, 3, 1).reshape(b, t, r1 - r0, c1 - c0, dim)
                if teacher is None:
                    teacher = maps.new_zeros((b, t, grid_h, grid_w, dim))
                teacher[:, :, r0:r1, c0:c1] = maps
            if teacher is None:
                raise RuntimeError("No camera views produced teacher features.")

        y_delta = teacher[:, 1] - teacher[:, 0]  # [B, H_l, W_l, D]
        return y_delta, valid

    def _teacher_features_multiview(
        self,
        views: list[torch.Tensor],
        *,
        rects: list[tuple[float, float, float, float]],
        grid_h: int,
        grid_w: int,
        input_range: str,
    ) -> torch.Tensor:
        """One VGGT forward over all (view, frame) images → [B, 2, H_l, W_l, D].

        VGGT's S axis means "images of the same scene", and its aggregator
        alternates frame-wise self-attention with *global* self-attention across
        all S. Packing every view and both frames into one forward therefore
        yields features that already share a coordinate frame and a single scale
        normalisation, which is what makes cross-view and cross-frame feature
        arithmetic (y_tgt - y_ref) meaningful.

        Per-view forwards cannot give this: VGGT canonicalises on the first frame
        of each call and re-normalises scale per call, so view A's features and
        view B's features live in unrelated frames.

        Ordering note: S is packed view-major, [v0_ref, v0_tgt, v1_ref, v1_tgt,
        ...], so frame 0 of the sequence is a reference frame — VGGT's canonical
        frame stays a reference view rather than a target view.

        Placement uses the same `rects` that sliced the pixels, so a non-strip
        layout (the RoboTwin mosaic puts cam_high across the full width of the top
        2/3) lands on the tokens that actually observe it.
        """
        n_views = len(views)
        if len(rects) != n_views:
            raise RuntimeError(f"Got {n_views} views but {len(rects)} layout rects.")
        prepared = [self._prepare_vggt_images(v, input_range=input_range) for v in views]
        # [B, V, 2, 3, S, S] -> [B, V*2, 3, S, S], view-major.
        stacked = torch.stack(prepared, dim=1)
        b, v, t = stacked.shape[0], stacked.shape[1], stacked.shape[2]
        if t != 2:
            raise RuntimeError(f"Expected 2 frames per view, got {t}.")
        flat = stacked.reshape(b, v * t, *stacked.shape[3:])
        tokens = self._vggt_patch_tokens(flat)  # [B, V*2, P, D]
        num_patches, dim = int(tokens.shape[2]), int(tokens.shape[3])
        patch_grid_h, patch_grid_w = self._vggt_patch_grid(prepared[0])
        if num_patches != patch_grid_h * patch_grid_w:
            raise RuntimeError(
                f"VGGT returned {num_patches} spatial patches for grid "
                f"{(patch_grid_h, patch_grid_w)}."
            )
        tokens = tokens.reshape(b, v, t, num_patches, dim)

        out: Optional[torch.Tensor] = None
        for view_idx, rect in enumerate(rects):
            r0, r1, c0, c1 = self._rect_to_bounds(
                rect, grid_h, grid_w, what="Token grid", strict=False
            )
            maps = tokens[:, view_idx].reshape(
                b * t, patch_grid_h, patch_grid_w, dim
            ).permute(0, 3, 1, 2)
            maps = F.interpolate(
                maps.float(), size=(r1 - r0, c1 - c0), mode="bilinear", align_corners=False
            )
            maps = maps.permute(0, 2, 3, 1).reshape(b, t, r1 - r0, c1 - c0, dim)
            if out is None:
                out = maps.new_zeros((b, t, grid_h, grid_w, dim))
            out[:, :, r0:r1, c0:c1] = maps
        if out is None:
            raise RuntimeError("No camera views produced teacher features.")
        return out

    def _vggt_patch_tokens(self, prepared: torch.Tensor) -> torch.Tensor:
        """Run the VGGT aggregator and return final-layer patch tokens [B,T,P,D].

        Upstream contract (vggt/models/aggregator.py):
          * `Aggregator.forward(images[B,S,3,H,W]) -> (output_list, patch_start_idx)`.
          * Each cached entry is `cat([frame_inter, global_inter], -1)`, i.e.
            `[B, S, P, 2*embed_dim]`; uncached layers are `None` so that layer
            indices stay stable.
          * `patch_start_idx = 1 + num_register_tokens` leading camera/register
            tokens precede the spatial grid.
        """
        aggregator = getattr(self.vggt, "aggregator", None)
        if aggregator is None:
            raise RuntimeError(
                "Frozen VGGT teacher has no `aggregator`; cannot extract patch features."
            )
        output = aggregator(prepared)
        if not (isinstance(output, tuple) and len(output) == 2):
            raise RuntimeError(
                "Expected VGGT aggregator to return (output_list, patch_start_idx); got "
                f"{type(output).__name__}. The vendored VGGT version may be incompatible."
            )
        token_list, patch_start = output
        # Walk back from the deepest layer: this fork caches only a subset of
        # layers and stores None elsewhere, so `[-1]` alone is not guaranteed to
        # be a tensor.
        tokens = None
        for entry in reversed(token_list):
            if entry is not None:
                tokens = entry
                break
        if tokens is None:
            raise RuntimeError("VGGT aggregator returned no cached layer outputs.")
        if tokens.ndim != 4:
            raise RuntimeError(
                f"Unexpected VGGT aggregator output shape {tuple(tokens.shape)}; expected [B,S,P,D]."
            )
        # Always drop the leading camera/register tokens. Testing "is P a perfect
        # square" instead would silently keep them whenever 5 + n**2 happens to
        # be square for some input resolution.
        patch_start = int(patch_start)
        if patch_start < 0 or patch_start >= tokens.shape[2]:
            raise RuntimeError(
                f"VGGT patch_start_idx={patch_start} is out of range for {tokens.shape[2]} tokens."
            )
        return tokens[:, :, patch_start:]

    # ----- relation-based supervision -----

    @staticmethod
    def _pairwise_distance(feats: torch.Tensor) -> torch.Tensor:
        """R(i,j) = ||f_i - f_j||_2 for [B,N,D] -> [B,N,N]."""
        return torch.cdist(feats.float(), feats.float(), p=2)

    @staticmethod
    def _normalize_relation(relation: torch.Tensor, eps: float) -> torch.Tensor:
        """Scale each relation matrix to unit mean.

        This is what makes the objective immune to VGGT's per-forward up-to-scale
        normalization: only the *shape* of the distance structure is compared, so
        an arbitrary global scale on either side cancels.
        """
        denom = relation.mean(dim=(-2, -1), keepdim=True).clamp(min=eps)
        return relation / denom

    def _sample_token_indices(self, num_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        """Pick a random token subset; the relation matrix is O(N^2) in memory."""
        budget = int(self.cfg.relation_num_tokens)
        if budget <= 0 or budget >= num_tokens:
            return None
        # Unbiased subsample: any pair surviving the draw is an unbiased sample of
        # the full relation matrix, so the expected loss is unchanged.
        return torch.randperm(num_tokens, device=device)[:budget]

    @staticmethod
    def motion_saliency_weights(
        r_target: torch.Tensor,
        off_diag: torch.Tensor,
        *,
        eta: float,
        eps: float,
    ) -> torch.Tensor:
        """Per-token supervision weight from the teacher's own motion → [B, N].

        w_i is token i's mean absolute relation *change* across all partners, i.e.
        exactly "how much did this token's geometry move", normalized to mean 1 and
        mixed with the uniform prior eta so no token is fully starved.

        This is deliberately parameter-free and computed under no_grad. A *learned*
        head scored by this loss cannot work: with
        L = sum(w*d)/sum(w), the gradient is dL/dw_kl = (d_kl - L)/sum(w), so
        descent raises the weight of *low-error* pairs. The lowest-error pairs are
        the static background (r_target ~ 0 there, and `loss_background` actively
        trains the student toward 0 on them), so a learned head is driven to invert
        the intended objective -- empirically it anti-correlated with real motion
        (Spearman -0.33..-0.67, top-10% IoU 0.000 across every sample checked) and
        locked onto a fixed grid corner independent of image content and action.
        """
        n_kept = int(r_target.shape[-1])
        denom = float(max(n_kept - 1, 1))
        saliency = (r_target.abs() * off_diag).sum(-1) / denom  # [B, N]
        saliency = saliency / saliency.mean(-1, keepdim=True).clamp(min=eps)
        return (1.0 - eta) * saliency + eta

    def relation_loss(
        self,
        z_delta: torch.Tensor,
        teacher_feats: torch.Tensor,
        *,
        source_grid: Optional[tuple[int, int]] = None,
        action_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Align the student's pairwise structure with the frozen VGGT teacher's.

        Supervises the *change* in relational geometry: the teacher term is the
        difference between the target-frame and reference-frame relation matrices,
        so static structure cancels to zero and only what moved contributes.

        Args:
            z_delta:      [B, N, C] deep z_Δ slots read out of the image stream.
            teacher_feats:[B, 2, H_l, W_l, D] frozen VGGT features (0=ref, 1=tgt).
            action_tokens:[B, T_a, C_a] action-expert hidden states for weighting.
        """
        if z_delta.ndim != 3:
            raise ValueError(f"z_delta must be [B,N,C], got {tuple(z_delta.shape)}")
        if teacher_feats.ndim != 5 or teacher_feats.shape[1] != 2:
            raise ValueError(
                f"teacher_feats must be [B,2,H,W,D], got {tuple(teacher_feats.shape)}"
            )
        eps = float(self.cfg.eps)
        n_tokens = int(z_delta.shape[1])
        grid = source_grid or self._infer_square_grid(n_tokens)
        z_map = rearrange(z_delta, "b (h w) c -> b c h w", h=grid[0], w=grid[1])
        # See the note in geometry_forcing_loss: align with the projector's own
        # parameter dtype instead of assuming the incoming stream already matches.
        proj_dtype = next(self.alignment_projector.parameters()).dtype
        student = self.alignment_projector(z_map.to(proj_dtype))  # [B, C_out, H_l, W_l]
        student = rearrange(student, "b c h w -> b (h w) c")

        teacher = teacher_feats.to(device=student.device)
        b, _, th, tw, _ = teacher.shape
        if (th, tw) != (grid[0], grid[1]):
            raise ValueError(
                f"Teacher grid {(th, tw)} does not match student grid {tuple(grid)}."
            )
        teacher_ref = teacher[:, 0].reshape(b, th * tw, -1)
        teacher_tgt = teacher[:, 1].reshape(b, th * tw, -1)

        idx = self._sample_token_indices(student.shape[1], student.device)
        if idx is not None:
            student = student[:, idx]
            teacher_ref = teacher_ref[:, idx]
            teacher_tgt = teacher_tgt[:, idx]

        r_student = self._normalize_relation(self._pairwise_distance(student), eps)
        with torch.no_grad():
            r_ref = self._normalize_relation(self._pairwise_distance(teacher_ref), eps)
            r_tgt = self._normalize_relation(self._pairwise_distance(teacher_tgt), eps)
            # Geometry *transition*: static structure cancels, motion survives.
            r_target = r_tgt - r_ref
            r_target = r_target.to(dtype=r_student.dtype)

        # The diagonal is identically zero on both sides and carries no signal, so
        # it is excluded from centring and from every reduction below. Leaving it
        # in would drag both means toward 0 by a factor that depends only on the
        # token budget.
        n_kept = int(r_student.shape[-1])
        off_diag = ~torch.eye(n_kept, device=r_student.device, dtype=torch.bool)
        off_diag = off_diag[None].expand(b, -1, -1)
        n_pairs = float(n_kept * (n_kept - 1))

        # The student predicts the change directly, so its own relation matrix is
        # centred before comparison — otherwise it would have to encode the
        # (unsupervised) absolute structure as well.
        def _center(mat: torch.Tensor) -> torch.Tensor:
            mean = (mat * off_diag).sum(dim=(-2, -1), keepdim=True) / max(n_pairs, eps)
            return (mat - mean) * off_diag

        r_student = _center(r_student)
        r_target = _center(r_target)

        # Concentrate supervision on what actually moved. The weights come from the
        # frozen teacher's own relation change, not from a learned head -- see
        # `motion_saliency_weights` for why a learned head provably inverts here.
        if bool(self.cfg.use_action_aware_weight):
            with torch.no_grad():
                weights = self.motion_saliency_weights(
                    r_target,
                    off_diag,
                    eta=float(self.cfg.action_weight_eta),
                    eps=eps,
                )
            # w_ij = sqrt(w_i * w_j) — MECo-WAM Eq.14. A pair matters only if both
            # of its endpoints moved.
            w_pair = torch.sqrt(
                weights[:, :, None].clamp(min=0.0) * weights[:, None, :].clamp(min=0.0) + eps
            ).to(dtype=r_student.dtype)
        else:
            weights = None
            w_pair = torch.ones_like(r_student)
        w_pair = w_pair * off_diag

        diff = (r_student - r_target).abs()
        loss_relation = (w_pair * diff).sum() / w_pair.sum().clamp(min=eps)

        # Background suppression: where the teacher says nothing changed, the
        # student must predict no change. Turns "background 3D does not matter"
        # from an omission into an explicit constraint.
        with torch.no_grad():
            # quantile() only supports float32/64, and r_target may be bf16.
            target_abs = r_target.detach().float().abs()
            # Rank only off-diagonal entries: the zero diagonal would otherwise
            # occupy the bottom of the distribution and bias the threshold down.
            ranked = target_abs.masked_fill(~off_diag, float("inf")).flatten(1)
            ranked, _ = ranked.sort(dim=1)
            thresh = ranked[:, max(int(n_pairs * 0.5) - 1, 0)][:, None, None]
            static = ((target_abs <= thresh) & off_diag).to(dtype=r_student.dtype)
        loss_background = (r_student.abs() * static).sum() / static.sum().clamp(min=eps)

        # `background_weight` defaults to `scale_weight` only for backwards
        # compatibility with single-objective relation configs. When both objectives
        # are active, `scale_weight` means lambda_Scale, so set background_weight
        # explicitly rather than letting one number drive two unrelated terms.
        bg_w = self.cfg.background_weight
        bg_w = float(self.cfg.scale_weight) if bg_w is None else float(bg_w)
        # Returned WITHOUT cfg.weight; see the note in geometry_forcing_loss.
        loss = loss_relation + bg_w * loss_background
        info = {
            "loss_geometry_relation_total": loss.detach(),
            "loss_geometry_relation": loss_relation.detach(),
            "loss_geometry_background": loss_background.detach(),
            "geometry_relation_pred_absmean": r_student.detach().float().abs().mean(),
            "geometry_relation_target_absmean": r_target.detach().float().abs().mean(),
            "geometry_action_weight_max": (
                weights.detach().float().max() if weights is not None else torch.zeros((), device=z_delta.device)
            ),
        }
        return loss, info

    def geometry_forcing_loss(
        self,
        z_delta: torch.Tensor,
        y_delta: torch.Tensor,
        *,
        source_grid: Optional[tuple[int, int]] = None,
        valid: Optional[torch.Tensor] = None,
        action_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Geometry Forcing (ICLR 2026, Sec 4.2) on the transition target y_Δ.

        Two decoupled alignment terms over per-patch teacher differences:

            L_Angular = -(1/P) Σ_p cos(y_Δ,p, f_φ(z_Δ,p))
            L_Scale   =  (1/P) Σ_p ‖ g_φ( f_φ(z_Δ,p)/‖f_φ(z_Δ,p)‖ ) - y_Δ,p ‖²

        Both alignment terms are invariant to the raw radius of f_φ. A small
        gauge-fixing term prevents that radius from drifting upward and shrinking
        the useful backward signal through normalisation:

            L_Norm = SmoothL1(‖f_φ(z_Δ,p)‖ / mean_p‖y_Δ,p‖ - 1)

        The split is the point: Angular fixes direction and is invariant to the
        student's activation magnitude, while Scale recovers magnitude *from the
        normalised direction*, so the student cannot satisfy it by rescaling its
        own features. A single MSE on y_Δ collapses for exactly that reason
        (paper Table 3: 1648 vs 243 FVD).

        Unlike the paper's static-scene setting, the target here is a temporal
        difference, so it is only well-defined under patch correspondence -- see
        `compute_teacher_delta`, which warps the reference frame first.

        Args:
            z_delta: [B, N, C] deep bottleneck slots.
            y_delta: [B, H_l, W_l, D] warped teacher difference.
            valid:   [B, H_l, W_l] per-patch correspondence weight, or None.
        """
        if z_delta.ndim != 3:
            raise ValueError(f"z_delta must be [B,N,C], got {tuple(z_delta.shape)}")
        if y_delta.ndim != 4:
            raise ValueError(f"y_delta must be [B,H,W,D], got {tuple(y_delta.shape)}")
        if self.scale_head is None or self.forcing_projector is None:
            raise RuntimeError(
                "geometry_forcing heads are absent; add 'geometry_forcing' to cfg.objective."
            )

        eps = float(self.cfg.eps)
        n_tokens = int(z_delta.shape[1])
        grid = source_grid or self._infer_square_grid(n_tokens)
        b, th, tw, _ = y_delta.shape
        if (th, tw) != (int(grid[0]), int(grid[1])):
            raise ValueError(
                f"Teacher grid {(th, tw)} does not match student grid {tuple(grid)}."
            )

        z_map = rearrange(z_delta, "b (h w) c -> b c h w", h=grid[0], w=grid[1])
        # Match the projector parameter dtype under mixed precision.
        proj_dtype = next(self.forcing_projector.parameters()).dtype
        student = self.forcing_projector(z_map.to(proj_dtype))  # f_phi(z): [B,D,H_l,W_l]
        student = rearrange(student, "b c h w -> b (h w) c")
        # Held in fp32, NOT cast to the student's dtype. y_delta is a difference of
        # two similar large feature vectors, so its magnitude is small relative to
        # the operands -- a bf16 round-trip would quantise away exactly the signal
        # being supervised. Every reduction below is fp32 for the same reason.
        teacher_f = y_delta.reshape(b, th * tw, -1).to(
            device=student.device, dtype=torch.float32
        )

        # Per-patch weights, held in fp32: every reduction below runs in fp32 so a
        # bf16 autocast context cannot round the summed loss to 3 mantissa bits.
        with torch.no_grad():
            w = torch.ones(b, th * tw, device=student.device, dtype=torch.float32)
            if valid is not None and bool(self.cfg.warp_drop_invalid):
                w = w * valid.reshape(b, th * tw).to(device=student.device, dtype=torch.float32)
            if bool(self.cfg.use_action_aware_weight):
                # Concentrate on patches whose geometry actually changed, so flat
                # background does not dominate. Teacher-derived and detached: a
                # *learned* weight scored by a normalised weighted mean is driven to
                # up-weight whatever is already easiest, which inverts the objective
                # (see `motion_saliency_weights`).
                mag = teacher_f.detach().norm(dim=-1)  # [B, P]
                mag = mag / mag.mean(dim=-1, keepdim=True).clamp(min=eps)
                eta = float(self.cfg.action_weight_eta)
                w = w * ((1.0 - eta) * mag + eta)
        w_sum = w.sum().clamp(min=eps)

        student_f = student.float()
        student_norm = student_f.norm(dim=-1)

        # --- Angular: direction only, magnitude-invariant on both sides. -----
        cos = F.cosine_similarity(student_f, teacher_f, dim=-1, eps=eps)
        loss_angular = -(w * cos).sum() / w_sum

        # --- Scale: predict the target magnitude from the unit direction. ----
        student_unit = student_f / student_norm.unsqueeze(-1).clamp(min=eps)
        # The normalisation above runs in fp32 on purpose, but the head's
        # parameters follow the model precision (bf16 under DeepSpeed, which
        # casts the whole model and runs WITHOUT autocast), so hand the head its
        # own dtype and lift the prediction back to fp32 for the reduction.
        head_dtype = next(self.scale_head.parameters()).dtype
        pred = self.scale_head(student_unit.to(head_dtype)).float()
        loss_scale = (w * (pred - teacher_f).pow(2).mean(dim=-1)).sum() / w_sum

        # --- Radius gauge: keep normalisation gradients well conditioned. ----
        # Use one detached target for the batch rather than matching each patch's
        # magnitude. The Scale head remains solely responsible for recovering
        # per-patch teacher scale from direction; this term only removes the
        # projector's arbitrary global radius. A unit floor avoids an unstable
        # ratio for batches whose teacher transition is almost exactly static.
        with torch.no_grad():
            teacher_norm = teacher_f.norm(dim=-1)
            norm_target = ((w * teacher_norm).sum() / w_sum).clamp(min=1.0)
        relative_norm_error = student_norm / norm_target - 1.0
        loss_norm = (
            w
            * F.smooth_l1_loss(
                relative_norm_error,
                torch.zeros_like(relative_norm_error),
                reduction="none",
            )
        ).sum() / w_sum

        # Returned WITHOUT cfg.weight: `alignment_loss` applies the shared weight
        # once, after mixing whichever objectives are active. Applying it here too
        # would square it when both objectives run.
        loss = (
            float(self.cfg.angular_weight) * loss_angular
            + float(self.cfg.scale_weight) * loss_scale
            + float(self.cfg.forcing_norm_weight) * loss_norm
        )
        info = {
            "loss_geometry_forcing": loss.detach(),
            "loss_geometry_angular": loss_angular.detach(),
            "loss_geometry_scale": loss_scale.detach(),
            "loss_geometry_norm": loss_norm.detach(),
            "loss_geometry_norm_weighted": (
                float(self.cfg.forcing_norm_weight) * loss_norm.detach()
            ),
            # cos in [-1, 1]; this is the readable "is alignment happening" number.
            "geometry_cos_mean": cos.detach().float().mean(),
            "geometry_teacher_delta_norm": teacher_norm.detach().mean(),
            "geometry_student_norm": student_norm.detach().mean(),
            "geometry_forcing_norm_target": norm_target.detach(),
            "geometry_student_teacher_norm_ratio": (
                student_norm.detach().mean() / norm_target.detach()
            ),
            "geometry_valid_frac": (
                valid.detach().float().mean()
                if valid is not None
                else torch.ones((), device=z_delta.device)
            ),
        }
        return loss, info

    def alignment_loss(
        self,
        z_delta: torch.Tensor,
        images: torch.Tensor,
        *,
        source_grid: Optional[tuple[int, int]] = None,
        action_tokens: Optional[torch.Tensor] = None,
        input_range: str = "tanh",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Geometry-transition supervision.

        Runs every objective in `cfg.objectives` and returns their weighted sum, so
        `relation` and `geometry_forcing` can train alone or together:

            total = weight * (relation_weight * relation + forcing_weight * forcing)

        Each sub-loss returns its terms unscaled; the shared `weight` is applied
        once here.
        """
        if not self.uses_teacher_supervision:
            raise RuntimeError(
                "alignment_loss is disabled because teacher_supervision_enabled=false."
            )
        grid = (
            source_grid
            or self._active_query_grid
            or self._infer_square_grid(int(z_delta.shape[1]))
        )
        total: Optional[torch.Tensor] = None
        info: dict[str, torch.Tensor] = {}

        # With warping enabled the two objectives need different teacher inputs
        # (forcing wants the motion-compensated reference; relation's r_ref is
        # defined on the true reference), so they get their own forwards unless
        # explicitly told to share.
        share = bool(self.cfg.share_teacher_forward) and not (
            bool(self.cfg.warp_ref_to_target) and str(self.cfg.warp_method) != "none"
        )
        shared_feats: Optional[torch.Tensor] = None
        forcing_target: Optional[torch.Tensor] = None
        forcing_valid: Optional[torch.Tensor] = None
        if share and self.uses_relation and self.uses_forcing:
            shared_feats = self.compute_teacher_features(
                images, grid=grid, input_range=input_range
            )

        if self.uses_relation:
            feats = (
                shared_feats
                if shared_feats is not None
                else self.compute_teacher_features(images, grid=grid, input_range=input_range)
            )
            loss_rel, info_rel = self.relation_loss(
                z_delta, feats, source_grid=grid, action_tokens=action_tokens
            )
            scaled = float(self.cfg.relation_weight) * loss_rel
            total = scaled if total is None else total + scaled
            info.update(info_rel)

        if self.uses_forcing:
            if shared_feats is not None:
                y_delta = shared_feats[:, 1] - shared_feats[:, 0]
                valid = None
            else:
                y_delta, valid = self.compute_teacher_delta(
                    images, grid=grid, input_range=input_range
                )
            loss_forc, info_forc = self.geometry_forcing_loss(
                z_delta,
                y_delta,
                source_grid=grid,
                valid=valid,
                action_tokens=action_tokens,
            )
            forcing_target = y_delta
            forcing_valid = valid
            scaled = float(self.cfg.forcing_weight) * loss_forc
            total = scaled if total is None else total + scaled
            info.update(info_forc)

        if total is None:
            raise RuntimeError(f"No geometry objective produced a loss: {self.objectives}")
        total = float(self.cfg.weight) * total
        info["loss_geometry_transition"] = total.detach()
        return total, info

    # Auxiliary mode helpers.

    def encode_transition_tokens(self, h_t: torch.Tensor, h_t1: torch.Tensor) -> torch.Tensor:
        return h_t1 - h_t

    def encode_transition_from_img_in(
        self,
        *,
        ref_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        img_in: nn.Module,
    ) -> torch.Tensor:
        h_t = img_in(ref_tokens)
        h_t1 = img_in(target_tokens)
        return self.encode_transition_tokens(h_t, h_t1)

    def _tokens_to_btc_hw(
        self,
        x: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        tokens_per_frame: Optional[int] = None,
        grid: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if x.ndim == 5 and x.shape[1] in (1, 3):
            return rearrange(x, "b c t h w -> b t c h w")
        if x.ndim == 5:
            return x
        if x.ndim == 4:
            b, t, n, c = x.shape
            h, w = grid or self._infer_square_grid(n)
            return rearrange(x, "b t (h w) c -> b t c h w", h=h, w=w)
        if x.ndim == 3:
            b, n_total, c = x.shape
            if tokens_per_frame is None:
                if num_frames is None:
                    raise ValueError("For [B,N,C] source tokens, pass tokens_per_frame or num_frames.")
                tokens_per_frame = n_total // num_frames
            t = n_total // tokens_per_frame
            h, w = grid or self._infer_square_grid(tokens_per_frame)
            return rearrange(x, "b (t h w) c -> b t c h w", t=t, h=h, w=w)
        raise ValueError(f"Unsupported source tensor shape: {tuple(x.shape)}")

    def compute_source_transition(
        self,
        source: torch.Tensor | Sequence[torch.Tensor],
        *,
        batch_size: int,
        num_frames: int,
        source_tokens_per_frame: Optional[int] = None,
        source_grid: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if isinstance(source, (list, tuple)):
            source = source[self.cfg.source_layer]
        source = self._tokens_to_btc_hw(
            source,
            batch_size=batch_size,
            num_frames=num_frames,
            tokens_per_frame=source_tokens_per_frame,
            grid=source_grid,
        )
        return source[:, 1:] - source[:, :-1]

    def forward(
        self,
        source: torch.Tensor | Sequence[torch.Tensor],
        images: torch.Tensor,
        *,
        source_tokens_per_frame: Optional[int] = None,
        source_grid: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Auxiliary path: align the encoded feature transition to the teacher."""
        images_btc = self._as_btc_hw(images)
        b, t = images_btc.shape[:2]
        source_transition = self.compute_source_transition(
            source,
            batch_size=b,
            num_frames=t,
            source_tokens_per_frame=source_tokens_per_frame,
            source_grid=source_grid,
        )
        # [B, T-1, C, H, W] → first step → tokens
        delta = source_transition[:, 0]
        z_delta = rearrange(delta, "b c h w -> b (h w) c")
        return self.alignment_loss(z_delta, images, source_grid=source_grid or (delta.shape[-2], delta.shape[-1]))

    def trainable_state_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "placement": self.placement,
            "architecture": {
                "alignment_projector_mid_channels": self._projector_mid_channels(
                    self.alignment_projector
                ),
                "forcing_projector_mid_channels": self._projector_mid_channels(
                    self.forcing_projector
                ),
            },
            "query_tokens": self.query_tokens.detach().cpu(),
            "updater": self.updater.state_dict(),
            "alignment_projector": self.alignment_projector.state_dict(),
            # Saved so a resumed run continues the loss warmup ramp.
            "train_step": self._train_step.detach().cpu(),
        }
        if self.forcing_projector is not None:
            payload["forcing_projector"] = self.forcing_projector.state_dict()
        if self.scale_head is not None:
            payload["scale_head"] = self.scale_head.state_dict()
        if self.action_weight_video_proj is not None:
            payload["action_weight_video_proj"] = self.action_weight_video_proj.state_dict()
            payload["action_weight_action_proj"] = self.action_weight_action_proj.state_dict()
        if self.action_residual_adapter is not None:
            payload["action_residual_adapter"] = self.action_residual_adapter.state_dict()
        return payload

    @staticmethod
    def _projector_mid_channels(projector: Optional[nn.Module]) -> Optional[int]:
        """Return a projector's hidden width without relying on its config."""
        if projector is None:
            return None
        try:
            weight = projector.net[0].weight
        except (AttributeError, IndexError):
            return None
        return int(weight.shape[0])

    @staticmethod
    def _checkpoint_projector_mid_channels(payload: object) -> Optional[int]:
        """Infer a projector width from its serialized convolution tensors."""
        if not isinstance(payload, dict):
            return None
        first = payload.get("net.0.weight")
        middle = payload.get("net.2.weight")
        if not isinstance(first, torch.Tensor) or first.ndim != 4:
            return None
        width = int(first.shape[0])
        if isinstance(middle, torch.Tensor) and middle.ndim == 4:
            if int(middle.shape[0]) != width or int(middle.shape[1]) != width:
                raise RuntimeError(
                    "Malformed projector checkpoint: net.0/net.2 hidden widths "
                    f"disagree ({width} vs {tuple(middle.shape[:2])})."
                )
        return width

    def _adopt_checkpoint_projector_width(self, name: str, payload: object) -> None:
        """Match a 256/1024 projector to checkpoint weights before strict load.

        No tensor conversion is performed.  When both the configured module and
        checkpoint use one of the supported widths, the existing convolution
        parameters are resized in place before loading, preserving optimizer
        references during full-state resume.
        """
        projector = getattr(self, name)
        checkpoint_width = self._checkpoint_projector_mid_channels(payload)
        current_width = self._projector_mid_channels(projector)
        if checkpoint_width is None or current_width == checkpoint_width:
            return
        if (
            current_width not in _CHECKPOINT_PROJECTOR_WIDTHS
            or checkpoint_width not in _CHECKPOINT_PROJECTOR_WIDTHS
        ):
            # Preserve the normal state-dict error for unsupported widths.
            return

        if name not in {"forcing_projector", "alignment_projector"}:
            raise ValueError(f"Unknown projector module {name!r}.")
        if not isinstance(projector, AlignmentProjector):
            return
        layers = (projector.net[0], projector.net[2], projector.net[4])
        if not all(isinstance(layer, nn.Conv2d) for layer in layers):
            return
        # Keep Parameter objects stable.  Accelerate may have prepared the
        # optimizer before restoring a full training state; replacing modules
        # would leave that optimizer pointing at stale parameters.
        for layer_index, layer in enumerate(layers):
            weight_key = f"net.{(0, 2, 4)[layer_index]}.weight"
            bias_key = f"net.{(0, 2, 4)[layer_index]}.bias"
            checkpoint_weight = payload.get(weight_key)  # type: ignore[union-attr]
            checkpoint_bias = payload.get(bias_key)  # type: ignore[union-attr]
            if not isinstance(checkpoint_weight, torch.Tensor):
                continue
            layer.weight.data = torch.empty(
                checkpoint_weight.shape,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            if isinstance(checkpoint_bias, torch.Tensor) and layer.bias is not None:
                layer.bias.data = torch.empty(
                    checkpoint_bias.shape,
                    device=layer.bias.device,
                    dtype=layer.bias.dtype,
                )
            layer.out_channels = int(checkpoint_weight.shape[0])
            layer.in_channels = int(checkpoint_weight.shape[1])
        if name == "alignment_projector":
            # Keep the compatibility alias in sync after the in-place resize.
            self.projector = self.alignment_projector
        logger.info(
            "Adopted checkpoint projector width for %s in place: config=%s checkpoint=%s",
            name,
            current_width,
            checkpoint_width,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Adopt supported projector widths for native PyTorch/Accelerate loads."""
        for name in ("alignment_projector", "forcing_projector"):
            module_prefix = f"{prefix}{name}."
            module_state = {
                key[len(module_prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(module_prefix)
            }
            self._adopt_checkpoint_projector_width(name, module_state)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def load_trainable_state_dict(self, payload: dict[str, object], strict: bool = True) -> None:
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload, got {type(payload)}")
        payload_placement = str(payload.get("placement", "video"))
        if payload_placement != self.placement:
            logger.warning(
                "Skipping geometry_transition payload with placement=%s for current "
                "placement=%s; the main ImageWAM/DiT checkpoint remains loaded.",
                payload_placement,
                self.placement,
            )
            return
        self._adopt_checkpoint_projector_width(
            "alignment_projector", payload.get("alignment_projector")
        )
        self._adopt_checkpoint_projector_width(
            "forcing_projector", payload.get("forcing_projector")
        )
        if "query_tokens" in payload:
            qt = payload["query_tokens"]
            if isinstance(qt, torch.Tensor) and qt.shape == self.query_tokens.shape:
                self.query_tokens.data.copy_(qt.to(device=self.query_tokens.device))
        if "updater" in payload:
            self.updater.load_state_dict(payload["updater"], strict=strict)
        if "train_step" in payload:
            step = payload["train_step"]
            if isinstance(step, torch.Tensor):
                self._train_step.fill_(int(step.item()))
            else:
                self._train_step.fill_(int(step))
        if "action_weight_video_proj" in payload and self.action_weight_video_proj is not None:
            self.action_weight_video_proj.load_state_dict(
                payload["action_weight_video_proj"], strict=strict
            )
            self.action_weight_action_proj.load_state_dict(
                payload["action_weight_action_proj"], strict=strict
            )
        if "action_residual_adapter" in payload and self.action_residual_adapter is not None:
            load_result = self.action_residual_adapter.load_state_dict(
                payload["action_residual_adapter"], strict=False
            )
            if strict and (load_result.missing_keys or load_result.unexpected_keys):
                raise RuntimeError(
                    "Incompatible action residual checkpoint: "
                    f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
                )
            if load_result.missing_keys or load_result.unexpected_keys:
                logger.warning(
                    "Loaded action residual with compatible keys; "
                    "missing=%s unexpected=%s.",
                    load_result.missing_keys,
                    load_result.unexpected_keys,
                )
        if "forcing_projector" in payload and self.forcing_projector is not None:
            self.forcing_projector.load_state_dict(payload["forcing_projector"], strict=strict)
        if "scale_head" in payload and self.scale_head is not None:
            self.scale_head.load_state_dict(payload["scale_head"], strict=strict)
        if "alignment_projector" in payload:
            self.alignment_projector.load_state_dict(payload["alignment_projector"], strict=strict)
        elif "transition_head" not in payload and payload and all(
            isinstance(k, str) and k.startswith("net.") for k in payload.keys()
        ):
            self.alignment_projector.load_state_dict(payload, strict=strict)
