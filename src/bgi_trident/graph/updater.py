"""Online graph updater for the predict-execute-learn loop.

Updates the heterogeneous consumption graph with new edges and
weight increments after each completed transaction. This closes
the learning loop: every order improves the next prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import torch

from bgi_trident.graph.schema import NodeType

logger = logging.getLogger(__name__)


@dataclass
class TransactionEvent:
    """A completed transaction to update the graph with."""

    user_id: str
    entity_id: str
    entity_type: str  # "restaurant", "product", "venue"
    timestamp: datetime
    amount: float
    metadata: dict | None = None


class GraphUpdater:
    """Updates graph edges and weights after completed transactions.

    In the predict-decide-execute-learn loop, this handles the Learn phase.
    Every completed order creates new edges or increments existing edge weights,
    making the next prediction more accurate.
    """

    def __init__(self) -> None:
        self._pending_updates: list[TransactionEvent] = []

    def record_transaction(self, event: TransactionEvent) -> None:
        """Queue a transaction for batch graph update."""
        self._pending_updates.append(event)
        logger.debug("Queued graph update: %s -> %s", event.user_id, event.entity_id)

    def apply_updates(
        self,
        node_maps: dict[NodeType, dict],
        edge_indices: dict[tuple, torch.Tensor],
        edge_weights: dict[tuple, torch.Tensor],
    ) -> tuple[dict[tuple, torch.Tensor], dict[tuple, torch.Tensor]]:
        """Apply queued updates to edge indices and weights.

        For existing edges: increment weight.
        For new edges: add to edge index and initialize weight.

        Returns updated (edge_indices, edge_weights).
        """
        for event in self._pending_updates:
            edge_key = self._get_edge_key(event.entity_type)
            if edge_key is None:
                continue

            src_type, rel, dst_type = edge_key
            src_map = node_maps.get(NodeType(src_type), {})
            dst_map = node_maps.get(NodeType(dst_type), {})

            src_idx = src_map.get(event.user_id)
            dst_idx = dst_map.get(event.entity_id)

            if src_idx is None or dst_idx is None:
                logger.warning("Unknown node: %s or %s", event.user_id, event.entity_id)
                continue

            if edge_key in edge_indices:
                idx = edge_indices[edge_key]
                mask = (idx[0] == src_idx) & (idx[1] == dst_idx)
                if mask.any():
                    pos = mask.nonzero(as_tuple=True)[0][0]
                    if edge_key in edge_weights:
                        edge_weights[edge_key][pos] += 1.0
                else:
                    new_edge = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
                    edge_indices[edge_key] = torch.cat([idx, new_edge], dim=1)
                    if edge_key in edge_weights:
                        edge_weights[edge_key] = torch.cat([
                            edge_weights[edge_key],
                            torch.tensor([1.0]),
                        ])
            else:
                edge_indices[edge_key] = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
                edge_weights[edge_key] = torch.tensor([1.0])

        count = len(self._pending_updates)
        self._pending_updates.clear()
        logger.info("Applied %d graph updates", count)
        return edge_indices, edge_weights

    def _get_edge_key(self, entity_type: str) -> tuple[str, str, str] | None:
        mapping = {
            "restaurant": ("user", "ordered_from", "restaurant"),
            "product": ("user", "purchased", "product"),
            "venue": ("user", "booked_at", "venue"),
        }
        return mapping.get(entity_type)

    @property
    def pending_count(self) -> int:
        return len(self._pending_updates)
