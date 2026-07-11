"""Qwen wrapper with graph-memory cross-attention adapters.

This module deliberately does not reimplement the entire Qwen model. Instead,
it wraps selected decoder layers and inserts a small trainable graph adapter
after self-attention and before the MLP. With graph_scale=0, the wrapper calls
the original frozen layer directly, which is the strongest base-equivalence
guard we can use.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kg_xattn.graph_modules import GatedGraphCrossAttention, GraphMemoryProjector


@dataclass
class GraphCausalLMOutput:
    """Output returned by GraphMemoryQwenForCausalLM.forward."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    graph_attentions: dict[int, torch.Tensor] | None
    gate_values: dict[int, float]


def _call_with_supported_kwargs(module: nn.Module, **kwargs: Any) -> Any:
    """Call a Transformers module while filtering version-specific kwargs."""

    signature = inspect.signature(module.forward)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return module(**supported)


class GraphInjectedQwenDecoderLayer(nn.Module):
    """Wrap one Qwen decoder layer and optionally inject graph cross-attention."""

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        cross_attention: GatedGraphCrossAttention | None,
    ) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.cross_attention = cross_attention

        self._graph_memory: torch.Tensor | None = None
        self._graph_mask: torch.Tensor | None = None
        self._text_mask: torch.Tensor | None = None
        self._graph_scale: float = 1.0
        self._return_graph_attentions: bool = False
        self.last_graph_attention: torch.Tensor | None = None

    def set_graph_state(
        self,
        graph_memory: torch.Tensor | None,
        graph_mask: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        graph_scale: float,
        return_graph_attentions: bool,
    ) -> None:
        """Store graph state that will be used during the next forward pass."""

        self._graph_memory = graph_memory
        self._graph_mask = graph_mask
        self._text_mask = text_mask
        self._graph_scale = float(graph_scale)
        self._return_graph_attentions = return_graph_attentions
        self.last_graph_attention = None

    def _should_inject(self) -> bool:
        """Return whether this layer should use graph cross-attention now."""

        return (
            self.cross_attention is not None
            and self._graph_memory is not None
            and self._graph_mask is not None
            and self._text_mask is not None
            and self._graph_scale != 0.0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        """Run the original layer, or original+self graph injection."""

        if not self._should_inject():
            return _call_with_supported_kwargs(
                self.original_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # The following block mirrors Qwen2DecoderLayer.forward:
        # input norm -> causal self-attention -> residual add.
        residual = hidden_states
        hidden_states = self.original_layer.input_layernorm(hidden_states)
        self_attn_outputs = _call_with_supported_kwargs(
            self.original_layer.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        attn_output = self_attn_outputs[0]
        self_attn_weights = self_attn_outputs[1] if output_attentions else None
        present_key_value = None
        if use_cache:
            present_key_value = self_attn_outputs[-1]

        hidden_states = residual + attn_output

        # Graph memory is injected after self-attention and before the MLP.
        # The cross-attention module owns its own RMSNorm on both text and graph
        # streams, so this does not disturb Qwen's original post-attention norm.
        graph_result = self.cross_attention(
            hidden_states=hidden_states,
            graph_memory=self._graph_memory,
            graph_mask=self._graph_mask,
            text_mask=self._text_mask,
            graph_scale=self._graph_scale,
            return_weights=self._return_graph_attentions,
        )
        hidden_states = graph_result.hidden_states
        self.last_graph_attention = graph_result.attention_weights

        # Finish the original layer: post-attention norm -> MLP -> residual add.
        residual = hidden_states
        hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.original_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: tuple[Any, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class GraphMemoryQwenForCausalLM(nn.Module):
    """Frozen Qwen plus trainable graph-memory projector and cross-attention."""

    def __init__(
        self,
        base_model: nn.Module,
        graph_layers: list[int],
        raw_graph_dim: int = 2304,
        graph_projector_hidden_dim: int = 512,
        max_memory_tokens: int = 48,
        gate_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.graph_layers = list(graph_layers)
        self.graph_scale = 1.0

        config = base_model.config
        hidden_size = int(config.hidden_size)
        num_query_heads = int(config.num_attention_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_query_heads))
        head_dim = int(getattr(config, "head_dim", hidden_size // num_query_heads))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.graph_projector = GraphMemoryProjector(
            raw_graph_dim=raw_graph_dim,
            projector_hidden_dim=graph_projector_hidden_dim,
            lm_hidden_dim=hidden_size,
            max_memory_tokens=max_memory_tokens,
            eps=eps,
        )

        self.cross_attentions = nn.ModuleDict(
            {
                str(layer_idx): GatedGraphCrossAttention(
                    hidden_size=hidden_size,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    eps=eps,
                    gate_init=gate_init,
                )
                for layer_idx in self.graph_layers
            }
        )

        self._wrap_decoder_layers()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        graph_layers: list[int],
        torch_dtype: torch.dtype | str | None = None,
        device_map: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "GraphMemoryQwenForCausalLM":
        """Load Qwen through Transformers and wrap it for graph memory."""

        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        return cls(base_model=base_model, graph_layers=graph_layers)

    def _get_layers(self) -> nn.ModuleList:
        """Return the decoder layer list from a Qwen causal-LM model."""

        if not hasattr(self.base_model, "model") or not hasattr(
            self.base_model.model, "layers"
        ):
            raise AttributeError("Expected base_model.model.layers on Qwen model")
        return self.base_model.model.layers

    def _wrap_decoder_layers(self) -> None:
        """Replace every Qwen layer with a wrapper; selected layers get adapters."""

        layers = self._get_layers()
        max_index = len(layers) - 1
        for layer_idx in self.graph_layers:
            if layer_idx < 0 or layer_idx > max_index:
                raise ValueError(f"graph layer {layer_idx} is outside 0..{max_index}")

        for index, layer in enumerate(layers):
            cross_attention = (
                self.cross_attentions[str(index)]
                if str(index) in self.cross_attentions
                else None
            )
            layers[index] = GraphInjectedQwenDecoderLayer(
                original_layer=layer,
                layer_idx=index,
                cross_attention=cross_attention,
            )

    def set_graph_scale(self, graph_scale: float) -> None:
        """Set the global inference-time graph residual scale."""

        self.graph_scale = float(graph_scale)

    def _selected_wrappers(self) -> list[GraphInjectedQwenDecoderLayer]:
        """Return only the wrappers that contain graph cross-attention."""

        layers = self._get_layers()
        return [layers[layer_idx] for layer_idx in self.graph_layers]

    def _set_layer_graph_state(
        self,
        graph_memory: torch.Tensor | None,
        graph_mask: torch.Tensor | None,
        text_mask: torch.Tensor | None,
        graph_scale: float,
        return_graph_attentions: bool,
    ) -> None:
        """Push graph tensors into selected layer wrappers before forward."""

        for wrapper in self._selected_wrappers():
            wrapper.set_graph_state(
                graph_memory=graph_memory,
                graph_mask=graph_mask,
                text_mask=text_mask,
                graph_scale=graph_scale,
                return_graph_attentions=return_graph_attentions,
            )

    def _collect_graph_attentions(self) -> dict[int, torch.Tensor] | None:
        """Collect attention weights saved by selected wrappers."""

        collected: dict[int, torch.Tensor] = {}
        for wrapper in self._selected_wrappers():
            if wrapper.last_graph_attention is not None:
                collected[wrapper.layer_idx] = wrapper.last_graph_attention.detach()
        return collected or None

    def gate_values(self) -> dict[int, float]:
        """Return current tanh-gated scalar values by graph layer."""

        values: dict[int, float] = {}
        for layer_idx, module in self.cross_attentions.items():
            values[int(layer_idx)] = float(torch.tanh(module.alpha.detach()).cpu())
        return values

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        raw_graph_tokens: torch.Tensor | None = None,
        graph_mask: torch.BoolTensor | None = None,
        hop_ids: torch.LongTensor | None = None,
        direction_ids: torch.LongTensor | None = None,
        rank_ids: torch.LongTensor | None = None,
        graph_scale: float | None = None,
        return_graph_attentions: bool = False,
        **kwargs: Any,
    ) -> GraphCausalLMOutput:
        """Run Qwen with optional graph-memory cross-attention."""

        active_scale = self.graph_scale if graph_scale is None else float(graph_scale)

        graph_memory = None
        if (
            raw_graph_tokens is not None
            and graph_mask is not None
            and hop_ids is not None
            and direction_ids is not None
            and rank_ids is not None
            and active_scale != 0.0
        ):
            graph_memory = self.graph_projector(
                raw_graph_tokens=raw_graph_tokens,
                hop_ids=hop_ids,
                direction_ids=direction_ids,
                rank_ids=rank_ids,
                graph_mask=graph_mask,
            )

        self._set_layer_graph_state(
            graph_memory=graph_memory,
            graph_mask=graph_mask,
            text_mask=attention_mask.to(torch.bool),
            graph_scale=active_scale,
            return_graph_attentions=return_graph_attentions,
        )

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            **kwargs,
        )

        return GraphCausalLMOutput(
            logits=outputs.logits,
            loss=getattr(outputs, "loss", None),
            graph_attentions=self._collect_graph_attentions()
            if return_graph_attentions
            else None,
            gate_values=self.gate_values(),
        )

    def save_graph_adapter(self, output_dir: str | Path) -> None:
        """Save only graph-adapter weights, not the frozen base Qwen weights."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "graph_projector": self.graph_projector.state_dict(),
                "cross_attentions": self.cross_attentions.state_dict(),
                "graph_layers": self.graph_layers,
            },
            output_dir / "graph_adapter.pt",
        )

    def load_graph_adapter(self, checkpoint_path: str | Path) -> None:
        """Load graph-adapter weights saved by save_graph_adapter."""

        payload = torch.load(checkpoint_path, map_location="cpu")
        self.graph_projector.load_state_dict(payload["graph_projector"])
        self.cross_attentions.load_state_dict(payload["cross_attentions"])
