"""
Сборка результата детектора ИИ (задача 9).

Вынесено из worker, чтобы поверх OLD-базы подключить минимально:

1. ai_percent = null, если сигнала нет (раньше часто уходило 0 → «точно человек»);
2. ai_fragments — по-чанковая локализация (какой кусок похож на ИИ).

Зависит только от ai_scoring (уже был в OLD): ai_percent_from_scores, sigmoid_ai_percent.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ai_scoring import AiConfig, ai_percent_from_scores, sigmoid_ai_percent


def resolve_ai_percent(
    *,
    enabled: bool,
    scores: Sequence[float],
    config: AiConfig,
) -> Optional[float]:
    """
    Итоговый % ИИ по документу.

    None — сигнала нет (детектор выключен или нет scored-окон).
    float — измеренный процент (в т.ч. 0.0 = «похоже на человека»).
    """
    if not enabled or not scores:
        return None
    return ai_percent_from_scores(list(scores), config)


def build_ai_fragments(
    ai_pairs: Sequence[tuple[str, float]],
    config: AiConfig,
    *,
    max_fragments: int = 50,
    preview_chars: int = 300,
) -> list[dict[str, Any]]:
    """
    Локализация: список фрагментов с d_score и ai_probability.

    ai_pairs — [(текст_окна, D), ...] после Fast-DetectGPT.
    """
    limit = max(0, int(max_fragments))
    preview = max(0, int(preview_chars))
    out: list[dict[str, Any]] = []
    for chunk, score in list(ai_pairs)[:limit]:
        out.append(
            {
                "text": (chunk or "")[:preview],
                "d_score": round(float(score), 4),
                "ai_probability": sigmoid_ai_percent(
                    float(score), config.d0, config.scale
                ),
            }
        )
    return out
