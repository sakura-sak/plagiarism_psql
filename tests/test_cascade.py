from cascade import (
    EXACT,
    PARAPHRASE,
    SEMANTIC,
    classify_match,
    content_tokens,
    paraphrase_similarity,
)


def test_content_tokens_drop_stopwords_and_short_words():
    tokens = content_tokens("И в работе рассмотрены методы поиска")
    assert "и" not in tokens
    assert "в" not in tokens
    assert "работе" in tokens
    assert "методы" in tokens


def test_paraphrase_similarity_is_high_for_shared_content_words():
    a = "методы поиска похожих документов в корпусе университета"
    b = "методы поиска похожих документов применяются в корпусе университета"
    assert paraphrase_similarity(a, b) > 0.15


def test_paraphrase_similarity_is_low_for_unrelated_texts():
    assert paraphrase_similarity("сортировка массива пузырьком", "рецепт борща со сметаной") < 0.15


def test_classify_match_cascade_exact_then_paraphrase_then_semantic():
    assert classify_match(0.9, 0.0, lexical_threshold=0.6) == EXACT
    assert classify_match(0.2, 0.4, lexical_threshold=0.6) == PARAPHRASE
    assert classify_match(0.1, 0.05, lexical_threshold=0.6) == SEMANTIC


def test_fragment_is_exact_for_lexical_hit_and_capped():
    from cascade import MAX_FRAGMENTS_PER_SOURCE, append_fragment, make_fragment

    query = "методы поиска похожих документов в корпусе университета подробно"
    source = "методы поиска похожих документов в корпусе университета подробно"
    frag = make_fragment(query, source, 0.91, "lexical", lexical_threshold=0.6)
    assert frag["match_type"] == EXACT
    assert frag["query_text"] == query
    assert frag["source_text"] == source

    bag = []
    for i in range(MAX_FRAGMENTS_PER_SOURCE + 5):
        append_fragment(bag, {"query_text": str(i), "source_text": "", "score": 0.1, "match_type": SEMANTIC})
    assert len(bag) == MAX_FRAGMENTS_PER_SOURCE


def test_fragment_is_semantic_when_words_do_not_overlap():
    from cascade import make_fragment

    frag = make_fragment(
        "сортировка массива пузырьком на практике",
        "рецепт борща со сметаной и чесноком",
        0.82,
        "semantic",
        lexical_threshold=0.6,
    )
    assert frag["match_type"] == SEMANTIC
