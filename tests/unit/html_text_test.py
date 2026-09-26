from latinitas_cards.html_text import source_html_to_text


def test_div_and_break_tags_become_line_boundaries() -> None:
    assert source_html_to_text("erste Bedeutung<div>zweite Bedeutung</div>") == "erste Bedeutung\nzweite Bedeutung"
    assert source_html_to_text("a<br>b<br/>c<br />d") == "a\nb\nc\nd"
    assert source_html_to_text("<p>Absatz</p>") == "Absatz"
    assert source_html_to_text("<div>a</div><div>b</div>") == "a\nb"


def test_nested_formatting_tags_are_flattened_while_text_is_kept() -> None:
    assert source_html_to_text("<div><b>äuße<i>re</i> Bedeutung</b></div>") == "äußere Bedeutung"
    assert source_html_to_text("<span>sequ<i>or</i></span>") == "sequor"


def test_entities_decode_once_without_encoded_formatting_literals() -> None:
    assert source_html_to_text("Ruh&apos;&nbsp;&amp;&nbsp;Träume") == "Ruh' & Träume"
    assert source_html_to_text("erste&lt;br&gt;zweite") == "erste\nzweite"
    assert source_html_to_text("a &amp;amp; b") == "a &amp; b"


def test_plain_text_with_umlauts_and_macrons_passes_through_unchanged() -> None:
    assert source_html_to_text("dīcere, dīcō — prüfen") == "dīcere, dīcō — prüfen"
    assert source_html_to_text("  führen  und  \t machen  ") == "führen und machen"
    assert source_html_to_text("amāre|amare") == "amāre|amare"
    assert source_html_to_text("3 < 5 und 6 > 4") == "3 < 5 und 6 > 4"


def test_active_and_event_markup_cannot_survive_as_markup_or_text() -> None:
    assert source_html_to_text("<script>alert(1)</script>sagen") == "sagen"
    assert source_html_to_text("a<img src=x onerror=alert(1)>b") == "ab"
    assert source_html_to_text('<div onclick="steal()">sagen</div>') == "sagen"
    assert source_html_to_text("<!-- hidden -->sagen") == "sagen"


def test_attribute_values_with_quoted_greater_than_do_not_leak_fragments() -> None:
    assert source_html_to_text('<span title="a > b">sagen</span>') == "sagen"
    assert source_html_to_text("<span title='a > b'>sagen</span>") == "sagen"
    assert source_html_to_text('<div title="a > b" class="gloss">sagen</div>') == "sagen"
    assert source_html_to_text('<img alt="a > b">sagen') == "sagen"


def test_encoded_comments_are_dropped_like_real_comments() -> None:
    assert source_html_to_text("&lt;!-- hidden --&gt;sagen") == "sagen"
    assert source_html_to_text("a &lt;!-- versteckt --&gt; b") == "a b"


def test_script_and_style_contents_are_dropped_entirely() -> None:
    assert source_html_to_text('<style media="all">.gloss{color:red}</style>sagen') == "sagen"
    assert source_html_to_text('<script type="text/javascript">alert(1)</script>sagen') == "sagen"
    assert source_html_to_text("sagen<style>.gloss{}</style>machen") == "sagenmachen"


def test_output_is_deterministic_for_identical_input() -> None:
    value = "x<div>y<br>&amp;z<script>bad()</script></div>"

    assert source_html_to_text(value) == source_html_to_text(value)
    assert source_html_to_text(value) == "x\ny\n&z"
