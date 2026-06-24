"""
Verification grid tab for FPV lap timing app.
Lets the user confirm or reject each detected gate pass candidate.
"""
from __future__ import annotations

from typing import Callable

import cv2
import streamlit as st

from core.frames import load_frame_cached as _load_frame
from storage.project import PassData


def _build_css(passes: list) -> str:
    """Build one <style> block with all per-card rules.

    Card border rule:
      Target the innermost stVerticalBlock containing the anchor span using
      :not(:has(child stVerticalBlock that also has the same span)).

    Button color rules:
      Same innermost-stVerticalBlock trick but scoped to each btn sub-column.
      The marker span is placed inside btn_col1 / btn_col2; its paragraph
      container is collapsed to zero height so it doesn't affect layout.
    """
    rules: list[str] = []

    color_map = {
        True:  ("#28a745", "rgba(40,167,69,0.12)"),
        False: ("#dc3545", "rgba(220,53,69,0.10)"),
        None:  ("#6c757d", "transparent"),
    }

    for i, p in enumerate(passes):
        anchor  = f"card-anchor-{i}"
        ok_mid  = f"vok-m-{i}"
        bad_mid = f"vbad-m-{i}"

        border_color, bg_color = color_map[p.verified]

        # ── Card border ───────────────────────────────────────────────
        # Innermost stVerticalBlock = the one that contains the anchor but
        # has NO child stVerticalBlock that also contains the anchor.
        card_sel = (
            f'div[data-testid="stVerticalBlock"]:has(span#{anchor})'
            f':not(:has(div[data-testid="stVerticalBlock"]:has(span#{anchor})))'
        )
        rules.append(
            f'{card_sel}{{'
            f'border:3px solid {border_color}!important;'
            f'background-color:{bg_color}!important;'
            f'border-radius:8px!important;'
            f'padding:6px!important;'
            f'margin-bottom:4px!important}}'
        )

        # ── Collapse the marker-span paragraph so it takes no space ──
        for mid in (ok_mid, bad_mid):
            rules.append(
                f'div[data-testid="stMarkdownContainer"]:has(span#{mid})'
                f'{{line-height:0!important;margin:0!important;padding:0!important}}'
                f'div[data-testid="stMarkdownContainer"]:has(span#{mid}) p'
                f'{{margin:0!important;padding:0!important;line-height:0!important;'
                f'height:0!important;overflow:hidden!important}}'
            )

        # ── Button colours ────────────────────────────────────────────
        def _btn_rules(mid: str, bg: str, bg_hover: str) -> str:
            # Innermost stVerticalBlock for the button sub-column
            sel = (
                f'div[data-testid="stVerticalBlock"]:has(span#{mid})'
                f':not(:has(div[data-testid="stVerticalBlock"]:has(span#{mid})))'
                f' button'
            )
            return (
                f'{sel}{{background-color:{bg}!important;'
                f'border-color:{bg}!important;color:#fff!important}}'
                f'{sel}:hover{{background-color:{bg_hover}!important}}'
            )

        if p.verified is True:
            rules.append(_btn_rules(ok_mid, "#28a745", "#218838"))
        if p.verified is False:
            rules.append(_btn_rules(bad_mid, "#dc3545", "#c82333"))

    return f'<style>{"".join(rules)}</style>' if rules else ''




def render_verify_tab(
    video_path: str,
    passes: list[PassData],
    frames_cache: dict,
    on_verified_change: Callable,  # (pass_idx: int, value: bool | None) -> None
    on_use_as_ref: Callable,       # (bgr_frame: np.ndarray) -> None
    on_smart_filter: Callable | None = None,  # () -> n_marked_fake | None
) -> None:
    """
    Render a verification grid for all detected gate pass candidates.

    Parameters
    ----------
    video_path         : Absolute path to the video file.
    passes             : List of PassData objects.
    frames_cache       : Mutable dict {frame_idx: bgr_frame} for caching.
    on_verified_change : Callback(pass_idx, True|False|None) when user changes status.
    on_use_as_ref      : Callback(bgr_frame) when user wants to use frame as reference.
    """
    # Inject all CSS in a single top-level call (before any columns).
    # Style tags placed inside st.columns() can behave inconsistently.
    st.markdown(_build_css(passes), unsafe_allow_html=True)

    # ── Floating scroll buttons ────────────────────────────────────────
    # st.markdown innerHTML never executes <script> tags (browser security).
    # st.components.v1.html() renders an iframe where JS *does* run.
    # From the iframe we reach window.parent.document to inject buttons
    # directly into the Streamlit page and scroll section.stMain.
    st.components.v1.html("""
<script>
(function () {
    var doc = window.parent.document;
    if (doc.getElementById('fpv-scroll-top')) return;

    var style = doc.createElement('style');
    style.textContent = [
        '.fpv-scroll-btn{position:fixed;right:24px;z-index:99999;',
        'width:42px;height:42px;border-radius:50%;border:none;',
        'cursor:pointer;font-size:20px;color:#fff;background:#1f77b4;',
        'box-shadow:0 2px 8px rgba(0,0,0,.45);',
        'display:flex;align-items:center;justify-content:center;',
        'opacity:.8;transition:opacity .2s,transform .15s}',
        '.fpv-scroll-btn:hover{opacity:1;transform:scale(1.12)}'
    ].join('');
    doc.head.appendChild(style);

    function doScroll(toBottom) {
        var el = doc.querySelector('section.stMain');
        if (el) el.scrollTo({top: toBottom ? 999999 : 0, behavior: 'smooth'});
    }

    function btn(id, bottom, label, toBottom) {
        var b = doc.createElement('button');
        b.id = id;
        b.className = 'fpv-scroll-btn';
        b.style.bottom = bottom;
        b.title = label;
        b.textContent = toBottom ? '\u2193' : '\u2191';
        b.addEventListener('click', function () { doScroll(toBottom); });
        doc.body.appendChild(b);
    }

    btn('fpv-scroll-top', '78px', '\u041d\u0430\u0432\u0435\u0440\u0445', false);
    btn('fpv-scroll-bot', '26px', '\u0412\u043d\u0438\u0437',  true);
}());
</script>
""", height=0)

    if not passes:
        st.info("Нет обнаруженных пролётов для верификации. Запустите анализ.")
        return

    # ── Bulk action buttons ────────────────────────────────────────────
    cols = st.columns(4 if on_smart_filter else 3)
    with cols[0]:
        if st.button("✅ Подтвердить все", key="verify_all_btn", width='stretch'):
            for i in range(len(passes)):
                on_verified_change(i, True)
            st.rerun()
    with cols[1]:
        if st.button("❌ Непроверенные — фейк", key="verify_all_fake_btn", width='stretch'):
            for i, p in enumerate(passes):
                if p.verified is None:
                    on_verified_change(i, False)
            st.rerun()
    with cols[2]:
        if st.button("🔄 Сбросить верификацию", key="verify_reset_btn", width='stretch'):
            for i in range(len(passes)):
                on_verified_change(i, None)
            st.rerun()
    if on_smart_filter:
        with cols[3]:
            if st.button("🧠 Умный фильтр", key="verify_smart_btn", width='stretch',
                         help="Классификатор пометит фейки ❌; остальных не трогает"):
                res = on_smart_filter()
                if res is not None:
                    st.toast(f"🧠 Помечено фейком: {res}")
                st.rerun()

    st.markdown(f"**Всего кандидатов: {len(passes)}**")
    st.markdown("---")

    # ── Pass grid (3 per row) ──────────────────────────────────────────
    cols_per_row = 3

    for row_start in range(0, len(passes), cols_per_row):
        row_passes = passes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col_offset, (col, pass_data) in enumerate(zip(cols, row_passes)):
            pass_idx = row_start + col_offset
            _render_pass_card(
                col=col,
                pass_data=pass_data,
                pass_idx=pass_idx,
                video_path=video_path,
                frames_cache=frames_cache,
                on_verified_change=on_verified_change,
                on_use_as_ref=on_use_as_ref,
            )


def _render_pass_card(
    col,
    pass_data: PassData,
    pass_idx: int,
    video_path: str,
    frames_cache: dict,
    on_verified_change: Callable,
    on_use_as_ref: Callable,
) -> None:
    """Render a single verification card inside a Streamlit column."""
    with col:
        # Anchor span — CSS in render_verify_tab targets the stVerticalBlock
        # that contains this span (and only this span) to draw the card border.
        anchor_id = f"card-anchor-{pass_idx}"
        st.markdown(
            f'<span id="{anchor_id}" style="display:none"></span>',
            unsafe_allow_html=True,
        )

        # Thumbnail
        frame_bgr = _load_frame(video_path, pass_data.frame, frames_cache)
        if frame_bgr is not None:
            thumb = cv2.resize(frame_bgr, (224, 126))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            st.image(thumb_rgb, width='stretch')
        else:
            st.warning("Кадр недоступен")

        # Pass info
        t = pass_data.time_sec
        minutes = int(t // 60)
        seconds = t - minutes * 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"

        st.caption(
            f"**#{pass_idx + 1}** · {time_str}\n\n"
            f"Сходство: {pass_data.similarity:.3f}"
        )

        # Status indicator
        if pass_data.verified is True:
            st.success("✅ Подтверждён", icon=None)
        elif pass_data.verified is False:
            st.error("❌ Фейк", icon=None)
        else:
            st.info("⬜ Не проверен", icon=None)

        # Action buttons — each btn sub-column gets a hidden marker span so
        # CSS can scope the button colour to exactly that column.
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            # Marker for ✅ colour rule; paragraph is collapsed to zero by CSS.
            st.markdown(
                f'<span id="vok-m-{pass_idx}" style="display:none"></span>',
                unsafe_allow_html=True,
            )
            ok_type = "primary" if pass_data.verified is True else "secondary"
            if st.button("✅", key=f"vok_{pass_idx}", width='stretch',
                         type=ok_type, help="Подтвердить пролёт"):
                new_val = None if pass_data.verified is True else True
                on_verified_change(pass_idx, new_val)
                st.rerun()

        with btn_col2:
            # Marker for ❌ colour rule.
            st.markdown(
                f'<span id="vbad-m-{pass_idx}" style="display:none"></span>',
                unsafe_allow_html=True,
            )
            bad_type = "primary" if pass_data.verified is False else "secondary"
            if st.button("❌", key=f"vbad_{pass_idx}", width='stretch',
                         type=bad_type, help="Отметить как фейк"):
                new_val = None if pass_data.verified is False else False
                on_verified_change(pass_idx, new_val)
                st.rerun()

        with btn_col3:
            if st.button("🎯", key=f"vref_{pass_idx}", width='stretch',
                         help="Использовать как референс"):
                if frame_bgr is not None:
                    on_use_as_ref(frame_bgr)
                    st.success("Добавлен как референс")
