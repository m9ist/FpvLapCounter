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


# Мета-строка и статус-плашка фиксированной высоты — кнопки не «прыгают».
_STATIC_CSS = (
    '<style>'
    '.fpv-meta{height:20px;line-height:20px;font-size:13px;color:#55596a;'
    'margin-bottom:8px;'
    'display:flex;justify-content:space-between;white-space:nowrap;overflow:hidden}'
    '.fpv-meta b{color:#31333f}'
    '.fpv-status{height:34px;display:flex;align-items:center;justify-content:center;'
    'gap:8px;border-radius:6px;font-weight:600;font-size:13px}'
    '.fpv-st-ok{background:rgba(33,163,102,0.18);color:#157347}'
    '.fpv-st-bad{background:rgba(224,67,85,0.16);color:#c22f40}'
    '.fpv-st-none{background:rgba(180,130,0,0.22);color:#7a5800}'
    '</style>'
)

# Дизайн-прототип: dev/verify_proto_v2.html.
# Фон на всю карточку; рамка одинаковой толщины во всех состояниях
# (у ok/bad прозрачная), чтобы смена статуса не сдвигала вёрстку.
# «Не проверен» — контрастный янтарный фон + красная рамка.
_CARD_COLORS = {
    True:  ("transparent", "rgba(33,163,102,0.14)"),
    False: ("transparent", "rgba(224,67,85,0.11)"),
    None:  ("#e04355", "#ffe08a"),
}


def _card_css(i: int, verified: bool | None) -> str:
    """CSS одной карточки: живёт внутри её фрагмента и обновляется вместе с ним.

    Виджеты с key получают CSS-класс st-key-<key> (Streamlit 1.39+):
    карточка — st.container(key=f"vcard_{i}"), кнопки — свои key.
    Правила по простым классам: селекторы :has() здесь недопустимы —
    браузер пересчитывает их по всему DOM при каждой перерисовке,
    на ~20 карточках это давало ~1 с на клик.
    """
    border_color, bg_color = _CARD_COLORS[verified]
    rules = (
        f'.st-key-vcard_{i}{{'
        f'border:3px solid {border_color}!important;'
        f'background-color:{bg_color}!important;'
        f'border-radius:10px!important;'
        f'padding:8px!important;'
        f'margin-bottom:4px!important}}'
    )

    def _btn_rules(key: str, bg: str, bg_hover: str) -> str:
        sel = f'.st-key-{key} button'
        return (
            f'{sel}{{background-color:{bg}!important;'
            f'border-color:{bg}!important;color:#fff!important}}'
            f'{sel}:hover{{background-color:{bg_hover}!important}}'
        )

    if verified is True:
        rules += _btn_rules(f'vok_{i}', "#21a366", "#1b8955")
    if verified is False:
        rules += _btn_rules(f'vbad_{i}', "#e04355", "#c73648")

    return f'<style>{rules}</style>'




@st.fragment
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

    Вся вкладка — st.fragment: клики ✅/❌ перезапускают только её
    (st.rerun(scope="fragment")), а не весь app.py. Остальной интерфейс
    (список видео, другие вкладки) обновится при следующем полном rerun —
    например, при переключении вкладки.

    Parameters
    ----------
    video_path         : Absolute path to the video file.
    passes             : List of PassData objects.
    frames_cache       : Mutable dict {frame_idx: bgr_frame} for caching.
    on_verified_change : Callback(pass_idx, True|False|None) when user changes status.
    on_use_as_ref      : Callback(bgr_frame) when user wants to use frame as reference.
    """
    # Inject static CSS in a single top-level call (before any columns).
    # Style tags placed inside st.columns() can behave inconsistently.
    st.markdown(_STATIC_CSS, unsafe_allow_html=True)

    # ── Floating scroll buttons ────────────────────────────────────────
    # st.markdown innerHTML never executes <script> tags (browser security).
    # st.iframe() renders an iframe where JS *does* run.
    # From the iframe we reach window.parent.document to inject buttons
    # directly into the Streamlit page and scroll section.stMain.
    st.iframe("""
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
""", height=1)  # 0 st.iframe не принимает (в отличие от components.v1.html)

    if not passes:
        st.info("Нет обнаруженных пролётов для верификации. Запустите анализ.")
        return

    # ── Bulk action buttons ────────────────────────────────────────────
    # on_click-колбэки выполняются ДО перерисовки фрагмента — статусы уже
    # новые к моменту рендера, и второй прогон (st.rerun) не нужен.
    def _set_all(value: bool | None) -> None:
        for i in range(len(passes)):
            on_verified_change(i, value)

    def _fake_unverified() -> None:
        for i, p in enumerate(passes):
            if p.verified is None:
                on_verified_change(i, False)

    def _smart() -> None:
        res = on_smart_filter()
        if res is not None:
            st.toast(f"🧠 Помечено фейком: {res}")

    cols = st.columns(4 if on_smart_filter else 3)
    with cols[0]:
        st.button("✅ Подтвердить все", key="verify_all_btn", width='stretch',
                  on_click=_set_all, args=(True,))
    with cols[1]:
        st.button("❌ Непроверенные — фейк", key="verify_all_fake_btn", width='stretch',
                  on_click=_fake_unverified)
    with cols[2]:
        st.button("🔄 Сбросить верификацию", key="verify_reset_btn", width='stretch',
                  on_click=_set_all, args=(None,))
    if on_smart_filter:
        with cols[3]:
            st.button("🧠 Умный фильтр", key="verify_smart_btn", width='stretch',
                      help="Классификатор пометит фейки ❌; остальных не трогает",
                      on_click=_smart)

    st.markdown(f"**Всего кандидатов: {len(passes)}**")
    st.markdown("---")

    # ── Pass grid (3 per row) ──────────────────────────────────────────
    cols_per_row = 3

    # Кеш готовых JPEG-превью: st.image(numpy) заново кодирует в PNG и
    # хеширует массив при каждом rerun — на 30+ карточках это заметно.
    thumb_cache: dict = (
        st.session_state.setdefault("_verify_thumb_cache", {})
        .setdefault(video_path, {})
    )

    for row_start in range(0, len(passes), cols_per_row):
        row_passes = passes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col_offset, (col, pass_data) in enumerate(zip(cols, row_passes)):
            pass_idx = row_start + col_offset
            # Фрагмент вызывается ВНУТРИ with col: — фрагмент не может писать
            # виджеты в контейнер, созданный снаружи него.
            with col:
                _render_pass_card(
                    pass_data=pass_data,
                    pass_idx=pass_idx,
                    video_path=video_path,
                    frames_cache=frames_cache,
                    thumb_cache=thumb_cache,
                    on_verified_change=on_verified_change,
                    on_use_as_ref=on_use_as_ref,
                )


@st.fragment
def _render_pass_card(
    pass_data: PassData,
    pass_idx: int,
    video_path: str,
    frames_cache: dict,
    thumb_cache: dict,
    on_verified_change: Callable,
    on_use_as_ref: Callable,
) -> None:
    """Render a single verification card (its own fragment).

    Клик ✅/❌ перерисовывает ТОЛЬКО эту карточку: ререндер всей сетки
    (~17 карточек одним delta-пакетом) занимал у браузера ~1 с.
    CSS карточки поэтому встроен в её же markdown (_card_css).
    """
    # Контейнер с key даёт класс st-key-vcard_{i} — _card_css красит
    # фон/рамку карточки по нему.
    with st.container(key=f"vcard_{pass_idx}"):
        # Thumbnail: готовые JPEG-байты из кеша (imencode принимает BGR).
        jpg = thumb_cache.get(pass_data.frame)
        if jpg is None:
            frame_bgr = _load_frame(video_path, pass_data.frame, frames_cache)
            if frame_bgr is not None:
                thumb = cv2.resize(frame_bgr, (224, 126))
                ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    jpg = buf.tobytes()
                    thumb_cache[pass_data.frame] = jpg
        if jpg is not None:
            st.image(jpg, width='stretch')
        else:
            st.warning("Кадр недоступен")

        # Pass info + status: свои div-ы фиксированной высоты вместо
        # st.caption/st.success/st.error/st.info — у тех высота зависит от
        # состояния, из-за чего кнопки «прыгали» (CSS-классы в _build_css).
        t = pass_data.time_sec
        minutes = int(t // 60)
        seconds = t - minutes * 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"

        if pass_data.verified is True:
            status_cls, status_txt = "fpv-st-ok", "✅ Подтверждён"
        elif pass_data.verified is False:
            status_cls, status_txt = "fpv-st-bad", "❌ Фейк"
        else:
            status_cls, status_txt = "fpv-st-none", "⬜ Не проверен"

        # CSS карточки — в том же markdown-блоке (отдельный элемент занял бы
        # слот в flex-gap колонки и сдвинул вёрстку).
        st.markdown(
            _card_css(pass_idx, pass_data.verified) +
            f'<div class="fpv-meta">'
            f'<span><b>#{pass_idx + 1}</b> · {time_str}</span>'
            f'<span>сходство {pass_data.similarity:.3f}</span></div>'
            f'<div class="fpv-status {status_cls}">{status_txt}</div>',
            unsafe_allow_html=True,
        )

        # Action buttons — цвет активной кнопки CSS находит по классу
        # st-key-vok_{i} / st-key-vbad_{i} её контейнера.
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            ok_type = "primary" if pass_data.verified is True else "secondary"
            # on_click — см. комментарий у bulk-кнопок: один прогон вместо двух.
            st.button("✅", key=f"vok_{pass_idx}", width='stretch',
                      type=ok_type, help="Подтвердить пролёт",
                      on_click=lambda i=pass_idx, v=pass_data.verified:
                          on_verified_change(i, None if v is True else True))

        with btn_col2:
            bad_type = "primary" if pass_data.verified is False else "secondary"
            st.button("❌", key=f"vbad_{pass_idx}", width='stretch',
                      type=bad_type, help="Отметить как фейк",
                      on_click=lambda i=pass_idx, v=pass_data.verified:
                          on_verified_change(i, None if v is False else False))

        with btn_col3:
            if st.button("🎯", key=f"vref_{pass_idx}", width='stretch',
                         help="Использовать как референс"):
                # BGR-кадр нужен только здесь — грузим лениво (обычно уже в кеше).
                # Фидбэк — st.toast в on_use_as_ref; инлайновый st.success ломал
                # фиксированную высоту карточки.
                frame_bgr = _load_frame(video_path, pass_data.frame, frames_cache)
                if frame_bgr is not None:
                    on_use_as_ref(frame_bgr)
