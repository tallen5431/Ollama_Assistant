"""The shell the rest of the app is drawn inside.

Measured on a 390x844 phone and a 1512x900 desktop before this: the header put
four things in a row and truncated the app's own name to fit; the conversation
list dimmed three quarters of a desktop to show a column of titles; the footer
spent 293px on three stacked rows; and a whole empty screen held one grey
sentence. These cover the parts of the answer that a browser cannot check for
itself.
"""

from __future__ import annotations

import re

import chat_ui
from conftest import page_script


def page() -> str:
    return chat_ui.render_page("Ollama Chat")


class TestTheSidebarIsAColumnWhereThereIsRoom:
    """A list of your own conversations is not a dialog. On a screen with
    1100px going spare next to it, dimming the app to show it — and taking it
    away again the moment you pick one — is the overlay's behaviour leaking
    into a place that has room for furniture.
    """

    def test_a_wide_screen_gets_a_rail_rather_than_an_overlay(self):
        text = page()
        at = text.index(".drawer { position:static;")
        media = text.rindex("@media", 0, at)
        assert "min-width: 1024px" in text[media:at]

    def test_the_rail_is_undimmed_without_disabling_the_backdrop_itself(self):
        """Hiding .backdrop in the desktop media query took it away from the
        photo-details sheet too, which then had only its ✕ to close it."""
        text = page()
        at = text.index(".drawer { position:static;")
        media = text.rindex("@media", 0, at)
        assert "backdrop { display:none" not in text[media:media + 400]
        js = page_script(text)
        assert 'backdropEl.addEventListener("click", () => { dismissDrawer(); ' \
               'hidePhotoDetails(); });' in js

    def test_it_is_open_before_you_ask_for_it(self):
        assert 'if (railed()) { drawerEl.hidden = false; showPane("chats"); }' in page()

    def test_picking_a_conversation_does_not_dismiss_the_rail(self):
        """closeDrawer here collapsed the sidebar on every single click."""
        js = page_script(page())
        at = js.index("function dismissDrawer")
        assert "if (!railed()) closeDrawer();" in js[at:at + 120]
        # the two places a chosen thread lands
        assert js.count("dismissDrawer();") >= 2

    def test_the_backdrop_is_only_for_the_overlay(self):
        js = page_script(page())
        at = js.index("function openDrawer")
        assert "backdropEl.hidden = railed();" in js[at:at + 220]

    def test_narrowing_with_the_rail_open_closes_it(self):
        """Otherwise it covers the conversation with nothing dimmed to say so."""
        js = page_script(page())
        at = js.index("wideScreen.addEventListener")
        window = js[at:at + 400]
        assert "else if (!drawerEl.hidden) closeDrawer();" in window


class TestTheBarNamesTheConversation:
    """The app's own name told you what you already knew."""

    def test_the_header_carries_the_thread_not_the_app(self):
        text = page()
        assert '<h2 id="chatTitle">New chat</h2>' in text
        # the app still names itself, once, at the top of the drawer
        assert text.index('class="brand"') < text.index('id="chatTitle"')

    def test_opening_a_thread_names_it(self):
        assert "setChatTitle(convo.title);" in page_script(page())

    def test_a_server_titled_thread_catches_up(self):
        """The title is written server-side from the first message, so the bar
        would have sat on "New chat" for the whole of the first conversation.
        """
        js = page_script(page())
        at = js.index("async function refreshConversations")
        window = js[at:js.index("function ", at + 40) + 4000]
        assert "if (convo.id === currentConvoId) setChatTitle(convo.title);" in window

    def test_a_new_chat_resets_it(self):
        js = page_script(page())
        at = js.index("function newChat()")
        assert 'setChatTitle("");' in js[at:at + 500]


class TestTheme:
    """A phone used in bed and at a desk wants both."""

    def test_both_schemes_exist_without_a_script(self):
        text = page()
        assert "@media (prefers-color-scheme: dark)" in text
        assert ':root[data-theme="dark"]' in text
        assert ':root[data-theme="light"]' in text

    def test_the_saved_choice_is_applied_before_the_stylesheet(self):
        """Applied from the script at the end of <body> it paints the other
        theme first, and a white flash at 2am is why someone chose dark."""
        text = page()
        assert text.index('localStorage.getItem("theme")') < text.index("<style>")

    def test_the_choice_is_remembered(self):
        js = page_script(page())
        at = js.index("themeBtn.addEventListener")
        window = js[at:at + 600]
        assert 'localStorage.setItem("theme", next);' in window
        assert 'next = now === "dark" ? "light" : "dark"' in window

    def test_a_browser_with_no_storage_still_switches(self):
        """Private mode throws on setItem; the theme must still change."""
        js = page_script(page())
        at = js.index("themeBtn.addEventListener")
        window = js[at:at + 600]
        assert window.index('setAttribute("data-theme", next)') < \
            window.index("localStorage.setItem"), \
            "apply first, then try to remember it"
        assert "try { localStorage.setItem" in window

    def test_the_address_bar_follows_too(self):
        text = page()
        assert 'name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)"' in text
        assert 'name="theme-color" content="#0a0e17" media="(prefers-color-scheme: dark)"' in text


class TestTheEmptyScreenOffersSomething:
    """It is the only moment the app gets to say what it is for, and on this
    app the answer is usually a routine.
    """

    def test_the_routines_are_the_offer(self):
        js = page_script(page())
        at = js.index("function paintEmpty")
        window = js[at:at + 1800]
        assert "routines.slice(0, 4)" in window
        assert "pickRoutine(routine)" in window

    def test_a_routine_name_is_never_written_as_markup(self):
        """Anyone who can reach this app could have written it."""
        js = page_script(page())
        at = js.index("function paintEmpty")
        window = js[at:at + 1800]
        assert "name.textContent = routine.name;" in window
        assert "innerHTML = routine" not in window

    def test_it_repaints_when_the_routines_arrive(self):
        """They are fetched after first paint, so the cards were never drawn."""
        js = page_script(page())
        at = js.index("renderRoutineChips();\n        paintEmpty();")
        assert at > 0

    def test_a_fresh_install_still_gets_the_headline(self):
        """With no routines saved, the early return must come after the text."""
        js = page_script(page())
        at = js.index("function paintEmpty")
        window = js[at:at + 1800]
        assert window.index("host.appendChild(sub)") < window.index("if (!routines.length) return;")


class TestIconsAreOneSetOfControls:
    """Emoji are drawn by the platform in full colour at whatever weight it
    likes: a paperclip, a camera and a screenshot button sat side by side in
    three different visual styles, and at 17px two of them were the same
    picture of a camera.
    """

    ICON_ONLY = ("menu", "drawerClose", "metaClose", "attach", "camera", "shot",
                 "mic", "theme", "routineEditBtn")

    def test_every_icon_only_control_uses_the_sprite(self):
        text = page()
        for name in self.ICON_ONLY:
            at = text.index('id="' + name + '"')
            tag = text[at:text.index("</button>", at)]
            assert "<use href=\"#i-" in tag, name

    def test_no_emoji_is_left_standing_in_for_an_icon(self):
        text = page()
        for name in self.ICON_ONLY:
            at = text.index('id="' + name + '"')
            tag = text[at:text.index("</button>", at)]
            # everything outside the tags and the title attribute
            visible = re.sub(r"<[^>]*>", "", tag)
            assert not [c for c in visible if ord(c) > 0x2100], (name, visible)

    def test_every_symbol_the_page_uses_is_defined(self):
        text = page()
        defined = set(re.findall(r'<symbol id="(i-[a-z]+)"', text))
        used = set(re.findall(r'<use href="#(i-[a-z]+)"', text))
        assert used - defined == set(), used - defined
        assert defined - used == set(), "an icon nothing points at"

    def test_the_microphone_swaps_glyph_without_being_rewritten(self):
        """textContent on that button would have deleted both SVGs."""
        text = page()
        assert "#mic .i-stop, #mic.rec .i-mic { display:none; }" in text
        assert "micBtn.textContent" not in text


class TestTheComposerIsOneObject:
    """The box, the photos riding with the message and the buttons that act on
    it were three rows that happened to be adjacent.
    """

    def test_they_are_one_card(self):
        text = page()
        card = text.index('class="composer-card"')
        assert card < text.index('id="thumbs"') < text.index('class="composer"')
        assert text.index('id="input"') > text.index('class="composer"')

    def test_the_box_gets_its_own_row(self):
        """Four buttons plus a textarea in one row collapsed the input to 63px
        at 360px wide, and to a 0px content box at 320px."""
        text = page()
        assert "flex:1 1 100%; order:-1;" in text

    def test_send_sits_at_the_end_whichever_button_is_showing(self):
        assert "#send, #stop { margin-left:auto;" in page()

    def test_the_card_shows_the_focus_rather_than_the_box_inside_it(self):
        text = page()
        assert ".composer-card:focus-within { border-color:var(--accent);" in text
        assert "textarea:focus { outline:none; }" in text


class TestGettingBackToTheLatest:
    """Scrolling up during a long reply stops the view following it — right,
    but then getting back meant flicking down through everything that arrived
    while you were reading.
    """

    def page(self):
        return chat_ui.render_page("t")

    def test_the_button_is_the_follow_flag_seen_twice(self):
        js = page_script(self.page())
        at = js.index("function showToLatest")
        assert "toLatestEl.hidden = stickBottom ||" in js[at:at + 250]

    def test_taking_it_goes_through_the_same_path_the_stream_does(self):
        """Two ways to reach the bottom is two things to keep in step."""
        js = page_script(self.page())
        assert 'toLatestEl.addEventListener("click", () => scrollDown(true));' in js

    def test_it_does_not_appear_when_there_is_nothing_to_scroll(self):
        js = page_script(self.page())
        at = js.index("function showToLatest")
        assert "chatEl.scrollHeight <= chatEl.clientHeight + 8" in js[at:at + 250]

    def test_the_conversation_never_scrolls_smoothly(self):
        """Following assigns scrollTop on every token. Animated, those
        assignments fire scroll events part of the way there, atBottom() reads
        that as the user scrolling away, and the follow switches itself off
        mid-reply — measured on a 1512px desktop.
        """
        text = self.page()
        at = text.index("      #chat {")
        assert "scroll-behavior" not in text[at:text.index("}", at)]


class TestTheListHasSomewhereForTheEyeToLand:
    def page(self):
        return chat_ui.render_page("t")

    def test_conversations_are_grouped_by_day(self):
        js = page_script(self.page())
        at = js.index("function dayOf")
        window = js[at:at + 900]
        for label in ('"Today"', '"Yesterday"', 'weekday: "long"'):
            assert label in window, label

    def test_a_day_is_announced_only_when_it_changes(self):
        """The list is newest first, so they already come out in order."""
        js = page_script(self.page())
        assert "if (day !== lastDay) {" in js

    def test_a_thread_from_another_year_says_which(self):
        js = page_script(self.page())
        at = js.index("function dayOf")
        assert 'year: "numeric"' in js[at:at + 900]


class TestTheKeyboardReachesIt:
    def page(self):
        return chat_ui.render_page("t")

    def test_ctrl_k_opens_the_drawer_on_the_search_box(self):
        js = page_script(self.page())
        at = js.index('e.key === "k"')
        window = js[at - 200:at + 300]
        assert 'openDrawer("chats");' in window and "searchEl.focus()" in window
        assert "e.metaKey" in window, "⌘K on a Mac"

    def test_escape_closes_the_innermost_thing_first(self):
        js = page_script(self.page())
        at = js.index('if (e.key !== "Escape") return;')
        window = js[at:at + 300]
        assert window.index("hidePhotoDetails()") < window.index("closeDrawer()")

    def test_escape_leaves_a_rail_alone(self):
        """There is nothing over the top of anything to dismiss."""
        js = page_script(self.page())
        at = js.index('if (e.key !== "Escape") return;')
        assert "!drawerEl.hidden && !railed()" in js[at:at + 300]


class TestItRespectsALessMotionSetting:
    def test_animations_and_smooth_scrolling_are_turned_off(self):
        text = chat_ui.render_page("t")
        at = text.index("@media (prefers-reduced-motion: reduce)")
        window = text[at:at + 400]
        assert "animation-duration:0.01ms !important" in window
        assert "transition-duration:0.01ms !important" in window
        assert "scroll-behavior:auto !important" in window
