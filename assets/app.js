/* Shared runtime: theme toggle, collapsible sidebar, pager, reading progress, quizzes, copy buttons,
   font-size control, back-to-top, figure/widget lightbox, deep-dive index, keyboard shortcuts.
   Works from file:// with no build step. */
(function () {
  "use strict";
  var BOOK = window.BOOK || { parts: [] };
  var ALL = [];
  BOOK.parts.forEach(function (p) { p.chapters.forEach(function (c) { c.part = p; ALL.push(c); }); });

  function store(key, val) { try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; } }
  var root = document.documentElement;

  /* ---------- theme (dark by default) ---------- */
  function applyTheme(t) {
    root.setAttribute("data-theme", t === "light" ? "light" : "dark");
    var b = document.querySelector(".theme-toggle");
    if (b) { b.textContent = isDark() ? "🌙 Dark" : "☀️ Light"; }
  }
  function isDark() { return root.getAttribute("data-theme") !== "light"; }
  applyTheme(store("ml-theme"));
  function toggleTheme() {
    var next = isDark() ? "light" : "dark";
    applyTheme(next); store("ml-theme", next);
  }

  /* ---------- font size (15–21px on <html>) ---------- */
  var FS_MIN = 15, FS_MAX = 21;
  function fontSize() { var v = parseInt(store("ml-font"), 10); return isNaN(v) ? null : Math.max(FS_MIN, Math.min(FS_MAX, v)); }
  function applyFont(px) {
    if (px) root.style.setProperty("--base-fs", px + "px"); else root.style.removeProperty("--base-fs");
    var cur = px || parseFloat(getComputedStyle(root).fontSize) || 17;
    var dn = document.querySelector(".font-ctl .fs-dn"), up = document.querySelector(".font-ctl .fs-up");
    if (dn) dn.disabled = cur <= FS_MIN;
    if (up) up.disabled = cur >= FS_MAX;
  }
  applyFont(fontSize());
  function stepFont(delta) {
    var cur = fontSize() || Math.round(parseFloat(getComputedStyle(root).fontSize)) || 17;
    var next = Math.max(FS_MIN, Math.min(FS_MAX, cur + delta));
    store("ml-font", String(next)); applyFont(next);
  }

  /* ---------- where are we? ---------- */
  var chapterNum = document.body.getAttribute("data-chapter"); // e.g. "05"
  var isChapter = !!chapterNum;
  var prefix = isChapter ? "" : "chapters/";
  var homeHref = isChapter ? "../index.html" : "index.html";
  var idx = -1;
  ALL.forEach(function (c, i) { if (c.num === chapterNum) idx = i; });

  /* ---------- sidebar visibility ---------- */
  var mqMobile = window.matchMedia ? window.matchMedia("(max-width: 960px)") : { matches: false };
  function sidebarHidden() { return store("ml-sidebar") === "hidden"; }
  function applySidebar() {
    document.body.classList.toggle("sidebar-hidden", sidebarHidden());
    var b = document.querySelector(".menu-toggle");
    if (b) b.setAttribute("aria-expanded", mqMobile.matches ? String(document.body.classList.contains("nav-open")) : String(!sidebarHidden()));
  }
  function toggleSidebar() {
    if (mqMobile.matches) { document.body.classList.toggle("nav-open"); }
    else { store("ml-sidebar", sidebarHidden() ? "shown" : "hidden"); }
    applySidebar();
  }

  /* ---------- top bar ---------- */
  function buildTopbar() {
    var bar = document.querySelector(".topbar");
    if (!bar) return;
    bar.innerHTML =
      '<button class="menu-toggle" aria-label="Show or hide the chapter menu" title="Show / hide menu  [ [ ]">☰ Menu</button>' +
      '<a class="brand" href="' + homeHref + '"><span class="logo">ML</span> <span>' + BOOK.title + '</span></a>' +
      '<span class="spacer"></span>' +
      (isChapter ? '<span class="chapter-pos">Chapter ' + chapterNum + ' of ' + ALL.length + '</span>' : '') +
      '<span class="font-ctl" role="group" aria-label="Text size">' +
        '<button class="fs-dn" title="Smaller text" aria-label="Smaller text">A−</button>' +
        '<button class="fs-up" title="Larger text" aria-label="Larger text">A+</button>' +
      '</span>' +
      '<button class="theme-toggle" aria-label="Switch theme" title="Switch theme  [ t ]"></button>';
    bar.querySelector(".theme-toggle").addEventListener("click", toggleTheme);
    bar.querySelector(".menu-toggle").addEventListener("click", toggleSidebar);
    bar.querySelector(".fs-dn").addEventListener("click", function () { stepFont(-1); });
    bar.querySelector(".fs-up").addEventListener("click", function () { stepFont(1); });
    applyTheme(store("ml-theme")); applyFont(fontSize()); applySidebar();
    document.addEventListener("click", function (e) {
      if (document.body.classList.contains("nav-open") && !e.target.closest(".sidebar") && !e.target.closest(".menu-toggle")) { document.body.classList.remove("nav-open"); applySidebar(); }
    });
    if (mqMobile.addEventListener) mqMobile.addEventListener("change", applySidebar);
  }

  /* ---------- sidebar ---------- */
  function isDone(num) { return store("ml-done-" + num) === "1"; }
  function buildSidebar() {
    var sb = document.querySelector(".sidebar");
    if (!sb) return;
    var html = '<a class="home" href="' + homeHref + '">🏠 Home & Curriculum Map</a>';
    BOOK.parts.forEach(function (p) {
      html += '<div class="part">' + p.title + '</div>';
      p.chapters.forEach(function (c) {
        html += '<a class="ch' + (c.num === chapterNum ? ' active' : '') + '" href="' + prefix + c.file + '">' +
          '<span class="num">' + c.num + '</span><span>' + c.title + '</span>' +
          (isDone(c.num) ? '<span class="done" title="Completed">✓</span>' : '') + '</a>';
      });
    });
    html += '<div class="kbd-hint"><span class="t">Keyboard</span>' +
      '<kbd>[</kbd> menu &nbsp; <kbd>t</kbd> theme<br>' +
      '<kbd>←</kbd> <kbd>→</kbd> previous / next chapter<br>' +
      '<kbd>Esc</kbd> close fullscreen figure</div>';
    sb.innerHTML = html;
    var active = sb.querySelector(".active");
    if (active && active.scrollIntoView) { try { active.scrollIntoView({ block: "center" }); } catch (e) {} }
  }

  /* ---------- pager & done ---------- */
  function buildPager() {
    var pg = document.querySelector(".pager");
    if (!pg || !isChapter) return;
    var prev = ALL[idx - 1], next = ALL[idx + 1];
    pg.innerHTML =
      (prev ? '<a class="prev" href="' + prev.file + '"><div class="dir">← Previous</div><div class="ttl">' + prev.num + ' · ' + prev.title + '</div></a>' : '<span></span>') +
      (next ? '<a class="next" href="' + next.file + '"><div class="dir">Next →</div><div class="ttl">' + next.num + ' · ' + next.title + '</div></a>' : '<a class="next" href="../index.html"><div class="dir">Finished!</div><div class="ttl">Back to the curriculum map</div></a>');
    var md = document.querySelector(".mark-done");
    if (md) {
      var cb = document.createElement("input"); cb.type = "checkbox"; cb.id = "done-cb"; cb.checked = isDone(chapterNum);
      var lab = document.createElement("label"); lab.htmlFor = "done-cb"; lab.textContent = "Mark this chapter as completed";
      md.appendChild(cb); md.appendChild(lab);
      cb.addEventListener("change", function () { store("ml-done-" + chapterNum, cb.checked ? "1" : "0"); buildSidebar(); });
    }
  }
  function goChapter(delta) {
    if (!isChapter) return;
    var t = ALL[idx + delta];
    if (t) window.location.href = t.file;
  }

  /* ---------- reading progress + back to top ---------- */
  function buildProgress() {
    var track = document.querySelector(".progress-track > div");
    var top = document.createElement("button");
    top.className = "to-top"; top.textContent = "↑ Top"; top.setAttribute("aria-label", "Back to top"); top.title = "Back to top";
    top.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: "smooth" }); });
    document.body.appendChild(top);
    function upd() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      if (track) track.style.width = (max > 0 ? Math.min(100, 100 * h.scrollTop / max) : 0) + "%";
      top.classList.toggle("show", h.scrollTop > 800);
    }
    window.addEventListener("scroll", upd, { passive: true }); upd();
  }

  /* ---------- quizzes ---------- */
  function buildQuizzes() {
    document.querySelectorAll(".quiz .q").forEach(function (q) {
      var buttons = q.querySelectorAll(".opts button");
      buttons.forEach(function (b) {
        b.addEventListener("click", function () {
          if (q.classList.contains("answered")) return;
          q.classList.add("answered");
          buttons.forEach(function (x) { if (x.hasAttribute("data-correct")) x.classList.add("correct"); });
          if (!b.hasAttribute("data-correct")) b.classList.add("wrong");
        });
      });
    });
  }

  /* ---------- copy buttons on code ---------- */
  function buildCopy() {
    document.querySelectorAll("pre.code").forEach(function (pre) {
      var btn = document.createElement("button"); btn.className = "copy"; btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        var txt = pre.innerText.replace(/^Copy\n?/, "");
        try { navigator.clipboard.writeText(txt); btn.textContent = "Copied!"; setTimeout(function () { btn.textContent = "Copy"; }, 1500); } catch (e) {}
      });
      pre.insertBefore(btn, pre.firstChild);
    });
  }

  /* ---------- home page progress summary ---------- */
  function buildHomeProgress() {
    var el = document.getElementById("home-progress");
    if (!el) return;
    var done = ALL.filter(function (c) { return isDone(c.num); }).length;
    el.innerHTML = '<div class="muted" style="font-size:.9rem">Your progress: <strong>' + done + ' / ' + ALL.length + '</strong> chapters completed</div>' +
      '<div style="height:8px;background:var(--surface-3);border-radius:999px;overflow:hidden;margin-top:6px"><div style="height:100%;width:' + (100 * done / ALL.length) + '%;background:var(--accent)"></div></div>';
  }

  /* ---------- range inputs -> outputs ---------- */
  function wireRanges() {
    document.querySelectorAll(".widget input[type=range][data-out]").forEach(function (r) {
      var out = document.getElementById(r.getAttribute("data-out"));
      if (!out) return;
      var f = function () { out.textContent = r.value; };
      r.addEventListener("input", f); f();
    });
  }

  /* ---------- deep-dive index (after the inline TOC) ---------- */
  function buildDeepdiveIndex() {
    var dds = document.querySelectorAll("details.deepdive");
    var toc = document.querySelector(".toc-inline");
    if (!dds.length || !toc) return;
    var box = document.createElement("div"); box.className = "deepdive-index";
    var html = '<strong>🔬 Deep dives in this chapter</strong><ol>';
    dds.forEach(function (d, i) {
      if (!d.id) d.id = "dd-" + (i + 1);
      var s = d.querySelector("summary");
      var label = "", time = "";
      if (s) {
        var clone = s.cloneNode(true);
        var t = clone.querySelector(".time");
        if (t) { time = t.textContent.trim(); t.remove(); }
        label = clone.textContent.replace(/^\s*Deep dive:\s*/i, "").replace(/\s+/g, " ").trim();
      }
      html += '<li><a href="#' + d.id + '">' + (label || d.id) + '</a>' + (time ? '<span class="time">' + time + '</span>' : '') + '</li>';
    });
    box.innerHTML = html + '</ol>';
    toc.parentNode.insertBefore(box, toc.nextSibling);
    // open the deep dive when its anchor is followed
    function openTarget() {
      var h = location.hash && location.hash.slice(1);
      if (!h) return;
      var el = document.getElementById(h);
      if (el && el.tagName === "DETAILS" && el.classList.contains("deepdive")) el.open = true;
    }
    window.addEventListener("hashchange", openTarget); openTarget();
  }

  /* ---------- lightbox: MOVE the element into an overlay and back (keeps widget handlers) ---------- */
  var lb = null;
  function fitSvg(el) {
    var body = lb && lb.querySelector(".lb-body");
    if (!body) return;
    var svg = el.querySelector(":scope > svg") || el.querySelector("svg");
    if (!svg) return;
    var vb = (svg.getAttribute("viewBox") || "").split(/[\s,]+/).map(Number);
    if (vb.length !== 4 || !vb[2] || !vb[3]) return;
    var ratio = vb[3] / vb[2];
    var availW = Math.min(window.innerWidth - 48, 1500) - 34;
    for (var pass = 0; pass < 2; pass++) { // second pass: caption/controls may re-wrap once the width is known
      var chrome = el.getBoundingClientRect().height - svg.getBoundingClientRect().height; // caption/controls/padding
      var availH = window.innerHeight - 48 - chrome - 8;
      var w = Math.max(240, Math.min(availW, availH / ratio));
      svg.style.width = w + "px";
      el.style.width = Math.min(1500, Math.max(w + 34, Math.min(window.innerWidth - 48, 900))) + "px";
    }
  }
  function closeLightbox() {
    if (!lb) return;
    var el = lb.querySelector(".lb-body").firstElementChild;
    var ph = lb._placeholder;
    if (el) {
      el.style.width = "";
      var svg = el.querySelector("svg"); if (svg) svg.style.width = "";
      if (ph && ph.parentNode) ph.parentNode.replaceChild(el, ph);
    }
    lb.remove(); lb = null;
    document.body.classList.remove("lb-open");
    window.removeEventListener("resize", onResize);
    if (lb_lastBtn) { try { lb_lastBtn.focus(); } catch (e) {} }
  }
  var lb_lastBtn = null;
  function onResize() { var el = lb && lb.querySelector(".lb-body").firstElementChild; if (el) fitSvg(el); }
  function openLightbox(el, btn) {
    if (lb) closeLightbox();
    lb_lastBtn = btn || null;
    lb = document.createElement("div"); lb.className = "lightbox"; lb.setAttribute("role", "dialog"); lb.setAttribute("aria-modal", "true");
    lb.innerHTML = '<button class="lb-close" aria-label="Close" title="Close (Esc)">✕</button><div class="lb-body"></div><div class="lb-hint">Esc or click outside to close</div>';
    var ph = document.createComment("lightbox-placeholder");
    el.parentNode.replaceChild(ph, el);
    lb._placeholder = ph;
    lb.querySelector(".lb-body").appendChild(el);
    lb.addEventListener("click", function (e) { if (e.target === lb || e.target.classList.contains("lb-body") || e.target.classList.contains("lb-close")) closeLightbox(); });
    document.body.appendChild(lb);
    document.body.classList.add("lb-open");
    fitSvg(el);
    window.addEventListener("resize", onResize);
    try { lb.querySelector(".lb-close").focus(); } catch (e) {}
  }
  function buildFullscreen() {
    document.querySelectorAll("figure.fig, .widget").forEach(function (el) {
      if (el.querySelector(":scope > .fs-btn")) return;
      var b = document.createElement("button"); b.className = "fs-btn"; b.type = "button"; b.textContent = "⛶";
      b.title = "View fullscreen"; b.setAttribute("aria-label", "View fullscreen");
      b.addEventListener("click", function (e) { e.preventDefault(); e.stopPropagation(); openLightbox(el, b); });
      el.appendChild(b);
    });
  }

  /* ---------- keyboard shortcuts ---------- */
  function isTyping(e) {
    var t = e.target;
    if (!t || t === document.body) return false;
    var tag = (t.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable;
  }
  function buildKeys() {
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { if (lb) { closeLightbox(); e.preventDefault(); } else if (document.body.classList.contains("nav-open")) { document.body.classList.remove("nav-open"); applySidebar(); } return; }
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e)) return;
      if (e.key === "[") { toggleSidebar(); e.preventDefault(); }
      else if (e.key === "t" || e.key === "T") { toggleTheme(); e.preventDefault(); }
      else if (e.key === "ArrowLeft" && !lb) { goChapter(-1); }
      else if (e.key === "ArrowRight" && !lb) { goChapter(1); }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    buildTopbar(); buildSidebar(); buildPager(); buildProgress(); buildQuizzes(); buildCopy(); buildHomeProgress(); wireRanges();
    buildDeepdiveIndex(); buildFullscreen(); buildKeys();
  });
  window.MLBook = { chapters: ALL, toggleTheme: toggleTheme, toggleSidebar: toggleSidebar, setFontSize: function (px) { store("ml-font", String(px)); applyFont(fontSize()); }, openLightbox: openLightbox, closeLightbox: closeLightbox };
})();
