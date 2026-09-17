/* Shared runtime: theme toggle, sidebar, pager, reading progress, quizzes, copy buttons.
   Works from file:// with no build step. */
(function () {
  "use strict";
  var BOOK = window.BOOK || { parts: [] };
  var ALL = [];
  BOOK.parts.forEach(function (p) { p.chapters.forEach(function (c) { c.part = p; ALL.push(c); }); });

  function store(key, val) { try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; } }

  /* ---------- theme ---------- */
  function applyTheme(t) {
    if (t === "dark" || t === "light") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(store("ml-theme"));
  function toggleTheme() {
    var cur = document.documentElement.getAttribute("data-theme");
    var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    var isDark = cur ? cur === "dark" : prefersDark;
    var next = isDark ? "light" : "dark";
    applyTheme(next); store("ml-theme", next);
  }

  /* ---------- where are we? ---------- */
  var chapterNum = document.body.getAttribute("data-chapter"); // e.g. "05"
  var isChapter = !!chapterNum;
  var prefix = isChapter ? "" : "chapters/";
  var homeHref = isChapter ? "../index.html" : "index.html";
  var idx = -1;
  ALL.forEach(function (c, i) { if (c.num === chapterNum) idx = i; });

  /* ---------- top bar ---------- */
  function buildTopbar() {
    var bar = document.querySelector(".topbar");
    if (!bar) return;
    bar.innerHTML =
      '<button class="nav-toggle" aria-label="Toggle chapters">☰ Chapters</button>' +
      '<a class="brand" href="' + homeHref + '"><span class="logo">ML</span> ' + BOOK.title + '</a>' +
      '<span class="spacer"></span>' +
      (isChapter ? '<span class="muted" style="font-size:.85rem">Chapter ' + chapterNum + ' of ' + ALL.length + '</span>' : '') +
      '<button class="theme-toggle" aria-label="Toggle dark mode">◐ Theme</button>';
    bar.querySelector(".theme-toggle").addEventListener("click", toggleTheme);
    bar.querySelector(".nav-toggle").addEventListener("click", function () { document.body.classList.toggle("nav-open"); });
    document.addEventListener("click", function (e) {
      if (document.body.classList.contains("nav-open") && !e.target.closest(".sidebar") && !e.target.closest(".nav-toggle")) document.body.classList.remove("nav-open");
    });
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
      var lb = document.createElement("label"); lb.htmlFor = "done-cb"; lb.textContent = "Mark this chapter as completed";
      md.appendChild(cb); md.appendChild(lb);
      cb.addEventListener("change", function () { store("ml-done-" + chapterNum, cb.checked ? "1" : "0"); buildSidebar(); });
    }
  }

  /* ---------- reading progress ---------- */
  function buildProgress() {
    var track = document.querySelector(".progress-track > div");
    if (!track) return;
    function upd() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      track.style.width = (max > 0 ? Math.min(100, 100 * h.scrollTop / max) : 0) + "%";
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

  document.addEventListener("DOMContentLoaded", function () {
    buildTopbar(); buildSidebar(); buildPager(); buildProgress(); buildQuizzes(); buildCopy(); buildHomeProgress(); wireRanges();
  });
  window.MLBook = { chapters: ALL, toggleTheme: toggleTheme };
})();
