(function () {
  var L = window.DaybookLogic;
  var KEY = "hk-daybook-v1";
  var tab = "today";
  var selected = L.toISODate(new Date());
  var cursor = new Date();
  var editingId = null;
  var notified = {};

  function emptyState() {
    return {
      checks: {},
      items: [],
      notes: {},
      ack: {},
      snooze: {},
      labs: { completedOn: "", appointmentOn: "", remindDaysBefore: 7, lastNudge: "" },
      profile: { feet: 5, inches: 8, weightKg: 90, age: 46 }
    };
  }

  function load() {
    try {
      var raw = JSON.parse(localStorage.getItem(KEY) || "null");
      if (!raw || typeof raw !== "object") return emptyState();
      var base = emptyState();
      base.checks = raw.checks || {};
      base.items = Array.isArray(raw.items) ? raw.items : [];
      base.notes = raw.notes || {};
      base.ack = raw.ack || {};
      base.snooze = raw.snooze || {};
      base.labs = Object.assign(base.labs, raw.labs || {});
      base.profile = Object.assign(base.profile, raw.profile || {});
      return base;
    } catch (e) {
      return emptyState();
    }
  }

  var state = load();

  function save() {
    localStorage.setItem(KEY, JSON.stringify(state));
  }

  function todayISO() {
    return L.toISODate(new Date());
  }

  function uid() {
    return "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function longDate(iso) {
    return L.parseISODate(iso).toLocaleDateString("en-IN", {
      weekday: "long",
      day: "numeric",
      month: "long"
    });
  }

  function monthLabel(date) {
    return date.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
  }

  function weekDates(iso) {
    var d = L.parseISODate(iso);
    var start = new Date(d);
    start.setDate(d.getDate() - d.getDay());
    var out = [];
    for (var i = 0; i < 7; i++) {
      var day = new Date(start);
      day.setDate(start.getDate() + i);
      out.push(L.toISODate(day));
    }
    return out;
  }

  function itemsOn(iso) {
    return state.items
      .filter(function (item) { return L.occursOn(item, iso); })
      .sort(function (a, b) { return a.time.localeCompare(b.time); });
  }

  function dueToday() {
    var iso = todayISO();
    var now = new Date();
    return itemsOn(iso).filter(function (item) {
      return L.isDue(item, iso, now, state.ack, state.snooze);
    });
  }

  function notify(title, body, tag) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    var options = {
      body: body,
      icon: "./assets/icon-192.png",
      tag: tag || title,
      renotify: true,
      data: { url: "./index.html" }
    };
    if (navigator.serviceWorker) {
      navigator.serviceWorker.ready.then(function (reg) {
        return reg.showNotification(title, options);
      }).catch(function () {
        try { new Notification(title, options); } catch (e) { /* ignore */ }
      });
    }
  }

  function maybeNotify() {
    var today = todayISO();
    dueToday().forEach(function (item) {
      var key = L.occurrenceKey(item.id, today);
      var gap = item.nag === "every15x3" ? 15 : item.nag === "every30x3" ? 30 : 180;
      var last = notified[key] || 0;
      if (Date.now() - last < gap * 60000) return;
      notified[key] = Date.now();
      notify(item.title, item.time + " · " + (item.notes || "Personal commitment"), key);
    });
    var card = L.labCardState(state.labs, today);
    var shouldNudge = card.nudge || (card.kind === "unbooked" && (!state.labs.lastNudge || L.diffDays(state.labs.lastNudge, today) >= 30));
    if (shouldNudge && state.labs.lastNudge !== today) {
      state.labs.lastNudge = today;
      save();
      notify(
        "Lal PathLabs",
        card.kind === "unbooked"
          ? "The yearly health panel is still unbooked."
          : "Health tests are due around " + card.date + ".",
        "labs"
      );
    }
  }

  function setTab(next) {
    tab = next;
    document.querySelectorAll(".panel").forEach(function (panel) {
      panel.classList.toggle("active", panel.id === next);
    });
    document.querySelectorAll("nav button").forEach(function (button) {
      button.classList.toggle("active", button.getAttribute("data-tab") === next);
    });
    render();
  }

  function openForm(item) {
    editingId = item ? item.id : null;
    document.getElementById("form-kicker").textContent = item ? "Edit commitment" : "New commitment";
    document.getElementById("f-title").value = item ? item.title : "";
    document.getElementById("f-date").value = item ? item.date : selected;
    document.getElementById("f-time").value = item ? item.time : "16:00";
    document.getElementById("f-notes").value = item ? item.notes || "" : "";
    document.getElementById("f-repeat").value = item ? item.repeat : "once";
    document.getElementById("f-remind").value = item ? item.remindMode : "before15";
    document.getElementById("f-nag").value = item ? item.nag : "once";
    document.getElementById("sheet").showModal();
  }

  function downloadIcs(item) {
    var blob = new Blob([L.buildIcs(item)], { type: "text/calendar" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "commitment.ics";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(link.href); }, 1500);
  }

  function dietBox(iso) {
    var checks = state.checks[iso] || {};
    return L.menuFor(iso, state.profile).map(function (item) {
      var on = !!checks[item.id];
      return (
        '<label class="check' + (on ? " done" : "") + '">' +
        '<input type="checkbox" data-check="' + item.id + '" data-date="' + iso + '"' + (on ? " checked" : "") + " />" +
        '<span><span class="time">' + item.time + '</span><div class="title">' + escapeHtml(item.title) +
        '</div><div class="detail">' + escapeHtml(item.detail) + "</div></span></label>"
      );
    }).join("");
  }

  function commitRow(item, iso) {
    var key = L.occurrenceKey(item.id, iso);
    var done = !!state.ack[key];
    return (
      '<div class="commit' + (done ? " done" : "") + '">' +
      '<input type="checkbox" data-ack="' + item.id + '" data-date="' + iso + '"' + (done ? " checked" : "") + " />" +
      '<div style="flex:1"><span class="time">' + escapeHtml(item.time) + "</span>" +
      '<div class="title">' + escapeHtml(item.title) + "</div>" +
      (item.notes ? '<div class="detail">' + escapeHtml(item.notes) + "</div>" : "") +
      '<div class="row" style="margin-top:8px">' +
      '<button class="quiet" type="button" data-edit="' + item.id + '">Edit</button>' +
      '<button class="quiet" type="button" data-ics="' + item.id + '">iPhone alarm</button>' +
      '<button class="warn" type="button" data-delete="' + item.id + '">Delete</button>' +
      "</div></div></div>"
    );
  }

  function renderToday() {
    var iso = todayISO();
    var due = dueToday();
    var banner = due.length
      ? '<div class="banner"><h2>Due now</h2>' + due.map(function (item) {
          return '<p><strong>' + escapeHtml(item.time) + "</strong> " + escapeHtml(item.title) + "</p>";
        }).join("") +
        '<div class="row"><button class="solid" type="button" id="ack-due">Mark these done</button>' +
        '<button class="quiet" type="button" id="snooze-due">Snooze 15 min</button></div></div>'
      : "";
    var week = weekDates(iso).map(function (day) {
      var d = L.parseISODate(day);
      var on = day === iso ? " on" : "";
      var has = itemsOn(day).length ? " has" : "";
      return '<button type="button" class="' + on + has + '" data-jump="' + day + '"><span>' +
        d.toLocaleDateString("en-IN", { weekday: "narrow" }) + "</span><strong>" + d.getDate() + "</strong></button>";
    }).join("");
    var commits = itemsOn(iso);
    var installed = window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone;
    var install = installed
      ? ""
      : '<div class="card"><h2>Install Daybook</h2><p class="muted">Android adds it to the app list. iPhone adds it from Safari’s share sheet. Either way it opens from its own icon.</p><button class="solid full" type="button" id="install-card">Install</button></div>';
    var lab = L.labCardState(state.labs, iso);
    var labHtml = "";
    if (lab.kind === "unbooked" || lab.nudge) {
      labHtml = '<div class="card"><p class="kicker">Lal PathLabs</p><h2>' +
        (lab.kind === "unbooked" ? "Yearly panel not booked" : "Tests due " + escapeHtml(lab.date)) +
        '</h2><p class="muted">Open Labs for the exact test names.</p>' +
        '<button class="quiet" type="button" data-tab-jump="labs">Open lab list</button></div>';
    }
    document.getElementById("today").innerHTML =
      install +
      banner +
      '<div class="week">' + week + "</div>" +
      '<div class="card"><p class="kicker">' + escapeHtml(longDate(iso)) + "</p><h2>Commitments</h2>" +
      '<p class="muted">More than one commitment can sit on this day. They stay in time order.</p>' +
      (commits.length ? commits.map(function (item) { return commitRow(item, iso); }).join("") : '<p class="muted">Nothing fixed today.</p>') +
      '<button class="solid full" type="button" id="add-today" style="margin-top:8px">' +
      (commits.length ? "Add another today" : "Add a commitment") + "</button></div>" +
      '<div class="card"><h2>Vegetarian day</h2><p class="muted">College hours 10 am to 7 pm. Tick each plate as it happens.</p>' +
      dietBox(iso) + "</div>" + labHtml;
  }

  function renderDiary() {
    var year = cursor.getFullYear();
    var month = cursor.getMonth();
    var first = new Date(year, month, 1);
    var padCount = first.getDay();
    var days = new Date(year, month + 1, 0).getDate();
    var cells = '<div class="dow">S</div><div class="dow">M</div><div class="dow">T</div><div class="dow">W</div><div class="dow">T</div><div class="dow">F</div><div class="dow">S</div>';
    for (var i = 0; i < padCount; i++) cells += "<span></span>";
    for (var day = 1; day <= days; day++) {
      var iso = L.toISODate(new Date(year, month, day));
      var on = iso === selected ? " on" : "";
      var has = itemsOn(iso).length ? " has" : "";
      cells += '<button type="button" class="' + on + has + '" data-select="' + iso + '"><strong>' + day + "</strong></button>";
    }
    var list = itemsOn(selected);
    document.getElementById("diary").innerHTML =
      '<div class="card"><div class="month-head"><button class="quiet" type="button" id="prev-month">Prev</button><h2>' +
      escapeHtml(monthLabel(cursor)) + '</h2><button class="quiet" type="button" id="next-month">Next</button></div><div class="cal">' +
      cells + "</div></div>" +
      '<div class="card"><p class="kicker">' + escapeHtml(longDate(selected)) + '</p><h2>That day</h2>' +
      '<p class="muted">Add as many as the day needs. Each keeps its own time and reminder.</p>' +
      (list.length ? list.map(function (item) { return commitRow(item, selected); }).join("") : '<p class="muted">No commitment on this date.</p>') +
      '<button class="solid full" type="button" id="add-day" style="margin-top:8px">' +
      (list.length ? "Add another on this date" : "Add on this date") + "</button></div>" +
      '<div class="card"><h2>Notepad</h2><p class="muted">A private note for this date. It stays on this phone.</p>' +
      '<textarea id="day-note" data-note="' + selected + '">' + escapeHtml(state.notes[selected] || "") + "</textarea></div>";
  }

  function weekMenusHtml() {
    var todayName = L.parseISODate(todayISO()).toLocaleDateString("en-IN", { weekday: "long" });
    return L.weekMenus(state.profile).map(function (day) {
      var today = day.name === todayName;
      return (
        '<div class="card' + (today ? " today-menu" : "") + '">' +
        (today ? '<p class="kicker">Today</p>' : "") +
        "<h2>" + escapeHtml(day.name) + "</h2>" +
        day.meals.map(function (meal) {
          return '<div class="meal"><span class="time">' + meal.time + "</span>" +
            '<div class="title">' + escapeHtml(meal.title) + "</div>" +
            '<div class="detail">' + escapeHtml(meal.detail) + "</div></div>";
        }).join("") +
        "</div>"
      );
    }).join("");
  }

  function refreshPortions() {
    var saved = state.profile;
    function pick(id, key, lo, hi) {
      var el = document.getElementById(id);
      if (!el || String(el.value).trim() === "") return saved[key];
      var n = Number(el.value);
      if (!Number.isFinite(n) || n < lo || n > hi) return saved[key];
      return n;
    }
    var m = L.measures({
      feet: pick("p-feet", "feet", 4, 7),
      inches: pick("p-inches", "inches", 0, 11),
      weightKg: pick("p-weight", "weightKg", 40, 200),
      age: pick("p-age", "age", 18, 90)
    });
    state.profile = { feet: m.feet, inches: m.inches, weightKg: m.weightKg, age: m.age };
    save();
    var bmiEl = document.getElementById("bmi-value");
    var heightEl = document.getElementById("height-label");
    if (bmiEl) bmiEl.textContent = m.bmi.toFixed(1);
    if (heightEl) heightEl.textContent = m.heightLabel;
    var menus = document.getElementById("week-menus");
    if (menus) menus.innerHTML = weekMenusHtml();
    var ticks = document.getElementById("today-ticks");
    if (ticks) ticks.innerHTML = dietBox(todayISO());
  }

  function renderDiet() {
    var m = L.measures(state.profile);
    document.getElementById("diet").innerHTML =
      '<div class="card"><p class="kicker">Fully vegetarian</p><h2>A college day, 10 to 7</h2>' +
      '<div class="measure-grid">' +
      '<label>Feet<input id="p-feet" type="number" inputmode="numeric" min="4" max="7" step="1" value="' + m.feet + '" /></label>' +
      '<label>Inches<input id="p-inches" type="number" inputmode="numeric" min="0" max="11" step="1" value="' + m.inches + '" /></label>' +
      '<label>Weight, kg<input id="p-weight" type="number" inputmode="decimal" min="40" max="200" step="0.1" value="' + m.weightKg + '" /></label>' +
      '<label>Age<input id="p-age" type="number" inputmode="numeric" min="18" max="90" step="1" value="' + m.age + '" /></label>' +
      "</div>" +
      '<div class="stats" style="margin-top:10px"><div class="stat"><span class="tiny">Height</span><b id="height-label">' +
      escapeHtml(m.heightLabel) + '</b></div><div class="stat"><span class="tiny">BMI</span><b id="bmi-value">' +
      m.bmi.toFixed(1) + "</b></div></div>" +
      '<p class="muted" style="margin-top:10px">Write height, weight, and age. The seven menus change their gram amounts, and BMI updates with them.</p></div>' +
      '<div id="week-menus">' + weekMenusHtml() + "</div>" +
      '<div class="card"><h2>Today’s ticks</h2><div id="today-ticks">' + dietBox(todayISO()) + "</div></div>";
  }

  function renderLabs() {
    var card = L.labCardState(state.labs, todayISO());
    var status = "Book the first panel this month. After that, repeat it every 12 months.";
    if (card.kind === "appointment") status = "Appointment on " + card.date + ".";
    if (card.kind === "scheduled") status = "Next panel " + card.date + ".";
    if (card.kind === "overdue") status = "The 12-month panel was due " + card.date + ".";
    document.getElementById("labs").innerHTML =
      '<div class="card"><p class="kicker">Dr Lal PathLabs</p><h2>Yearly screen</h2><p>' + escapeHtml(status) +
      "</p>" +
      L.LABS.map(function (lab) {
        return '<div class="lab"><div class="title">' + escapeHtml(lab.name) + "</div>" +
          (lab.code ? '<div class="code">' + escapeHtml(lab.code) + "</div>" : "") +
          '<div class="detail">' + escapeHtml(lab.detail) + "</div></div>";
      }).join("") +
      "</div>" +
      '<div class="card"><h2>Dates</h2>' +
      '<label class="field" for="lab-appt">Appointment date</label>' +
      '<input id="lab-appt" type="date" value="' + escapeHtml(state.labs.appointmentOn || "") + '" />' +
      '<label class="field" for="lab-done">Completed on</label>' +
      '<input id="lab-done" type="date" value="' + escapeHtml(state.labs.completedOn || "") + '" />' +
      '<label class="field" for="lab-remind">Remind me this many days before</label>' +
      '<select id="lab-remind">' +
      [1, 3, 7, 14, 30].map(function (n) {
        return '<option value="' + n + '"' + (Number(state.labs.remindDaysBefore) === n ? " selected" : "") + ">" + n + " days</option>";
      }).join("") +
      "</select>" +
      '<button class="solid" type="button" id="lab-mark" style="margin-top:12px">Mark completed today</button></div>';
  }

  function render() {
    if (tab === "today") renderToday();
    if (tab === "diary") renderDiary();
    if (tab === "diet") renderDiet();
    if (tab === "labs") renderLabs();
    var note = document.getElementById("day-note");
    if (note) {
      note.addEventListener("input", function () {
        state.notes[note.getAttribute("data-note")] = note.value;
        save();
      });
    }
  }

  document.body.addEventListener("click", function (event) {
    var target = event.target.closest("button");
    if (!target) return;
    if (target.getAttribute("data-tab")) setTab(target.getAttribute("data-tab"));
    if (target.getAttribute("data-tab-jump")) setTab(target.getAttribute("data-tab-jump"));
    if (target.getAttribute("data-jump")) {
      selected = target.getAttribute("data-jump");
      cursor = L.parseISODate(selected);
      setTab("diary");
    }
    if (target.getAttribute("data-select")) {
      selected = target.getAttribute("data-select");
      render();
    }
    if (target.id === "add-today" || target.id === "add-day") openForm(null);
    if (target.id === "install-card") openInstall();
    if (target.id === "prev-month") {
      cursor = new Date(cursor.getFullYear(), cursor.getMonth() - 1, 1);
      render();
    }
    if (target.id === "next-month") {
      cursor = new Date(cursor.getFullYear(), cursor.getMonth() + 1, 1);
      render();
    }
    if (target.getAttribute("data-edit")) {
      var found = state.items.find(function (item) { return item.id === target.getAttribute("data-edit"); });
      if (found) openForm(found);
    }
    if (target.getAttribute("data-ics")) {
      var icsItem = state.items.find(function (item) { return item.id === target.getAttribute("data-ics"); });
      if (icsItem) downloadIcs(icsItem);
    }
    if (target.getAttribute("data-delete")) {
      var id = target.getAttribute("data-delete");
      state.items = state.items.filter(function (item) { return item.id !== id; });
      save();
      render();
    }
    if (target.id === "ack-due") {
      dueToday().forEach(function (item) {
        state.ack[L.occurrenceKey(item.id, todayISO())] = true;
      });
      save();
      render();
    }
    if (target.id === "snooze-due") {
      var until = new Date(Date.now() + 15 * 60000).toISOString();
      dueToday().forEach(function (item) {
        state.snooze[L.occurrenceKey(item.id, todayISO())] = until;
        delete notified[L.occurrenceKey(item.id, todayISO())];
      });
      save();
      render();
    }
    if (target.id === "lab-mark") {
      state.labs.completedOn = todayISO();
      state.labs.appointmentOn = "";
      save();
      render();
    }
    if (target.id === "close-sheet") document.getElementById("sheet").close();
  });

  document.body.addEventListener("change", function (event) {
    var box = event.target;
    if (box.getAttribute("data-check")) {
      var date = box.getAttribute("data-date");
      state.checks[date] = state.checks[date] || {};
      state.checks[date][box.getAttribute("data-check")] = box.checked;
      save();
      render();
    }
    if (box.getAttribute("data-ack")) {
      var key = L.occurrenceKey(box.getAttribute("data-ack"), box.getAttribute("data-date"));
      if (box.checked) state.ack[key] = true;
      else delete state.ack[key];
      save();
      render();
    }
    if (box.id === "p-feet" || box.id === "p-inches" || box.id === "p-weight" || box.id === "p-age") {
      refreshPortions();
    }
    if (box.id === "lab-appt" || box.id === "lab-done" || box.id === "lab-remind") {
      state.labs.appointmentOn = document.getElementById("lab-appt").value;
      state.labs.completedOn = document.getElementById("lab-done").value;
      state.labs.remindDaysBefore = Number(document.getElementById("lab-remind").value);
      save();
      render();
    }
  });

  document.body.addEventListener("input", function (event) {
    var id = event.target.id;
    if (id === "p-feet" || id === "p-inches" || id === "p-weight" || id === "p-age") refreshPortions();
  });

  document.body.addEventListener("focusout", function (event) {
    var map = { "p-feet": "feet", "p-inches": "inches", "p-weight": "weightKg", "p-age": "age" };
    var key = map[event.target.id];
    if (!key) return;
    event.target.value = L.measures(state.profile)[key];
  });

  document.getElementById("commit-form").addEventListener("submit", function (event) {
    event.preventDefault();
    var data = {
      id: editingId || uid(),
      title: document.getElementById("f-title").value.trim(),
      date: document.getElementById("f-date").value,
      time: document.getElementById("f-time").value,
      notes: document.getElementById("f-notes").value.trim(),
      repeat: document.getElementById("f-repeat").value,
      remindMode: document.getElementById("f-remind").value,
      nag: document.getElementById("f-nag").value
    };
    if (!data.title || !data.date || !data.time) return;
    var index = state.items.findIndex(function (item) { return item.id === data.id; });
    var isNew = index < 0;
    if (isNew) state.items.push(data);
    else state.items[index] = data;
    selected = data.date;
    cursor = L.parseISODate(data.date);
    save();
    document.getElementById("sheet").close();
    setTab("diary");
    if (isNew) downloadIcs(data);
  });

  var deferredPrompt = null;

  function standaloneMode() {
    return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
  }

  function openInstall() {
    if (standaloneMode()) return;
    document.getElementById("install-ios").hidden = false;
    document.getElementById("install-android").hidden = false;
    document.getElementById("install-now").hidden = !deferredPrompt;
    document.getElementById("install-sheet").showModal();
  }

  window.addEventListener("beforeinstallprompt", function (event) {
    event.preventDefault();
    deferredPrompt = event;
    document.getElementById("install-btn").textContent = "Install";
  });

  window.addEventListener("appinstalled", function () {
    deferredPrompt = null;
    document.getElementById("install-btn").textContent = "Installed";
  });

  document.getElementById("install-btn").addEventListener("click", function () {
    openInstall();
  });

  document.getElementById("install-now").addEventListener("click", function () {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    deferredPrompt.userChoice.then(function () {
      deferredPrompt = null;
      document.getElementById("install-sheet").close();
    });
  });

  document.getElementById("close-install").addEventListener("click", function () {
    document.getElementById("install-sheet").close();
  });

  if (standaloneMode()) document.getElementById("install-btn").textContent = "Installed";

  document.getElementById("notify-btn").addEventListener("click", function () {
    if (!("Notification" in window)) {
      alert("This browser cannot show alerts. Use Add to Home Screen, then the iPhone alarm button on each commitment.");
      return;
    }
    Notification.requestPermission().then(function () {
      document.getElementById("notify-btn").textContent =
        Notification.permission === "granted" ? "Alerts on" : "Alerts off";
    });
  });

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./service-worker.js").catch(function () {});
  }

  render();
  maybeNotify();
  var dueSig = "";
  setInterval(function () {
    maybeNotify();
    if (tab !== "today") return;
    var sig = dueToday().map(function (item) { return item.id; }).join("|");
    if (sig !== dueSig) {
      dueSig = sig;
      renderToday();
    }
  }, 20000);
})();
