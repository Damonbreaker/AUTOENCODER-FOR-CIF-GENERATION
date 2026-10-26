// app.js — Maternal Health PWA (Camera, Notifications & Clinical Care Engine)

(function () {
  // -------------------------------------------------------------------------
  // 1. Safe Local Storage Wrapper (works in standalone PWA, HTTPS, and file://)
  // -------------------------------------------------------------------------
  var memoryStore = {};
  function safeGet(key) {
    try {
      var val = window.localStorage.getItem(key);
      if (val) return JSON.parse(val);
    } catch (e) {}
    try {
      if (window.name && window.name.indexOf("{") === 0) {
        var parsed = JSON.parse(window.name);
        if (parsed[key]) return parsed[key];
      }
    } catch (e) {}
    return memoryStore[key] || null;
  }

  function safeSet(key, val) {
    memoryStore[key] = val;
    try {
      window.localStorage.setItem(key, JSON.stringify(val));
    } catch (e) {}
    try {
      var wn =
        window.name && window.name.indexOf("{") === 0
          ? JSON.parse(window.name)
          : {};
      wn[key] = val;
      window.name = JSON.stringify(wn);
    } catch (e) {}
  }

  // -------------------------------------------------------------------------
  // 1B. Native PWA Install Prompt Handler (beforeinstallprompt)
  // -------------------------------------------------------------------------
  var deferredInstallPrompt = null;
  var installPwaBtn = document.getElementById("installPwaBtn");
  var pwaInstallStatus = document.getElementById("pwaInstallStatus");

  window.addEventListener("beforeinstallprompt", function (e) {
    e.preventDefault();
    deferredInstallPrompt = e;
    if (installPwaBtn) {
      installPwaBtn.style.display = "inline-block";
    }
    if (pwaInstallStatus) {
      pwaInstallStatus.textContent =
        "App is ready to install! Tap 'Install App on Phone' above.";
    }
  });

  if (installPwaBtn) {
    installPwaBtn.addEventListener("click", async function () {
      if (!deferredInstallPrompt) return;
      deferredInstallPrompt.prompt();
      var choiceResult = await deferredInstallPrompt.userChoice;
      if (choiceResult && choiceResult.outcome === "accepted") {
        installPwaBtn.style.display = "none";
        if (pwaInstallStatus) {
          pwaInstallStatus.textContent = "App installed on your phone!";
        }
      }
      deferredInstallPrompt = null;
    });
  }

  window.addEventListener("appinstalled", function () {
    if (installPwaBtn) installPwaBtn.style.display = "none";
    if (pwaInstallStatus) {
      pwaInstallStatus.textContent = "Maternal Health App installed!";
    }
  });

  // -------------------------------------------------------------------------
  // 2. Camera Access via navigator.mediaDevices.getUserMedia
  // -------------------------------------------------------------------------
  var activeStream = null;
  var openCameraBtn = document.getElementById("openCameraBtn");
  var capturePhotoBtn = document.getElementById("capturePhotoBtn");
  var webcamVideo = document.getElementById("webcam");
  var cameraStatus = document.getElementById("cameraStatus");
  var capturedCanvas = document.getElementById("capturedCanvas");
  var capturedGallery = document.getElementById("capturedGallery");

  async function toggleCamera() {
    if (!webcamVideo) return;

    // If stream is already running, stop it
    if (activeStream) {
      activeStream.getTracks().forEach(function (track) {
        track.stop();
      });
      activeStream = null;
      webcamVideo.srcObject = null;
      webcamVideo.style.display = "none";
      if (capturePhotoBtn) capturePhotoBtn.style.display = "none";
      if (openCameraBtn) openCameraBtn.textContent = "Open Camera";
      if (cameraStatus) {
        cameraStatus.textContent = "Camera closed.";
      }
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (cameraStatus) {
        cameraStatus.textContent =
          "Camera API (navigator.mediaDevices.getUserMedia) is not supported in this browser or requires HTTPS / localhost.";
      }
      return;
    }

    try {
      if (cameraStatus) {
        cameraStatus.textContent = "Requesting camera permission...";
      }
      // Prefer rear camera on mobile for scanning medical reports, fallback to any camera
      try {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false
        });
      } catch (innerErr) {
        activeStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
      }

      webcamVideo.srcObject = activeStream;
      webcamVideo.style.display = "block";
      await webcamVideo.play();

      if (openCameraBtn) openCameraBtn.textContent = "Close Camera";
      if (capturePhotoBtn) capturePhotoBtn.style.display = "inline-block";
      if (cameraStatus) {
        cameraStatus.textContent =
          "Camera active — point at a lab report or prescription to capture.";
      }
    } catch (err) {
      if (cameraStatus) {
        cameraStatus.textContent =
          "Camera access denied or unavailable: " + (err && err.message ? err.message : String(err));
      }
    }
  }

  function captureReportSnapshot() {
    if (!webcamVideo || !activeStream || !capturedCanvas) return;
    var width = webcamVideo.videoWidth || 640;
    var height = webcamVideo.videoHeight || 480;
    capturedCanvas.width = width;
    capturedCanvas.height = height;
    var ctx = capturedCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, width, height);
    var dataUrl = capturedCanvas.toDataURL("image/jpeg", 0.82);

    var snaps = safeGet("rajniCapturedSnaps") || [];
    snaps.unshift({
      time: new Date().toLocaleString(),
      dataUrl: dataUrl
    });
    if (snaps.length > 6) snaps = snaps.slice(0, 6);
    safeSet("rajniCapturedSnaps", snaps);
    renderCapturedSnaps();
    if (cameraStatus) {
      cameraStatus.textContent = "Report photo captured and saved to app!";
    }
  }

  function renderCapturedSnaps() {
    if (!capturedGallery) return;
    var snaps = safeGet("rajniCapturedSnaps") || [];
    if (!snaps.length) {
      capturedGallery.innerHTML = "";
      return;
    }
    capturedGallery.innerHTML = snaps
      .map(function (s) {
        return (
          '<div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:8px;">' +
          '<img src="' +
          s.dataUrl +
          '" alt="Captured Report" style="width:100%;border-radius:6px;display:block;margin-bottom:4px;" />' +
          '<div style="font-size:0.72rem;color:#94a3b8;">Captured: ' +
          s.time +
          "</div></div>"
        );
      })
      .join("");
  }

  if (openCameraBtn) {
    openCameraBtn.addEventListener("click", toggleCamera);
  }
  if (capturePhotoBtn) {
    capturePhotoBtn.addEventListener("click", captureReportSnapshot);
  }

  // -------------------------------------------------------------------------
  // 3. Browser Notifications via Notification.requestPermission()
  // -------------------------------------------------------------------------
  var enableNotificationsBtn = document.getElementById("enableNotificationsBtn");
  var testNotificationBtn = document.getElementById("testNotificationBtn");
  var notificationStatus = document.getElementById("notificationStatus");

  function showAppNotification(title, body) {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.ready
        .then(function (registration) {
          if (registration && registration.showNotification) {
            registration.showNotification(title, {
              body: body,
              icon: "assets/icon-192.png",
              badge: "assets/icon-96.png",
              vibrate: [200, 100, 200]
            });
          } else {
            new Notification(title, { body: body, icon: "assets/icon-192.png" });
          }
        })
        .catch(function () {
          try {
            new Notification(title, { body: body, icon: "assets/icon-192.png" });
          } catch (e) {}
        });
    } else if ("Notification" in window) {
      try {
        new Notification(title, { body: body, icon: "assets/icon-192.png" });
      } catch (e) {}
    }
  }

  async function requestNotificationPermission() {
    if (!("Notification" in window)) {
      if (notificationStatus) {
        notificationStatus.textContent =
          "Notifications are not supported by this browser.";
      }
      return;
    }

    try {
      var permission = await Notification.requestPermission();
      if (permission === "granted") {
        if (enableNotificationsBtn) {
          enableNotificationsBtn.textContent = "Notifications Enabled";
        }
        if (testNotificationBtn) {
          testNotificationBtn.style.display = "inline-block";
        }
        if (notificationStatus) {
          notificationStatus.textContent =
            "Notifications enabled! You will receive medicine, water, and test reminders.";
        }
        showAppNotification(
          "Maternal Health App — Reminders Active",
          "Daily reminders enabled for Ms. Rajni Singh (Rapeed D 7:00 AM, Glynamic MP2 + Ondero 5 8:45 AM, Lemon Water 11:15 AM & 3:15 PM, Obimet GX2 7:45 PM, Prohance HP 9:30 PM)."
        );
      } else {
        if (notificationStatus) {
          notificationStatus.textContent =
            "Notification permission status: " + permission;
        }
      }
    } catch (err) {
      if (notificationStatus) {
        notificationStatus.textContent =
          "Could not request notification permission: " +
          (err && err.message ? err.message : String(err));
      }
    }
  }

  if (enableNotificationsBtn) {
    enableNotificationsBtn.addEventListener("click", requestNotificationPermission);
  }
  if (testNotificationBtn) {
    testNotificationBtn.addEventListener("click", function () {
      showAppNotification(
        "Maa Care Reminder",
        "Time to check today's water glass count (2.5-3.0L goal), lemon water, and medicine schedule!"
      );
    });
  }

  // -------------------------------------------------------------------------
  // 4. Clinical Care Data & State Engine for Ms. Rajni Singh
  // -------------------------------------------------------------------------
  var WEEKLY_DIET = [
    {
      bf: "2 Oats Idli OR Veggie Idli + Sambar/Chutney + 1 glass Chaas/Dahi",
      lunch:
        "2 katori Karela Sabzi + 2 Bran/Multigrain Rotis + 1 katori Dal + Chaas/Dahi + Salad (deseed tomatoes)",
      dinner:
        "2 Moong Dal Chillas + 1 katori Raita (Avoid Sabudana) + 1.5 scoops Prohance HP at 9:30 PM"
    },
    {
      bf: "1 small bowl Vegetable Poha + Unsweetened Tea + Nuts + 1 scoop Prohance HP",
      lunch:
        "2 katori Methi/Lauki Paneer (OR blanched & drained Palak Paneer) + 2 Bran Rotis + 1 katori Dal + Salad",
      dinner:
        "2 Large Idlis + Sambar OR Veg Upma + Chaas + 1 scoop Prohance HP at 9:30 PM"
    },
    {
      bf: "1-2 Moong Dal or Besan Chilla stuffed with 50g Low-Fat Paneer & Veggies + Chutney",
      lunch: "2 katori Lauki Sabzi + 2 Rotis + 1 katori Dal + Salad",
      dinner:
        "1 bowl Veg Tahri / Brown Rice Pulao (double veggies & soya) + Clear Veg Soup + 1.5 scoops Prohance HP at 9:30 PM"
    },
    {
      bf: "1 small bowl Veg Upma + Handful Nuts & Seeds + 1 scoop Prohance HP",
      lunch:
        "1 katori Mix Veg + 1 katori Methi/Lauki Dal + 2 Multigrain Rotis + Salad",
      dinner:
        "1 bowl Roasted Wheat/Ragi Vermicelli + Raita + 50g Roasted Paneer + Broccoli + 1 scoop Prohance HP at 9:30 PM"
    },
    {
      bf: "1 bowl Steamed Sprouts (Green Moong, Soybean, Kala Chana) + 1 scoop Prohance HP",
      lunch:
        "2 katori Rajma / Soybean / Kala Chana Sabzi + 1 Multigrain Roti + 1 small katori Red/Brown Rice + Salad",
      dinner:
        "1 katori Thick Dal + 1 Multigrain Roti + Salad + 1.5 scoops Prohance HP at 9:30 PM"
    },
    {
      bf: "1 bowl Veg Daliya Khichdi OR Skimmed Milk + Daliya/Oats + 1 scoop Prohance HP",
      lunch:
        "2 katori Taroee / Munga (Drumstick) Sabzi (Avoid excess Bhindi) + 2 Rotis + Dahi/Raita + Salad",
      dinner:
        "1 large bowl Skimmed Milk + Oats OR Milk Daliya (Unsweetened + 1.5 scoops Prohance HP)"
    },
    {
      bf: "1 Dal/Oats Dosa + Sambar OR 2 katori Sabzi + 1 Roti + Dahi",
      lunch:
        "2 katori Beans (Rajma/Sem/Lobia/Gwar Phali) Sabzi + 2 Rotis + Salad + Chaas + Weekly Vit D3 60k Syrup after lunch",
      dinner:
        "1 large bowl Moong Dal Veg Khichdi + Dahi + 1 scoop Prohance HP at 9:30 PM"
    }
  ];

  var DEFAULT_REPORTS = [
    {
      date: "2026-10-04",
      label: "Baseline Lal PathLabs",
      fasting: 230.9,
      pp: 459.3,
      hba1c: 12.0,
      tg: 232,
      creat: 0.61,
      egfr: 106,
      vitD: 36.7,
      calc: 9.29,
      yeast: 36.6,
      crystals: 1.5
    },
    {
      date: "2026-10-25",
      label: "3-Wk Check + Itra 200",
      fasting: 225.0,
      pp: 254.0,
      hba1c: 12.0,
      tg: 232,
      creat: 0.61,
      egfr: 106,
      vitD: 48.0,
      calc: 9.29,
      yeast: 5.0,
      crystals: 1.0
    }
  ];

  var state = safeGet("rajniCareStateV2") || {
    dailyDone: {},
    water: 6,
    lemon: 1,
    vitD: 3,
    dayIdx: 0,
    reports: DEFAULT_REPORTS,
    checkupsDone: {},
    groceriesDone: {}
  };

  function render() {
    safeSet("rajniCareStateV2", state);
    var taskBoxes = document.querySelectorAll("input[data-task]");
    var doneCount = 0;
    for (var i = 0; i < taskBoxes.length; i++) {
      var id = taskBoxes[i].getAttribute("data-task");
      taskBoxes[i].checked = Boolean(state.dailyDone[id]);
      if (state.dailyDone[id]) doneCount++;
    }
    var statTasksEl = document.getElementById("statTasks");
    if (statTasksEl) statTasksEl.textContent = doneCount + " / 12";
    var badgeEl = document.getElementById("headerProgressBadge");
    if (badgeEl) badgeEl.textContent = "Today: " + doneCount + "/12 Done";
    var waterEl = document.getElementById("statWater");
    if (waterEl)
      waterEl.textContent =
        state.water + " (" + (state.water * 0.25).toFixed(2) + " L)";
    var lemonEl = document.getElementById("statLemon");
    if (lemonEl) lemonEl.textContent = state.lemon + " / 2";
    var vitDEl = document.getElementById("statVitD");
    if (vitDEl) vitDEl.textContent = state.vitD + " / 8";
    var vitDSubEl = document.getElementById("statVitDSub");
    if (vitDSubEl)
      vitDSubEl.textContent =
        state.vitD >= 8
          ? "8-Wk Course Done!"
          : 8 - state.vitD + " Sunday doses left";

    var d = WEEKLY_DIET[state.dayIdx] || WEEKLY_DIET[0];
    var daySelectEl = document.getElementById("daySelect");
    if (daySelectEl) daySelectEl.value = String(state.dayIdx);
    var menuBoxEl = document.getElementById("todayMenuBox");
    if (menuBoxEl) {
      menuBoxEl.innerHTML =
        '<div><strong style="color:var(--primary);">Breakfast (8:30 AM):</strong> ' +
        d.bf +
        "</div>" +
        '<div><strong style="color:var(--success);">Lunch (1:30 PM):</strong> ' +
        d.lunch +
        "</div>" +
        '<div><strong style="color:var(--warn);">Dinner (7:30 PM) & Bedtime:</strong> ' +
        d.dinner +
        "</div>";
    }

    var chkBoxes = document.querySelectorAll("input[data-chk]");
    for (var j = 0; j < chkBoxes.length; j++) {
      var cid = chkBoxes[j].getAttribute("data-chk");
      chkBoxes[j].checked = Boolean(state.checkupsDone[cid]);
    }
    var grocBoxes = document.querySelectorAll("input[data-groc]");
    var grocDoneCount = 0;
    for (var k = 0; k < grocBoxes.length; k++) {
      var gid = grocBoxes[k].getAttribute("data-groc");
      grocBoxes[k].checked = Boolean(state.groceriesDone[gid]);
      if (state.groceriesDone[gid]) grocDoneCount++;
    }
    var grocTitleEl = document.getElementById("groceryCountTitle");
    if (grocTitleEl) {
      grocTitleEl.textContent =
        "1-Month Complete Grocery Checklist (" +
        grocDoneCount +
        " / " +
        grocBoxes.length +
        " Purchased)";
    }

    var latest =
      state.reports[state.reports.length - 1] || DEFAULT_REPORTS[1];
    var adv = [];
    if (latest.fasting < 70 || latest.pp < 70) {
      adv.push({
        cls: "danger",
        title:
          "URGENT LOW SUGAR ALERT (" +
          Math.min(latest.fasting, latest.pp) +
          " mg/dL < 70)",
        body: "Give 3 tsp sugar/glucose in water immediately and contact doctor to reduce Glimepiride dose."
      });
    } else if (latest.fasting > 180 || latest.pp > 220) {
      adv.push({
        cls: "warn",
        title:
          "Active Sugar Reduction (Fasting: " +
          latest.fasting +
          " | PP: " +
          latest.pp +
          " mg/dL)",
        body: "Continue Glynamic MP2 + Ondero 5 after breakfast, Obimet GX2 after 7:30 PM dinner, and 1.5 scoops Prohance HP at 9:30 PM."
      });
    } else {
      adv.push({
        cls: "success",
        title:
          "Great Glycemic Progress (Fasting: " +
          latest.fasting +
          " | PP: " +
          latest.pp +
          " mg/dL)",
        body: "Sugars are in or near target range! Continue salad-first eating order and 30-35m walk."
      });
    }
    if (latest.yeast > 1.0 || latest.crystals >= 1.0) {
      adv.push({
        cls: "warn",
        title:
          "Urine Yeast (" +
          latest.yeast +
          ") & Stone Crystal (" +
          latest.crystals +
          ") Shield Active",
        body: "Drink 2.5-3.0L water + 2 glasses unsweetened Lemon Water daily. Repeat Urine R/E around Nov 8-10."
      });
    } else {
      adv.push({
        cls: "success",
        title: "Urine Infection & Stone Crystals Cleared!",
        body: "Continue 2.5L water and daily lemon water."
      });
    }
    if (state.vitD >= 8 || latest.vitD >= 75) {
      adv.push({
        cls: "success",
        title: "8-Week Vitamin D3 Course Complete (" + state.vitD + "/8)",
        body: "Stop weekly 60k IU doses due to kidney stone history; switch to once a month."
      });
    } else {
      adv.push({
        cls: "",
        title:
          "Vitamin D3 Weekly Course (" +
          state.vitD +
          "/8 Done — " +
          (8 - state.vitD) +
          " Left)",
        body: "Give 1 Sugar-Free 60k IU bottle every Sunday right AFTER Lunch."
      });
    }

    var adviceContainer = document.getElementById("dynamicAdviceContainer");
    if (adviceContainer) {
      adviceContainer.innerHTML = adv
        .map(function (a) {
          return (
            '<div class="callout ' +
            a.cls +
            '"><div class="callout-title">' +
            a.title +
            '</div><div class="callout-body">' +
            a.body +
            "</div></div>"
          );
        })
        .join("");
    }

    var reportsTable = document.getElementById("reportsTable");
    if (reportsTable) {
      reportsTable.innerHTML =
        "<thead><tr><th>Date</th><th>Fasting/PP</th><th>HbA1c/TG</th><th>Kidney/VitD</th><th>Urine</th></tr></thead><tbody>" +
        state.reports
          .slice()
          .reverse()
          .map(function (r) {
            return (
              "<tr><td><strong>" +
              r.date +
              "</strong><br/>" +
              r.label +
              "</td><td><strong>" +
              r.fasting +
              " / " +
              r.pp +
              "</strong></td><td>" +
              r.hba1c +
              "% | TG:" +
              r.tg +
              "</td><td>Cr:" +
              r.creat +
              " | VitD:" +
              r.vitD +
              "</td><td>Y:" +
              r.yeast +
              " C:" +
              r.crystals +
              "</td></tr>"
            );
          })
          .join("") +
        "</tbody>";
    }
  }

  var btnWaterPlus = document.getElementById("btnWaterPlus");
  if (btnWaterPlus)
    btnWaterPlus.addEventListener("click", function () {
      state.water = Math.min(18, state.water + 1);
      render();
    });
  var btnWaterMinus = document.getElementById("btnWaterMinus");
  if (btnWaterMinus)
    btnWaterMinus.addEventListener("click", function () {
      state.water = Math.max(0, state.water - 1);
      render();
    });
  var btnLemonPlus = document.getElementById("btnLemonPlus");
  if (btnLemonPlus)
    btnLemonPlus.addEventListener("click", function () {
      state.lemon = Math.min(4, state.lemon + 1);
      render();
    });
  var btnLemonMinus = document.getElementById("btnLemonMinus");
  if (btnLemonMinus)
    btnLemonMinus.addEventListener("click", function () {
      state.lemon = Math.max(0, state.lemon - 1);
      render();
    });
  var btnVitDPlus = document.getElementById("btnVitDPlus");
  if (btnVitDPlus)
    btnVitDPlus.addEventListener("click", function () {
      state.vitD = Math.min(8, state.vitD + 1);
      render();
    });
  var btnVitDMinus = document.getElementById("btnVitDMinus");
  if (btnVitDMinus)
    btnVitDMinus.addEventListener("click", function () {
      state.vitD = Math.max(0, state.vitD - 1);
      render();
    });
  var btnResetDay = document.getElementById("btnResetDay");
  if (btnResetDay)
    btnResetDay.addEventListener("click", function () {
      state.dailyDone = {};
      state.water = 0;
      state.lemon = 0;
      render();
    });
  var daySelect = document.getElementById("daySelect");
  if (daySelect)
    daySelect.addEventListener("change", function (e) {
      state.dayIdx = parseInt(e.target.value, 10) || 0;
      render();
    });

  var btnCheckAllGroc = document.getElementById("btnCheckAllGroc");
  if (btnCheckAllGroc)
    btnCheckAllGroc.addEventListener("click", function () {
      var grocBoxes = document.querySelectorAll("input[data-groc]");
      for (var i = 0; i < grocBoxes.length; i++) {
        state.groceriesDone[grocBoxes[i].getAttribute("data-groc")] = true;
      }
      render();
    });

  var btnResetGroc = document.getElementById("btnResetGroc");
  if (btnResetGroc)
    btnResetGroc.addEventListener("click", function () {
      state.groceriesDone = {};
      render();
    });

  var allChecks = document.querySelectorAll("input.task-check");
  for (var i = 0; i < allChecks.length; i++) {
    allChecks[i].addEventListener("change", function (e) {
      var t = e.target;
      if (t.getAttribute("data-task"))
        state.dailyDone[t.getAttribute("data-task")] = t.checked;
      if (t.getAttribute("data-chk"))
        state.checkupsDone[t.getAttribute("data-chk")] = t.checked;
      if (t.getAttribute("data-groc"))
        state.groceriesDone[t.getAttribute("data-groc")] = t.checked;
      render();
    });
  }

  var btnSaveReport = document.getElementById("btnSaveReport");
  if (btnSaveReport) {
    btnSaveReport.addEventListener("click", function () {
      var last =
        state.reports[state.reports.length - 1] || DEFAULT_REPORTS[1];
      state.reports.push({
        date: document.getElementById("inpDate").value || "2026-10-26",
        label: document.getElementById("inpLabel").value || "Follow-up",
        fasting:
          parseFloat(document.getElementById("inpFasting").value) ||
          last.fasting,
        pp: parseFloat(document.getElementById("inpPP").value) || last.pp,
        hba1c:
          parseFloat(document.getElementById("inpHba1c").value) || last.hba1c,
        tg: parseFloat(document.getElementById("inpTG").value) || last.tg,
        creat:
          parseFloat(document.getElementById("inpCreat").value) || last.creat,
        egfr:
          parseFloat(document.getElementById("inpEgfr").value) || last.egfr,
        vitD:
          parseFloat(document.getElementById("inpVitD").value) || last.vitD,
        calc:
          parseFloat(document.getElementById("inpCalc").value) || last.calc,
        yeast:
          document.getElementById("inpYeast").value !== ""
            ? parseFloat(document.getElementById("inpYeast").value)
            : last.yeast,
        crystals:
          document.getElementById("inpCrystals").value !== ""
            ? parseFloat(document.getElementById("inpCrystals").value)
            : last.crystals
      });
      render();
    });
  }

  render();
  renderCapturedSnaps();
})();
