(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.DaybookLogic = factory();
})(typeof self !== "undefined" ? self : this, function () {
  var PROFILE = {
    name: "Dr. Hariprasad Kodamana",
    role: "JEE Vice Chairperson",
    ageLabel: "45–48 years",
    heightIn: "5'8\"",
    heightCm: 173,
    weightKg: 90,
    collegeStart: "10:00",
    collegeEnd: "19:00"
  };

  function bmi(weightKg, heightCm) {
    var m = heightCm / 100;
    return Math.round((weightKg / (m * m)) * 10) / 10;
  }

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  function toISODate(date) {
    return date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate());
  }

  function parseISODate(iso) {
    var p = iso.split("-");
    return new Date(Number(p[0]), Number(p[1]) - 1, Number(p[2]));
  }

  function addDays(iso, days) {
    var d = parseISODate(iso);
    d.setDate(d.getDate() + days);
    return toISODate(d);
  }

  function addMonths(iso, months) {
    var d = parseISODate(iso);
    var day = d.getDate();
    d.setDate(1);
    d.setMonth(d.getMonth() + months);
    var last = new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate();
    d.setDate(Math.min(day, last));
    return toISODate(d);
  }

  function diffDays(fromISO, toISO) {
    var a = parseISODate(fromISO);
    var b = parseISODate(toISO);
    return Math.round((b.getTime() - a.getTime()) / 86400000);
  }

  function weekday(iso) {
    return parseISODate(iso).getDay();
  }

  function occursOn(item, iso) {
    if (!item || !item.date || !iso) return false;
    if (iso < item.date) return false;
    var repeat = item.repeat || "once";
    if (repeat === "once") return iso === item.date;
    if (repeat === "daily") return true;
    if (repeat === "weekdays") {
      var day = weekday(iso);
      return day !== 0 && day !== 6;
    }
    if (repeat === "weekly") return weekday(iso) === weekday(item.date);
    if (repeat === "monthly") return parseISODate(iso).getDate() === parseISODate(item.date).getDate();
    return false;
  }

  function minutesFromTime(time) {
    var p = String(time || "09:00").split(":");
    return Number(p[0]) * 60 + Number(p[1] || 0);
  }

  function firstOffsetMinutes(item) {
    var mode = item.remindMode || "before15";
    if (mode === "at") return 0;
    if (mode === "before15") return 15;
    if (mode === "before30") return 30;
    if (mode === "before60") return 60;
    if (mode === "morning8") {
      var eventMin = minutesFromTime(item.time);
      var morning = 8 * 60;
      if (eventMin <= morning + 15) return 0;
      return eventMin - morning;
    }
    return 15;
  }

  function nagPlan(item) {
    var nag = item.nag || "once";
    if (nag === "every15x3") return { every: 15, times: 3 };
    if (nag === "every30x3") return { every: 30, times: 3 };
    return { every: 0, times: 1 };
  }

  function alarmOffsets(item) {
    var first = firstOffsetMinutes(item);
    var plan = nagPlan(item);
    var offsets = [first];
    if (plan.every > 0) {
      for (var i = 1; i < plan.times; i++) offsets.push(first - plan.every * i);
    }
    return offsets;
  }

  function occurrenceKey(id, iso) {
    return id + "|" + iso;
  }

  function isDue(item, iso, now, ackMap, snoozeMap) {
    if (!occursOn(item, iso)) return false;
    var key = occurrenceKey(item.id, iso);
    if (ackMap && ackMap[key]) return false;
    if (snoozeMap && snoozeMap[key] && new Date(snoozeMap[key]).getTime() > now.getTime()) return false;
    var event = parseISODate(iso);
    var mins = minutesFromTime(item.time);
    event.setHours(Math.floor(mins / 60), mins % 60, 0, 0);
    var remindAt = new Date(event.getTime() - firstOffsetMinutes(item) * 60000);
    return now.getTime() >= remindAt.getTime();
  }

  function labNextDue(completedOn, intervalMonths) {
    if (!completedOn) return null;
    return addMonths(completedOn, intervalMonths || 12);
  }

  function labCardState(labs, today) {
    var completed = labs && labs.completedOn;
    var appointment = labs && labs.appointmentOn;
    var remindDays = labs && labs.remindDaysBefore ? labs.remindDaysBefore : 7;
    if (appointment && (!completed || appointment > completed)) {
      var untilAppt = diffDays(today, appointment);
      return {
        kind: "appointment",
        date: appointment,
        days: untilAppt,
        nudge: untilAppt <= remindDays
      };
    }
    if (!completed) {
      return { kind: "unbooked", date: null, days: null, nudge: false };
    }
    var next = labNextDue(completed, 12);
    var until = diffDays(today, next);
    return {
      kind: until < 0 ? "overdue" : "scheduled",
      date: next,
      days: until,
      nudge: until <= remindDays
    };
  }

  function icsEscape(text) {
    return String(text || "")
      .replace(/\\/g, "\\\\")
      .replace(/\n/g, "\\n")
      .replace(/,/g, "\\,")
      .replace(/;/g, "\\;");
  }

  function icsStamp(iso, time) {
    var mins = minutesFromTime(time);
    var d = parseISODate(iso);
    d.setHours(Math.floor(mins / 60), mins % 60, 0, 0);
    return (
      d.getFullYear() +
      pad(d.getMonth() + 1) +
      pad(d.getDate()) +
      "T" +
      pad(d.getHours()) +
      pad(d.getMinutes()) +
      "00"
    );
  }

  function rrule(item) {
    if (item.repeat === "daily") return "RRULE:FREQ=DAILY";
    if (item.repeat === "weekdays") return "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR";
    if (item.repeat === "weekly") {
      var names = ["SU", "MO", "TU", "WE", "TH", "FR", "SA"];
      return "RRULE:FREQ=WEEKLY;BYDAY=" + names[weekday(item.date)];
    }
    if (item.repeat === "monthly") return "RRULE:FREQ=MONTHLY";
    return "";
  }

  function buildIcs(item) {
    var start = icsStamp(item.date, item.time);
    var endMins = minutesFromTime(item.time) + 45;
    var endH = Math.floor(endMins / 60);
    var endM = endMins % 60;
    var endTime = pad(endH % 24) + ":" + pad(endM);
    var endDate = endH >= 24 ? addDays(item.date, 1) : item.date;
    var end = icsStamp(endDate, endTime);
    var alarms = alarmOffsets(item)
      .map(function (offset) {
        var trigger = offset === 0 ? "PT0S" : "-PT" + Math.abs(offset) + "M";
        if (offset < 0) trigger = "PT" + Math.abs(offset) + "M";
        return [
          "BEGIN:VALARM",
          "ACTION:DISPLAY",
          "DESCRIPTION:" + icsEscape(item.title),
          "TRIGGER:" + trigger,
          "END:VALARM"
        ].join("\r\n");
      })
      .join("\r\n");
    var lines = [
      "BEGIN:VCALENDAR",
      "VERSION:2.0",
      "PRODID:-//Daybook//Hariprasad Kodamana//EN",
      "CALSCALE:GREGORIAN",
      "BEGIN:VEVENT",
      "UID:" + icsEscape(item.id) + "@daybook",
      "DTSTAMP:" + icsStamp(toISODate(new Date()), pad(new Date().getHours()) + ":" + pad(new Date().getMinutes())),
      "DTSTART:" + start,
      "DTEND:" + end,
      "SUMMARY:" + icsEscape(item.title),
      "DESCRIPTION:" + icsEscape(item.notes || "Personal commitment")
    ];
    var rule = rrule(item);
    if (rule) lines.push(rule);
    lines.push(alarms, "END:VEVENT", "END:VCALENDAR");
    return lines.filter(Boolean).join("\r\n");
  }

  function numOr(value, fallback) {
    if (value === "" || value === null || value === undefined) return fallback;
    var n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function clamp(n, lo, hi) {
    if (n < lo) return lo;
    if (n > hi) return hi;
    return n;
  }

  function heightCmFrom(feet, inches) {
    return Math.round((feet * 12 + inches) * 2.54);
  }

  var BASE_BMR = 10 * 90 + 6.25 * heightCmFrom(5, 8) - 5 * 46 + 5;

  function measures(input) {
    var src = input || {};
    var feet = clamp(Math.round(numOr(src.feet, 5)), 4, 7);
    var inches = clamp(Math.round(numOr(src.inches, 8)), 0, 11);
    var weightKg = clamp(Math.round(numOr(src.weightKg, 90) * 10) / 10, 40, 200);
    var age = clamp(Math.round(numOr(src.age, 46)), 18, 90);
    var heightCm = heightCmFrom(feet, inches);
    var bmr = 10 * weightKg + 6.25 * heightCm - 5 * age + 5;
    var factor = Math.round(clamp(bmr / BASE_BMR, 0.7, 1.4) * 100) / 100;
    return {
      feet: feet,
      inches: inches,
      weightKg: weightKg,
      age: age,
      heightCm: heightCm,
      heightLabel: feet + "'" + inches + '"',
      bmi: bmi(weightKg, heightCm),
      factor: factor
    };
  }

  function grams(base, factor) {
    return Math.max(5, Math.round((base * factor) / 5) * 5);
  }

  function dayPlan(day, factor) {
    var g = function (n) { return grams(n, factor) + " g"; };
    var water = grams(1000, factor) + " ml";
    var plans = [
      ["Sunday", "Vegetable upma", g(180) + " vegetable upma, " + g(100) + " curd, one fruit. Tea or coffee, no sugar.", "2 rotis, " + g(180) + " mixed dal, " + g(150) + " bhindi or another seasonal sabzi, salad, " + g(100) + " curd.", g(25) + " roasted makhana and one fruit.", "2 rotis, " + g(180) + " moong dal, " + g(150) + " lauki, salad."],
      ["Monday", "Vegetable poha", g(180) + " vegetable poha, " + g(15) + " peanuts, " + g(80) + " curd, one fruit. Tea or coffee, no sugar.", "2 phulkas without ghee, " + g(180) + " toor dal, " + g(150) + " seasonal sabzi, salad, " + g(100) + " curd.", g(30) + " roasted chana and one fruit.", "2 rotis, " + g(180) + " masoor dal, " + g(150) + " cabbage, salad."],
      ["Tuesday", "Moong chilla", "2 moong chillas, mint chutney, " + g(100) + " curd, one fruit. Tea or coffee, no sugar.", "1 roti and " + g(100) + " cooked rice, " + g(180) + " rajma, salad, " + g(100) + " curd.", g(20) + " unsalted peanuts, " + g(20) + " murmura, one fruit.", "2 rotis, jowar or wheat, " + g(180) + " dal palak, salad."],
      ["Wednesday", "Paneer rotis", "2 small rotis filled with " + g(80) + " paneer, " + g(100) + " curd, one fruit.", "2 rotis, " + g(180) + " chole, salad, " + g(100) + " curd. Chole is the dal today.", g(30) + " roasted chana.", g(200) + " moong khichdi, " + g(100) + " curd, salad."],
      ["Thursday", "Besan chilla", "2 besan chillas, " + g(100) + " curd, one fruit. Tea or coffee, no sugar.", "2 rotis, " + g(180) + " dal, " + g(150) + " carrot and peas, salad, " + g(100) + " curd.", "One fruit and " + g(15) + " peanuts.", "2 rotis, " + g(80) + " paneer bhurji in place of dal, " + g(150) + " sabzi, salad."],
      ["Friday", "Vegetable daliya", g(180) + " vegetable daliya, " + g(100) + " curd, one fruit. Tea or coffee, no sugar.", "2 rotis, " + g(180) + " dal, " + g(120) + " dry aloo-gobi, salad, " + g(100) + " curd.", g(30) + " roasted chana and one fruit.", "2 rotis, " + g(180) + " dal, " + g(150) + " beans or peas, salad."],
      ["Saturday", "Idli or dhokla", "3 idlis or 4 pieces of dhokla, a small bowl of sambar, " + g(100) + " curd.", "1 roti and " + g(100) + " cooked rice, " + g(200) + " sambar, " + g(150) + " vegetable, " + g(100) + " curd.", "One fruit and " + g(20) + " roasted chana.", "2 rotis, " + g(180) + " dal, " + g(150) + " seasonal sabzi, salad."]
    ];
    var row = plans[day];
    return {
      name: row[0],
      meals: [
        { id: "breakfast", time: "08:15", title: row[0] + " breakfast — " + row[1], detail: row[2] },
        { id: "bottle", time: "09:40", title: "Steel bottle", detail: water + " of water. Carry the same bottle instead of buying one." },
        { id: "lunch", time: "13:15", title: row[0] + " lunch", detail: row[3] },
        { id: "snack", time: "16:30", title: row[0] + " drawer snack", detail: row[4] + " Tea without sugar." },
        { id: "water2", time: "18:30", title: "Second fill", detail: "Refill the same bottle with " + water + " and finish it before leaving." },
        { id: "dinner", time: "20:15", title: row[0] + " dinner", detail: row[5] + " Finish by 9." },
        { id: "walk", time: "21:00", title: "20-minute walk", detail: "After dinner." }
      ]
    };
  }

  function menuFor(iso, profile) {
    var m = measures(profile);
    return dayPlan(parseISODate(iso).getDay(), m.factor).meals;
  }

  function weekMenus(profile) {
    var factor = measures(profile).factor;
    var out = [];
    for (var i = 0; i < 7; i++) out.push(dayPlan(i, factor));
    return out;
  }

  var LABS = [
    {
      id: "super4",
      name: "SWASTHFIT SUPER 4",
      code: "WM17SU",
      detail: "Ask for fasting sugar. This panel covers fasting glucose, HbA1c, lipids, liver, kidney, blood count, vitamin D, and vitamin B12."
    },
    {
      id: "thyroid",
      name: "THYROID PROFILE, FREE",
      code: "Z045",
      detail: "Add this if the counter says Super 4 does not already include free T3, free T4, and TSH."
    },
    {
      id: "urine",
      name: "URINE EXAMINATION, ROUTINE",
      code: "",
      detail: "First morning sample."
    },
    {
      id: "acr",
      name: "MICROALBUMIN/ALBUMIN, 1ST MORNING / RANDOM URINE",
      code: "U027",
      detail: "A yearly kidney screen that belongs with this weight, even when he feels well."
    },
    {
      id: "bp",
      name: "Blood pressure, same morning",
      code: "",
      detail: "Ask the centre to record it. Home collection does not replace a seated reading."
    },
    {
      id: "ecg",
      name: "Resting ECG",
      code: "",
      detail: "Once this year, at the centre or a clinic. Home collection does not include an ECG."
    }
  ];

  return {
    PROFILE: PROFILE,
    bmi: bmi,
    toISODate: toISODate,
    parseISODate: parseISODate,
    addDays: addDays,
    addMonths: addMonths,
    diffDays: diffDays,
    occursOn: occursOn,
    alarmOffsets: alarmOffsets,
    occurrenceKey: occurrenceKey,
    isDue: isDue,
    labNextDue: labNextDue,
    labCardState: labCardState,
    buildIcs: buildIcs,
    measures: measures,
    menuFor: menuFor,
    weekMenus: weekMenus,
    grams: grams,
    LABS: LABS
  };
});
