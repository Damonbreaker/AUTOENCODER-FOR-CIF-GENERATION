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

  var DIET = [
    {
      id: "breakfast",
      time: "08:15",
      title: "Breakfast before leaving",
      detail: "2 eggs or 100 g paneer, one fruit, tea or coffee without sugar."
    },
    {
      id: "bottle",
      time: "09:40",
      title: "Fill a 1 litre bottle",
      detail: "Carry it into college. Water only, through the morning."
    },
    {
      id: "lunch",
      time: "13:15",
      title: "Lunch plate",
      detail: "2 rotis or one small bowl of rice, dal, sabzi, salad, and curd."
    },
    {
      id: "snack",
      time: "16:30",
      title: "Drawer snack",
      detail: "Roasted chana or a small handful of unsalted nuts, plus a fruit. Tea without sugar."
    },
    {
      id: "water2",
      time: "18:30",
      title: "Second litre finished",
      detail: "Refill once during the afternoon and finish it before 7 pm."
    },
    {
      id: "dinner",
      time: "20:15",
      title: "Dinner, finished by 9",
      detail: "Same plate as lunch. Keep rice to one small bowl."
    },
    {
      id: "walk",
      time: "21:00",
      title: "20-minute walk",
      detail: "After dinner. A corridor loop between meetings counts only for the afternoon, not for this."
    }
  ];

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
    DIET: DIET,
    LABS: LABS
  };
});
