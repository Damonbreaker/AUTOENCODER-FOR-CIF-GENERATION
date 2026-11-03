const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../logic.js");

test("two commitments on one date are both kept", function () {
  const morning = { date: "2026-11-10", repeat: "once", time: "09:00" };
  const evening = { date: "2026-11-10", repeat: "once", time: "18:30" };
  assert.equal(L.occursOn(morning, "2026-11-10"), true);
  assert.equal(L.occursOn(evening, "2026-11-10"), true);
  assert.equal(L.occursOn(evening, "2026-11-11"), false);
});

test("bmi at 173 cm and 90 kg is about 30", function () {
  assert.equal(L.bmi(90, 173), 30.1);
});

test("weekday repeat skips weekends and includes the next Monday", function () {
  const item = { date: "2026-11-06", repeat: "weekdays", time: "15:00" };
  assert.equal(L.occursOn(item, "2026-11-06"), true);
  assert.equal(L.occursOn(item, "2026-11-07"), false);
  assert.equal(L.occursOn(item, "2026-11-08"), false);
  assert.equal(L.occursOn(item, "2026-11-09"), true);
});

test("weekly repeat stays on the same weekday", function () {
  const item = { date: "2026-11-04", repeat: "weekly", time: "11:00" };
  assert.equal(L.occursOn(item, "2026-11-04"), true);
  assert.equal(L.occursOn(item, "2026-11-05"), false);
  assert.equal(L.occursOn(item, "2026-11-11"), true);
});

test("a commitment is due after the reminder time and clears when acknowledged", function () {
  const item = {
    id: "a",
    date: "2026-11-04",
    time: "16:00",
    repeat: "once",
    remindMode: "before15"
  };
  const before = new Date(2026, 10, 4, 15, 44, 0);
  const after = new Date(2026, 10, 4, 15, 45, 0);
  assert.equal(L.isDue(item, "2026-11-04", before, {}, {}), false);
  assert.equal(L.isDue(item, "2026-11-04", after, {}, {}), true);
  const ack = {};
  ack[L.occurrenceKey("a", "2026-11-04")] = true;
  assert.equal(L.isDue(item, "2026-11-04", after, ack, {}), false);
});

test("three nags produce three calendar alarms", function () {
  const item = {
    id: "meet",
    title: "JEE review",
    date: "2026-11-10",
    time: "16:00",
    repeat: "weekdays",
    remindMode: "before30",
    nag: "every15x3",
    notes: "Bring the file"
  };
  const ics = L.buildIcs(item);
  assert.equal((ics.match(/BEGIN:VALARM/g) || []).length, 3);
  assert.match(ics, /SUMMARY:JEE review/);
  assert.match(ics, /RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR/);
  assert.match(ics, /TRIGGER:-PT30M/);
});

test("default measures keep BMI 30.1 and Monday portions at the written grams", function () {
  const m = L.measures({ feet: 5, inches: 8, weightKg: 90, age: 46 });
  assert.equal(m.heightCm, 173);
  assert.equal(m.bmi, 30.1);
  assert.equal(m.factor, 1);
  const monday = L.menuFor("2026-11-02", m);
  assert.match(monday[0].detail, /15 g peanuts/);
  assert.match(monday[2].detail, /180 g toor dal/);
});

test("a lower weight reduces the gram amounts", function () {
  const light = L.menuFor("2026-11-02", { feet: 5, inches: 8, weightKg: 60, age: 46 });
  const base = L.menuFor("2026-11-02", { feet: 5, inches: 8, weightKg: 90, age: 46 });
  function gramsIn(text) {
    const found = text.match(/(\d+) g/);
    return found ? Number(found[1]) : 0;
  }
  assert.ok(gramsIn(light[2].detail) < gramsIn(base[2].detail));
  assert.equal(L.measures({ feet: 5, inches: 0, weightKg: 90, age: 46 }).inches, 0);
});

test("the week has seven different vegetarian days", function () {
  const week = L.weekMenus({ feet: 5, inches: 8, weightKg: 90, age: 46 });
  assert.equal(week.length, 7);
  const names = week.map(function (day) { return day.name; });
  assert.deepEqual(names, ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]);
  const breakfasts = week.map(function (day) { return day.meals[0].title; });
  assert.equal(new Set(breakfasts).size, 7);
  const text = JSON.stringify(week).toLowerCase();
  assert.equal(text.includes("egg"), false);
});

test("the iPhone profile only adds a full-screen Daybook icon", function () {
  const xml = L.buildWebClip("https://daybook.example/path/?a=1&b=2", "QUJD");
  assert.match(xml, /com\.apple\.webClip\.managed/);
  assert.match(xml, /<key>FullScreen<\/key>\n<true\/>/);
  assert.match(xml, /<string>https:\/\/daybook\.example\/path\/\?a=1&amp;b=2<\/string>/);
  assert.match(xml, /<key>Label<\/key>\n<string>Daybook<\/string>/);
  assert.match(xml, /<data>QUJD<\/data>/);
  assert.equal(xml.includes("VPN"), false);
  assert.equal(xml.includes("PayloadType"), true);
  const types = xml.match(/<key>PayloadType<\/key>\n<string>([^<]+)<\/string>/g);
  assert.deepEqual(types, [
    "<key>PayloadType</key>\n<string>com.apple.webClip.managed</string>",
    "<key>PayloadType</key>\n<string>Configuration</string>"
  ]);
});

test("lab anniversary is twelve months after the completed date", function () {
  assert.equal(L.labNextDue("2026-11-04", 12), "2027-11-04");
  const soon = L.labCardState(
    { completedOn: "2025-11-10", appointmentOn: "", remindDaysBefore: 14 },
    "2026-11-04"
  );
  assert.equal(soon.kind, "scheduled");
  assert.equal(soon.nudge, true);
});
