#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$(cd "$ROOT/.." && pwd)"
SDK="${ANDROID_SDK:-/tmp/android-sdk}"
BT="$(find "$SDK" -type f -name aapt2 | head -1)"
BT_DIR="$(dirname "$BT")"
ANDROID_JAR="$(find "$SDK" -type f -name android.jar | head -1)"
OUT="$APP/MaaCare.apk"
WORK="$(mktemp -d)"

chmod +x "$BT_DIR/aapt2" "$BT_DIR/d8" "$BT_DIR/zipalign" "$BT_DIR/apksigner" || true

mkdir -p "$WORK/res/mipmap-xxxhdpi" "$WORK/res/values"
cp "$ROOT/res/values/strings.xml" "$WORK/res/values/strings.xml"
cp "$APP/assets/icon-192.png" "$WORK/res/mipmap-xxxhdpi/ic_launcher.png"
cp "$ROOT/AndroidManifest.xml" "$WORK/AndroidManifest.xml"

"$BT_DIR/aapt2" compile --dir "$WORK/res" -o "$WORK/compiled.zip"
"$BT_DIR/aapt2" link \
  -o "$WORK/base.apk" \
  -I "$ANDROID_JAR" \
  --manifest "$WORK/AndroidManifest.xml" \
  --min-sdk-version 24 \
  --target-sdk-version 34 \
  --version-code 3 \
  --version-name 1.2 \
  "$WORK/compiled.zip"

javac --release 8 -classpath "$ANDROID_JAR" -d "$WORK/classes" "$ROOT/src/com/maacare/app/MainActivity.java"
mkdir -p "$WORK/dex"
"$BT_DIR/d8" --min-api 24 --output "$WORK/dex" "$WORK/classes/com/maacare/app/"*.class

cp "$WORK/base.apk" "$WORK/unsigned.apk"
zip -q -0 -j "$WORK/unsigned.apk" "$WORK/dex/classes.dex"

mkdir -p "$WORK/stage/assets/www/assets"
cp "$APP/index.html" "$APP/app.js" "$APP/service-worker.js" "$APP/manifest.json" "$WORK/stage/assets/www/"
cp -a "$APP/assets/." "$WORK/stage/assets/www/assets/"
(
  cd "$WORK/stage"
  zip -q -r -X "$WORK/unsigned.apk" assets
)

"$BT_DIR/zipalign" -f -p 4 "$WORK/unsigned.apk" "$WORK/aligned.apk"

if [[ ! -f "$ROOT/maacare.keystore" ]]; then
  keytool -genkeypair -keystore "$ROOT/maacare.keystore" -storepass maacare -keypass maacare \
    -alias maacare -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=Maa Care, OU=Family, O=Maa Care, L=Ghaziabad, C=IN"
fi

"$BT_DIR/apksigner" sign \
  --ks "$ROOT/maacare.keystore" \
  --ks-key-alias maacare \
  --ks-pass pass:maacare \
  --key-pass pass:maacare \
  --out "$OUT" \
  "$WORK/aligned.apk"
"$BT_DIR/apksigner" verify "$OUT"
echo "Wrote $OUT"
