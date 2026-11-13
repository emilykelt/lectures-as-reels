#!/bin/bash
# Download a Panopto lecture (speaker + slides streams) using an exported cookies.txt.
#   ./download.sh "https://cambridgelectures.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=..." cookies.txt
# Get cookies.txt with the "Get cookies.txt LOCALLY" browser extension while on the lecture page.
set -e
cd "$(dirname "$0")"
URL="$1"; COOKIES="${2:-$HOME/Downloads/cambridgelectures.cloud.panopto.eu_cookies.txt}"
[ -f "$COOKIES" ] || { echo "cookies file not found: $COOKIES"; exit 1; }
mkdir -p lectures
.venv/bin/yt-dlp --cookies "$COOKIES" -F "$URL"
.venv/bin/yt-dlp --cookies "$COOKIES" -f "all[vcodec!=none]" \
  -o "lectures/%(title)s [%(format_id)s].%(ext)s" "$URL"
echo; echo "Saved to ./lectures/ :"; ls -la lectures
