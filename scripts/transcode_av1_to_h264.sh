#!/usr/bin/env bash
# Re-encode the merged TAMP dataset videos AV1 -> H.264 (codec-only; resolution/fps/frame
# count preserved exactly) into a NEW dataset dir, so the AV1-on-CPU decode bottleneck goes
# away. Non-destructive: data/ + meta/ are copied verbatim; only videos/ are re-encoded.
#
# GOP matters a LOT for this workload: each training sample fetches 4 frames spanning a
# 15-frame window at a RANDOM offset in an 80-minute file. With a large GOP, torchcodec must
# decode a long run from the preceding keyframe (~47 ms/sample on the real files). With GOP=1
# (all-intra) every frame is independently seekable -> ~3 ms/sample. Disk is the only tradeoff
# (all-intra ~3.5x larger). Default GOP=1.
#
# Usage: GOP=1 bash scripts/transcode_av1_to_h264.sh
set -uo pipefail

GOP="${GOP:-1}"
SRC=/weka/robots/aguru/datasets/tamp_may28_topdown_flip_norand_3b_merged
DST="${SRC}_h264_g${GOP}"

echo "[$(date +%T)] SRC=$SRC"
echo "[$(date +%T)] DST=$DST   (GOP=$GOP)"

mkdir -p "$DST"
echo "[$(date +%T)] copying meta/ ..."; cp -r "$SRC/meta" "$DST/meta"
echo "[$(date +%T)] copying data/ ..."; cp -r "$SRC/data" "$DST/data"
echo "[$(date +%T)] data/meta copied."

frames() {  # robust frame count: container nb_frames, else decode-count
  local n
  n=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$1" 2>/dev/null)
  if [ -z "$n" ] || [ "$n" = "N/A" ]; then
    n=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$1" 2>/dev/null)
  fi
  echo "$n"
}
export -f frames

transcode_one() {
  local f="$1"
  local rel="${f#"$SRC"/videos/}"
  local out="$DST/videos/$rel"
  mkdir -p "$(dirname "$out")"
  local inn; inn=$(frames "$f")
  if ! ffmpeg -y -v error -nostdin -i "$f" \
        -c:v libx264 -crf 18 -g "$GOP" -preset fast -pix_fmt yuv420p -an -vsync 0 \
        -threads 4 "$out"; then
    echo "FAIL_ENCODE  $rel"; return 1
  fi
  local outn; outn=$(frames "$out")
  if [ "$inn" != "$outn" ]; then
    echo "FRAME_MISMATCH  $rel  in=$inn out=$outn"; return 1
  fi
  echo "OK  $rel  frames=$outn"
}
export -f transcode_one
export SRC DST GOP

NF=$(find "$SRC/videos" -name '*.mp4' | wc -l)
echo "[$(date +%T)] transcoding $NF video files (32-way parallel, GOP=$GOP)..."
find "$SRC/videos" -name '*.mp4' | nice -n 10 xargs -P 32 -I{} bash -c 'transcode_one "$@"' _ {}

echo "[$(date +%T)] === summary ==="
NIN=$(find "$SRC/videos" -name '*.mp4' | wc -l)
NOUT=$(find "$DST/videos" -name '*.mp4' 2>/dev/null | wc -l)
echo "input videos=$NIN  output videos=$NOUT"
echo "src size: $(du -sh "$SRC/videos" | cut -f1)   dst size: $(du -sh "$DST/videos" 2>/dev/null | cut -f1)"
if [ "$NIN" = "$NOUT" ]; then echo "ALL DONE (counts match)"; else echo "WARNING: file count mismatch"; fi
