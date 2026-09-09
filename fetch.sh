#!/bin/bash
# Fetch the Multimodal-Mind2Web test_domain split, verifying each shard.
#
# A non-empty file is not a complete one. The first version of this checked
# `-s` and a shard that stopped at 129 MB of 340 passed, then failed at read
# time with "No magic bytes found at end of file". Parquet keeps its footer at
# the end, so opening the file is the only check that means anything.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data

# Any python with duckdb installed. Override with PY=... for a venv.
PY="${PY:-python3}"
BASE="https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/resolve/main"

readable() {
    "$PY" - "$1" <<'EOF' 2>/dev/null
import duckdb, sys
duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{sys.argv[1]}')").fetchone()
EOF
}

[ -s shards.txt ] || { echo "no shards.txt"; exit 1; }

# `while read -r p` alone drops the final line when the file has no trailing
# newline, silently. That cost a shard here: shards.txt listed 11 and the loop
# fetched 10, and nothing said so. The `|| [ -n "$p" ]` keeps the last line.
fail=0
want=$(grep -c . shards.txt)
while read -r p || [ -n "$p" ]; do
    [ -n "$p" ] || continue
    f="data/$(basename "$p")"
    if readable "$f"; then
        echo "  ok      $(basename "$f")"
        continue
    fi
    [ -e "$f" ] && echo "  refetch $(basename "$f")  (unreadable)" || echo "  fetch   $(basename "$f")"
    # `-C -` resumes an interrupted part rather than restarting it, and the
    # part is NOT deleted on failure. An earlier version overwrote a completed
    # 410 MB part and then deleted it on the next failure, which turned a
    # finished download into a fresh one twice over.
    if ! curl -fL --retry 5 --retry-delay 3 --retry-all-errors -C - \
             -o "$f.part" "$BASE/$p"; then
        echo "  DOWNLOAD INCOMPLETE, part kept for resume"
        fail=1
        continue
    fi
    # Verify before the rename, so a bad download never replaces a good file.
    if readable "$f.part"; then
        mv "$f.part" "$f"
        echo "  ok      $(basename "$f")"
    else
        echo "  REFUSED $(basename "$f")  (downloaded but will not open)"
        mv "$f.part" "$f.bad"
        fail=1
    fi
done < shards.txt

# The published candidate-generation scores. Without this the ranked tables use
# a bag-of-words stand-in, which answers the central question backwards, so this
# is fetched by default rather than offered as an extra.
SCORES="data/scores_all_data.pkl"
if [ -s "$SCORES" ]; then
    echo "  ok      scores_all_data.pkl"
else
    echo "  fetch   scores_all_data.pkl  (245 MB)"
    if curl -fL --retry 5 --retry-delay 3 --retry-all-errors -C - -o "$SCORES.part" \
            "https://huggingface.co/datasets/osunlp/Mind2Web/resolve/main/scores_all_data.pkl"; then
        # A pickle that will not load is as useless as a truncated parquet, and
        # the failure would surface as a wrong number rather than an error.
        if "$PY" -c "import pickle,sys; d=pickle.load(open(sys.argv[1],'rb')); \
                     assert 'scores' in d and len(d['scores'])>1000" "$SCORES.part" 2>/dev/null; then
            mv "$SCORES.part" "$SCORES"; echo "  ok      scores_all_data.pkl"
        else
            echo "  REFUSED scores_all_data.pkl  (downloaded but will not load)"; fail=1
        fi
    else
        echo "  DOWNLOAD INCOMPLETE, part kept for resume"; fail=1
    fi
fi

n=$(ls data/test_domain-*.parquet 2>/dev/null | wc -l | tr -d ' ')
echo
echo "  $n readable shard(s) of $want listed"
# A partial split is a different measurement, so say so rather than let a
# result be quoted over whatever happened to land.
[ "$n" = "$want" ] || { echo "  INCOMPLETE: refusing to call this the split"; fail=1; }
exit $fail
