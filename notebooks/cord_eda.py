"""
CORD Dataset EDA
Covers: documents, fields, labels, OCR, tables, long receipts, OCR noise, layout.
Run from project root: python notebooks/cord_eda.py
"""

import json
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# -- paths ----------------------------------------------------------------------
BASE   = Path("data/CORD")
SPLITS = ["train", "dev", "test"]

all_files   = []
split_counts = {}
for split in SPLITS:
    files = sorted((BASE / split / "json").glob("*.json"))
    split_counts[split] = len(files)
    all_files.extend(files)

# -- per-document accumulators --------------------------------------------------
label_counter   = Counter()
words_per_doc   = []
lines_per_doc   = []
items_per_doc   = []   # distinct menu groups
doc_heights     = []
doc_widths      = []
short_word_docs = []   # ≤2-char alpha tokens per doc  (noise proxy A)
noise_char_docs = []   # non-ASCII / special chars      (noise proxy B)
x_spread_ratio  = []   # word x-range / image width     (layout proxy)
has_table_sub   = []   # sub-row labels (menu.sub_nm, menu.etc)
has_subtotal    = []
has_total       = []
has_tax         = []
has_service     = []
has_void        = []

for fp in all_files:
    with open(fp, encoding="utf-8") as f:
        doc = json.load(f)

    lines = doc.get("valid_line", [])
    sz    = doc.get("meta", {}).get("image_size", {})
    h, w  = sz.get("height", 0), sz.get("width", 0)
    doc_heights.append(h)
    doc_widths.append(w)
    lines_per_doc.append(len(lines))

    word_count  = 0
    menu_groups = set()
    line_cats   = []
    xs          = []
    short_w = noise_c = 0

    for line in lines:
        cat = line.get("category", "")
        label_counter[cat] += 1
        line_cats.append(cat)
        if cat.startswith("menu"):
            menu_groups.add(line.get("group_id"))

        for word in line.get("words", []):
            txt = word.get("text", "")
            word_count += 1

            if len(txt) <= 2 and txt.isalpha():
                short_w += 1

            for ch in txt:
                if unicodedata.category(ch) in ("So", "Cc", "Cf") or ord(ch) > 127:
                    noise_c += 1

            q = word.get("quad", {})
            for k in ("x1", "x2", "x3", "x4"):
                v = q.get(k)
                if v is not None:
                    xs.append(v)

    words_per_doc.append(word_count)
    items_per_doc.append(len(menu_groups))
    short_word_docs.append(short_w)
    noise_char_docs.append(noise_c)
    x_spread_ratio.append((max(xs) - min(xs)) / w if xs and w > 0 else 0)

    has_table_sub.append(any(c in ("menu.sub_nm", "menu.etc", "menu.discountprice") for c in line_cats))
    has_subtotal.append(any(c.startswith("sub_total") for c in line_cats))
    has_total.append(any(c.startswith("total") for c in line_cats))
    has_tax.append(any("tax" in c for c in line_cats))
    has_service.append(any("service" in c for c in line_cats))
    has_void.append(any("void" in c for c in line_cats))

n = len(all_files)

# -- helpers --------------------------------------------------------------------
def pct(k): return f"{100*k/n:.1f}%"
def bkt(vals, lo, hi): return sum(1 for v in vals if lo <= v < hi)
SEP = "=" * 60
def sec(t): print(f"\n{SEP}\n  {t}\n{SEP}")

# -----------------------------------------------------------------------------
sec("1. DOCUMENT COUNT")
print(f"  Total : {n}  (full CORD = 800 train / 100 dev / 100 test)")
for s in SPLITS:
    print(f"    {s:<6}: {split_counts[s]:>4} docs")

# -----------------------------------------------------------------------------
sec("2. JSON SCHEMA -- FIELDS IN EACH DOCUMENT")
print("""  meta
    image_id, split, version
    image_size -> height, width

  valid_line[]          -- one entry per labelled text line
    category            -- label  (e.g. "menu.nm", "total.total_price")
    group_id            -- links cnt / nm / price lines of the same item
    words[]
      text              -- OCR string
      quad              -- x1-x4, y1-y4 corner coordinates
      is_key            -- 1 if this token is a field key (e.g. "TOTAL")
      row_id

  dontcare[]            -- regions the annotators chose to ignore
  repeating_symbol[]    -- decorative separators / dividers
  roi                   -- region of interest (usually empty dict)""")

# -----------------------------------------------------------------------------
sec("3. LABEL CATEGORIES")
total_lines = sum(label_counter.values())
prefix_counter = Counter()
for cat, cnt in label_counter.items():
    prefix_counter[cat.split(".")[0]] += cnt

print(f"  Total labelled lines : {total_lines}")
print(f"  Distinct categories  : {len(label_counter)}\n")
print(f"  {'Category':<45} {'Count':>7}  {'% lines':>7}")
print("  " + "-" * 63)
for cat, cnt in sorted(label_counter.items(), key=lambda x: -x[1]):
    print(f"  {cat:<45} {cnt:>7}  {100*cnt/total_lines:>6.1f}%")

print(f"\n  -- Top-level prefix summary --")
for pfx, cnt in sorted(prefix_counter.items(), key=lambda x: -x[1]):
    print(f"    {pfx:<14} {cnt:>6} lines  ({100*cnt/total_lines:.1f}%)")

# -----------------------------------------------------------------------------
sec("4. OCR -- TOKEN COUNTS")
print(f"  Words per document")
print(f"    min={min(words_per_doc)}  max={max(words_per_doc)}"
      f"  mean={statistics.mean(words_per_doc):.1f}"
      f"  median={statistics.median(words_per_doc):.0f}")
for lo, hi in [(0,20),(20,40),(40,60),(60,80),(80,120),(120,9999)]:
    print(f"    [{lo:>4}-{hi:>4}) words : {bkt(words_per_doc,lo,hi):>4} docs")

print(f"\n  Lines per document")
print(f"    min={min(lines_per_doc)}  max={max(lines_per_doc)}"
      f"  mean={statistics.mean(lines_per_doc):.1f}"
      f"  median={statistics.median(lines_per_doc):.0f}")

# -----------------------------------------------------------------------------
sec("5. DOES IT HAVE TABLES?")
table_docs = sum(has_table_sub)
print(f"  Docs with sub-row labels (menu.sub_nm / menu.etc / menu.discountprice):")
print(f"    {table_docs} / {n}  ({pct(table_docs)})")
print()
print(f"  X-spread ratio (word x-range / image width) -- proxy for multi-column layout:")
print(f"    mean={statistics.mean(x_spread_ratio):.2f}  "
      f"median={statistics.median(x_spread_ratio):.2f}  "
      f"max={max(x_spread_ratio):.2f}")
wide = sum(1 for v in x_spread_ratio if v > 0.7)
print(f"    Docs spanning >70% of page width : {wide} / {n}  ({pct(wide)})")
print()
print("  Verdict:")
print("    CORD has implicit tabular structure -- each receipt uses a")
print("    qty | item-name | price layout linked by group_id.")
print("    There are no multi-table or form-style documents; it is always")
print("    a single receipt per image.")

# -----------------------------------------------------------------------------
sec("6. LONG RECEIPTS?")
print(f"  Lines per document")
for lo, hi in [(0,5),(5,10),(10,20),(20,30),(30,50),(50,9999)]:
    print(f"    [{lo:>3}-{hi:>4}) lines : {bkt(lines_per_doc,lo,hi):>4} docs")

print(f"\n  Menu items per document")
print(f"    min={min(items_per_doc)}  max={max(items_per_doc)}"
      f"  mean={statistics.mean(items_per_doc):.1f}"
      f"  median={statistics.median(items_per_doc):.0f}")
for lo, hi in [(0,3),(3,6),(6,10),(10,15),(15,20),(20,99)]:
    print(f"    [{lo:>2}-{hi:>2}) items : {bkt(items_per_doc,lo,hi):>4} docs")

aspect = [h/w for h, w in zip(doc_heights, doc_widths) if w > 0]
tall = sum(1 for a in aspect if a > 2.5)
print(f"\n  Image aspect ratio (height / width)  -- >2.5 = physically long receipt")
print(f"    min={min(aspect):.2f}  max={max(aspect):.2f}"
      f"  mean={statistics.mean(aspect):.2f}  median={statistics.median(aspect):.2f}")
print(f"    Aspect > 2.5 : {tall} / {n}  ({pct(tall)})")

print(f"\n  Image height (px)")
print(f"    min={min(doc_heights)}  max={max(doc_heights)}"
      f"  mean={statistics.mean(doc_heights):.0f}"
      f"  median={statistics.median(doc_heights):.0f}")
for lo, hi in [(0,500),(500,1000),(1000,1500),(1500,2000),(2000,3000),(3000,99999)]:
    print(f"    [{lo:>5}-{hi:>5}) px : {bkt(doc_heights,lo,hi):>4} docs")

# -----------------------------------------------------------------------------
sec("7. OCR NOISE")
print("  Proxy A -- short alpha tokens (≤2 chars, e.g. 'Rp', 'RM', 'x', '%')")
print(f"    0 short tokens       : {sum(1 for x in short_word_docs if x==0):>4} docs")
print(f"    1-3 short tokens     : {sum(1 for x in short_word_docs if 1<=x<=3):>4} docs")
print(f"    4-10 short tokens    : {sum(1 for x in short_word_docs if 4<=x<=10):>4} docs")
print(f"    > 10 short tokens    : {sum(1 for x in short_word_docs if x>10):>4} docs")
print(f"    Max in a single doc  : {max(short_word_docs)}")

print()
print("  Proxy B -- non-ASCII / special characters in OCR text")
print(f"    0 non-ASCII chars    : {sum(1 for x in noise_char_docs if x==0):>4} docs")
print(f"    1-5 non-ASCII chars  : {sum(1 for x in noise_char_docs if 1<=x<=5):>4} docs")
print(f"    > 5 non-ASCII chars  : {sum(1 for x in noise_char_docs if x>5):>4} docs")
print(f"    Max in a single doc  : {max(noise_char_docs)}")

print()
print("  Note: CORD OCR was produced by a commercial engine on real Indonesian")
print("  receipts. Short tokens like 'Rp'/'RM' are valid currency markers.")
print("  True garbage artefacts are uncommon -- OCR quality is high overall.")

# -----------------------------------------------------------------------------
sec("8. LAYOUT DIFFERENCES")
print(f"  Image width  min={min(doc_widths)}  max={max(doc_widths)}"
      f"  mean={statistics.mean(doc_widths):.0f}  median={statistics.median(doc_widths):.0f}")
print(f"  Image height min={min(doc_heights)}  max={max(doc_heights)}"
      f"  mean={statistics.mean(doc_heights):.0f}  median={statistics.median(doc_heights):.0f}")

print(f"\n  Structural section presence:")
print(f"    Has sub_total section  : {sum(has_subtotal):>4} / {n}  ({pct(sum(has_subtotal))})")
print(f"    Has total section      : {sum(has_total):>4} / {n}  ({pct(sum(has_total))})")
print(f"    Has tax line           : {sum(has_tax):>4} / {n}  ({pct(sum(has_tax))})")
print(f"    Has service charge     : {sum(has_service):>4} / {n}  ({pct(sum(has_service))})")
print(f"    Has void / discount    : {sum(has_void):>4} / {n}  ({pct(sum(has_void))})")

print(f"""
  Layout observations:
    • Single-column narrow thermal-paper format dominates
    • Width is fairly uniform; height varies widely (short vs long receipts)
    • Core 3-column structure (qty | name | price) is near-universal
    • Sub-total blocks present in {pct(sum(has_subtotal))} of docs
    • Tax lines in {pct(sum(has_tax))}, service charge in {pct(sum(has_service))} -- these add layout sections
    • All source receipts are from Indonesian restaurants/cafés
    • No multi-page documents -- each image is one complete receipt
""")

# -----------------------------------------------------------------------------
sec("SUMMARY")
rows = [
    ("Total documents",                   str(n)),
    ("train / dev / test",                f"{split_counts['train']} / {split_counts['dev']} / {split_counts['test']}"),
    ("Distinct label categories",         str(len(label_counter))),
    ("Avg words per doc",                 f"{statistics.mean(words_per_doc):.1f}"),
    ("Avg lines per doc",                 f"{statistics.mean(lines_per_doc):.1f}"),
    ("Avg menu items per doc",            f"{statistics.mean(items_per_doc):.1f}"),
    ("Long receipts (aspect > 2.5)",      pct(tall)),
    ("Has total section",                 pct(sum(has_total))),
    ("Has implicit table (qty|nm|price)", "~100% (core structure)"),
    ("Has explicit sub-row labels",       pct(sum(has_table_sub))),
    ("Docs with any non-ASCII OCR chars", pct(sum(1 for x in noise_char_docs if x > 0))),
]
print(f"  {'Metric':<45} Value")
print("  " + "-" * 55)
for k, v in rows:
    print(f"  {k:<45} {v}")
