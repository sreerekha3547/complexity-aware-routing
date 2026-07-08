"""EXP-03: Premise check — does a routable difficulty gap exist?

Runs both model tiers (Haiku 4.5 vs Opus 4.8) over a SROIE sample, scores
field-level F1 for each, and computes the per-document gap:

    gap = F1(large) - F1(small)

A document is labeled **large-required** if gap > tau, else **small-sufficient**.
The resulting distribution is the go/no-go signal for the routing premise.

Results cache per document — re-runs are free. Parallelised across docs.

    python experiments/exp_02_oracle_labeling.py --limit 10   # smoke test (~30s)
    python experiments/exp_02_oracle_labeling.py               # full run  (~2min)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data import get_loader
from src.extraction.evaluation import evaluate
from src.extraction.model_large import LargeModel
from src.extraction.model_small import SmallModel

TAU = 0.02
TABLE_DIR = Path("results/tables")

# Dataset schemas — constrain the model to ground-truth label names so that
# F1 evaluation is meaningful. Without this, models invent their own labels
# and every extraction scores F1=0 despite correct values.

SROIE_SCHEMA = {
    "company": "the business or store name",
    "date"   : "the transaction date (e.g. DD/MM/YYYY)",
    "address": "the full store or business address",
    "total"  : "the final total amount paid",
}

# CORD has 24 label types covering menu items, subtotals, and totals.
# Use the exact label strings so they match ground-truth annotations.
CORD_SCHEMA = {
    "menu.nm"                  : "name of a menu/line item",
    "menu.price"               : "price of a menu item",
    "menu.cnt"                 : "quantity or count of a menu item",
    "menu.unitprice"           : "unit price of a menu item",
    "menu.sub_nm"              : "sub-item or modifier name",
    "menu.sub_price"           : "sub-item price",
    "menu.sub_cnt"             : "sub-item quantity",
    "menu.discountprice"       : "discount applied to a menu item",
    "menu.num"                 : "menu item number or code",
    "menu.itemsubtotal"        : "subtotal for a single menu item",
    "menu.etc"                 : "other menu-level field",
    "sub_total.subtotal_price" : "subtotal before tax and service",
    "sub_total.tax_price"      : "tax amount",
    "sub_total.service_price"  : "service charge",
    "sub_total.discount_price" : "total discount amount",
    "sub_total.etc"            : "other subtotal-level field",
    "total.total_price"        : "final total amount due",
    "total.cashprice"          : "cash tendered by customer",
    "total.changeprice"        : "change returned to customer",
    "total.creditcardprice"    : "amount paid by credit card",
    "total.emoneyprice"        : "amount paid by e-money or voucher",
    "total.menutype_cnt"       : "number of distinct menu item types",
    "total.menuqty_cnt"        : "total quantity of all menu items",
    "total.total_etc"          : "other total-level field",
}

VRDU_SCHEMA = {
    "registration_num"     : "the FARA registration number assigned to the registrant",
    "registrant_name"      : "the full name of the registrant (person or organization)",
    "file_date"            : "the date the amendment was filed",
    "foreign_principle_name": "the name of the foreign principal or client being represented",
    "signer_name"          : "the full name of the person who signed the form",
    "signer_title"         : "the title or position of the signer",
}

# DocILE invoices/orders -- KILE key-value fields (line items excluded), the
# frequent (>=8%) fieldtypes from the annotated-trainval split.
DOCILE_SCHEMA = {
    "document_id"              : "the invoice or document number",
    "date_issue"              : "the date the document was issued",
    "date_due"                : "the payment due date",
    "vendor_name"             : "the seller/vendor (issuer) name",
    "vendor_address"          : "the vendor's full address",
    "vendor_tax_id"           : "the vendor's tax identification number",
    "vendor_email"            : "the vendor's email address",
    "customer_billing_name"   : "the customer/buyer (bill-to) name",
    "customer_billing_address": "the customer's billing address",
    "customer_id"             : "the customer account identifier",
    "customer_order_id"       : "the customer's order number",
    "order_id"                : "the order or purchase-order number",
    "amount_total_gross"      : "the total amount including tax",
    "amount_total_net"        : "the total amount before tax",
    "amount_total_tax"        : "the total tax amount",
    "amount_due"              : "the amount due for payment",
    "amount_paid"             : "the amount already paid",
    "currency_code_amount_due": "the currency of the amount due",
    "payment_terms"           : "the payment terms",
}

# POIE photographed nutrition labels. 21 classes: <NUTRIENT>-<MEASURE>, where
# nutrient in {CE=calories/energy, TF=total fat, SO=sodium, CAR=carbohydrate,
# PRO=protein} and measure in {PS=per serving, P1=per 100g/100ml, D=% daily
# value, PP=per package}; plus SS=serving size.
POIE_SCHEMA = {
    "SS"    : "the serving size",
    "CE-PS" : "calories/energy per serving", "CE-P1": "calories/energy per 100g or 100ml",
    "CE-D"  : "calories/energy % daily value", "CE-PP": "calories/energy per package",
    "TF-PS" : "total fat per serving", "TF-P1": "total fat per 100g or 100ml",
    "TF-D"  : "total fat % daily value", "TF-PP": "total fat per package",
    "SO-PS" : "sodium per serving", "SO-P1": "sodium per 100g or 100ml",
    "SO-D"  : "sodium % daily value", "SO-PP": "sodium per package",
    "CAR-PS": "carbohydrate per serving", "CAR-P1": "carbohydrate per 100g or 100ml",
    "CAR-D" : "carbohydrate % daily value", "CAR-PP": "carbohydrate per package",
    "PRO-PS": "protein per serving", "PRO-P1": "protein per 100g or 100ml",
    "PRO-D" : "protein % daily value", "PRO-PP": "protein per package",
}

# DeepForm: FCC political ad-buy disclosure forms (degraded scans). The five
# canonical DeepForm target fields.
DEEPFORM_SCHEMA = {
    "advertiser"  : "the advertiser (name of the campaign, committee, or sponsor)",
    "contract_num": "the contract number (unique transaction identifier)",
    "flight_from" : "the flight start date (first air date of the ad)",
    "flight_to"   : "the flight end date (last air date of the ad)",
    "gross_amount": "the gross amount / total paid for the ad buy",
}

_SCHEMAS = {
    "sroie"   : SROIE_SCHEMA,
    "cord"    : CORD_SCHEMA,
    "vrdu"    : VRDU_SCHEMA,
    "docile"  : DOCILE_SCHEMA,
    "poie"    : POIE_SCHEMA,
    "deepform": DEEPFORM_SCHEMA,
}


def _process_doc(doc, small: SmallModel, large: LargeModel, schema: dict) -> dict:
    """Run both tiers on one doc and return a result row. Thread-safe.

    F1 is scored with the field-type canonical matcher (the headline
    f1_small/f1_large/gap), and with the strict exact-match scorer alongside
    (f1_small_strict/... ) for the strict-vs-canonical transparency comparison.
    """
    rs = small.extract(doc, schema=schema)
    rl = large.extract(doc, schema=schema)
    fs = evaluate({"fields": rs.fields}, doc).f1                       # canonical
    fl = evaluate({"fields": rl.fields}, doc).f1
    fs_s = evaluate({"fields": rs.fields}, doc, canonical=False).f1    # strict
    fl_s = evaluate({"fields": rl.fields}, doc, canonical=False).f1
    return {
        "doc_id"        : doc.doc_id,
        "split"         : doc.split,
        "f1_small"      : round(fs, 4),
        "f1_large"      : round(fl, 4),
        "gap"           : round(fl - fs, 4),
        "f1_small_strict": round(fs_s, 4),
        "f1_large_strict": round(fl_s, 4),
        "gap_strict"    : round(fl_s - fs_s, 4),
        "cost_small"    : round(rs.cost_usd, 6),
        "cost_large"    : round(rl.cost_usd, 6),
        "cached"        : rs.cached and rl.cached,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sroie")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20,
                        help="parallel threads (default 20 — safe for Tier 2 accounts)")
    parser.add_argument("--tau", type=float, default=TAU)
    args = parser.parse_args()

    docs = get_loader(args.dataset).load_split(args.split)
    if args.limit:
        docs = docs[: args.limit]

    schema = _SCHEMAS.get(args.dataset)
    if schema is None:
        print(f"WARNING: no schema defined for '{args.dataset}' — "
              f"models will invent their own labels and F1 will be ~0.")
    small, large = SmallModel(), LargeModel()
    n = len(docs)
    print(f"EXP-03  {args.dataset}/{args.split}  {n} docs  "
          f"workers={args.workers}  tau={args.tau}\n")

    rows: list[dict] = []
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_doc, d, small, large, schema): d for d in docs}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as e:
                    msg = str(e)
                    if "api_key" in msg.lower() or "authentication" in msg.lower():
                        print("\nERROR: ANTHROPIC_API_KEY not set or invalid.")
                        print("  $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
                        return 2
                    if "rate_limit" in msg.lower() or "429" in msg:
                        print(f"\nRate limit hit — re-run with --workers {max(1, args.workers // 2)}")
                        return 3
                    raise
                rows.append(row)
                completed += 1
                tag = " (cached)" if row["cached"] else ""
                print(f"  [{completed:>3}/{n}] {row['doc_id']:<20}  "
                      f"small={row['f1_small']:.3f}  large={row['f1_large']:.3f}  "
                      f"gap={row['gap']:+.3f}{tag}")
    except KeyboardInterrupt:
        print(f"\nInterrupted at {completed}/{n} docs — partial results saved.")

    if not rows:
        return 1

    df = pd.DataFrame(rows)
    df["tier_label"] = df["gap"].apply(
        lambda g: "large-required" if g > args.tau else "small-sufficient"
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"oracle_labels_{args.dataset}_{args.split}.csv"
    df.to_csv(TABLE_DIR / out_name, index=False)

    n_done = len(df)
    n_large = (df["tier_label"] == "large-required").sum()

    print(f"\n{'─'*55}")
    print(f"  docs processed   : {n_done}")
    print(f"  mean F1 small    : {df['f1_small'].mean():.3f}")
    print(f"  mean F1 large    : {df['f1_large'].mean():.3f}")
    print(f"  mean gap         : {df['gap'].mean():+.3f}  (max {df['gap'].max():+.3f})")
    print(f"  large-required   : {n_large}/{n_done}  ({100*n_large/n_done:.0f}%)")
    print(f"  small-sufficient : {n_done-n_large}/{n_done}  ({100*(n_done-n_large)/n_done:.0f}%)")
    print(f"\n  cost always-small : ${df['cost_small'].sum():.4f}")
    print(f"  cost always-large : ${df['cost_large'].sum():.4f}")
    print(f"\n  -> {TABLE_DIR}/{out_name}")

    if n_large == 0:
        print("\n  VERDICT: no routable gap — small model suffices on this sample.")
    elif n_large == n_done:
        print("\n  VERDICT: every doc needs the large model — nothing to save by routing.")
    else:
        print(f"\n  VERDICT: routable gap exists ({100*n_large/n_done:.0f}% need large). "
              f"Premise holds — proceed to oracle labeling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
