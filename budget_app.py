import json
import os
from datetime import date, datetime, timedelta
import calendar

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monthly Budget Tracker", page_icon="💰", layout="wide")

# ----------------- Config -----------------
DATA_FILE = "budget_data.json"
CATEGORIES = ["Necessity", "Need", "Want", "Savings"]
MONTH_FMT = "%Y-%m"

# ----------------- Helpers -----------------
def month_key(d: date) -> str:
    return d.strftime(MONTH_FMT)

def start_of_month(d: date) -> date:
    return d.replace(day=1)

def end_of_month(d: date) -> date:
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.replace(day=last_day)

@st.cache_data(show_spinner=False)
def _bootstrap():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
    return True

def load_data():
    _bootstrap()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw

def save_data(dobj):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(dobj, f, indent=2)

def inr(x: float) -> str:
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return "₹0"

def safe_sum(series):
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())

# ----------------- Default Month Template -----------------
def default_month_payload(ym: str):
    """Create a month dict with inputs based on user's template for December by default."""
    year, month = map(int, ym.split("-"))

    payload = {
        "incomes": [
            {"source": "Salary", "amount": 73833.0, "note": "Monthly salary"},
            {"source": "Meal card", "amount": 2200.0, "note": "Food wallet / card"},
        ],
        "expenses": [
            {"item": "Rent", "amount": 10500.0, "category": "Necessity", "note": "House rent"},
            {"item": "Bike Petrol", "amount": 4000.0, "category": "Necessity", "note": "Fuel"},
            {"item": "Bike EMI", "amount": 6181.0, "category": "Necessity", "note": "EMI"},
            {"item": "Send to Sister", "amount": 10000.0, "category": "Necessity", "note": "Family support"},
            {"item": "Investment", "amount": 5000.0, "category": "Necessity", "note": "SIP / Investment"},
            # December tracking (personal expense plan)
            {"item": "Ear bud EMI", "amount": 665.0, "category": "Want", "note": "Gadget EMI"},
            {"item": "Meds", "amount": 2100.0, "category": "Necessity", "note": "Health"},
            {"item": "GYM", "amount": 2000.0, "category": "Need", "note": "Fitness"},
            {"item": "Clothing (wardrobe)", "amount": 5000.0, "category": "Want", "note": "Clothes"},
            {"item": "Food (outside)", "amount": 3000.0, "category": "Want", "note": "Eating out"},
            {"item": "December trip budget", "amount": 7000.0, "category": "Want", "note": "Trip"},
            {"item": "Hair care", "amount": 2000.0, "category": "Necessity", "note": "Hair"},
            {"item": "Future travel fund", "amount": 5000.0, "category": "Necessity", "note": "Set aside"},
            {"item": "Emergency/Savings (target)", "amount": 1587.0, "category": "Savings", "note": "Month-end left"},
        ],
        "planned_savings": [
            {"name": "Education Loan monthly savings", "amount": 12000.0, "note": "1.44 L in 12 months", "category": "Savings"},
        ],
        "one_off_inflows": [
            {"name": "Security Deposit refund", "amount": 12571.0, "month": f"{year}-{(month % 12) + 1:02d}", "note": "Add to future travels"},
        ],
    }
    return payload

# ----------------- Load / Save -----------------
data = load_data()

# ----------------- Sidebar: Backup & Tools -----------------
st.sidebar.header("⚙️ Settings & Backup")
export_json = json.dumps({"data": data}, indent=2)
st.sidebar.download_button("⬇️ Export JSON", data=export_json, file_name="budget-data.json", mime="application/json")

up = st.sidebar.file_uploader("⬆️ Import JSON", type=["json"])
if up is not None:
    try:
        obj = json.load(up)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
            data = obj["data"]
            save_data(data)
            st.sidebar.success("Imported!")
        else:
            st.sidebar.error("Invalid JSON structure.")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

st.sidebar.divider()
st.sidebar.subheader("🧹 Clear Month")
clear_month = st.sidebar.date_input("Select month to clear", value=date.today().replace(day=1))
if st.sidebar.button("Clear selected month"):
    mkey_cm = month_key(clear_month)
    if mkey_cm in data:
        data.pop(mkey_cm, None)
        save_data(data)
    st.sidebar.success(f"Cleared {clear_month.strftime('%B %Y')}")

# ----------------- Header -----------------
st.title("💰 Monthly Budget Tracker")
st.caption("Plan → Track → Save. Categories: Necessity, Need, Want, Savings.")

# ----------------- Month Selector & Load -----------------
col_sel, col_new = st.columns([2, 1])
with col_sel:
    msel = st.date_input("📅 Select month", value=date.today().replace(day=1))
    mkey = month_key(msel)

with col_new:
    if st.button("📄 Use Template for this month", use_container_width=True):
        if mkey not in data:
            data[mkey] = default_month_payload(mkey)
            save_data(data)
            st.success("Template applied!")
        else:
            st.info("Month already exists. Edit below.")

month_obj = data.get(mkey, {
    "incomes": [],
    "expenses": [],
    "planned_savings": [],
    "one_off_inflows": [],
})

# ----------------- Editors -----------------
left, right = st.columns([1, 1], vertical_alignment="top")

with left:
    st.subheader("Income")
    df_inc = pd.DataFrame(month_obj.get("incomes", []))
    if df_inc.empty:
        df_inc = pd.DataFrame([{"source": "", "amount": 0.0, "note": ""}])
    if "delete" not in df_inc.columns:
        df_inc["delete"] = False

    edited_inc = st.data_editor(
        df_inc,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "amount": st.column_config.NumberColumn("amount", min_value=0.0, step=100.0),
            "delete": st.column_config.CheckboxColumn("❌ Delete", help="Tick to remove this row on save"),
        },
    )

    st.subheader("Planned Savings / Allocations")
    df_ps = pd.DataFrame(month_obj.get("planned_savings", []))
    if df_ps.empty:
        df_ps = pd.DataFrame([{"name": "", "amount": 0.0, "note": "", "category": "Savings"}])
    if "delete" not in df_ps.columns:
        df_ps["delete"] = False

    edited_ps = st.data_editor(
        df_ps,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "amount": st.column_config.NumberColumn("amount", min_value=0.0, step=100.0),
            "category": st.column_config.SelectboxColumn("category", options=CATEGORIES, default="Savings"),
            "delete": st.column_config.CheckboxColumn("❌ Delete", help="Tick to remove this row on save"),
        },
    )

with right:
    st.subheader("Expenses")
    df_exp = pd.DataFrame(month_obj.get("expenses", []))

    # --- Backward compatibility & sensible defaults ---
    if df_exp.empty:
        df_exp = pd.DataFrame([{
            "item": "", "planned": 0.0, "actual": 0.0,
            "category": "Necessity", "note": ""
        }])
    else:
        if "planned" not in df_exp.columns:
            if "amount" in df_exp.columns:
                df_exp["planned"] = pd.to_numeric(df_exp["amount"], errors="coerce").fillna(0.0)
            else:
                df_exp["planned"] = 0.0
        if "actual" not in df_exp.columns:
            if "amount" in df_exp.columns:
                df_exp["actual"] = pd.to_numeric(df_exp["amount"], errors="coerce").fillna(0.0)
            else:
                df_exp["actual"] = 0.0
        if "category" not in df_exp.columns:
            df_exp["category"] = "Necessity"
        if "note" not in df_exp.columns:
            df_exp["note"] = ""

    if "delete" not in df_exp.columns:
        df_exp["delete"] = False

    # variance (read-only view)
    _tmp = df_exp.copy()
    _tmp["variance"] = pd.to_numeric(_tmp.get("actual", 0), errors="coerce").fillna(0.0) - \
                       pd.to_numeric(_tmp.get("planned", 0), errors="coerce").fillna(0.0)

    edited_exp = st.data_editor(
        _tmp,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "planned": st.column_config.NumberColumn("planned", min_value=0.0, step=100.0, help="Budget target"),
            "actual":  st.column_config.NumberColumn("actual",  min_value=0.0, step=100.0, help="What you actually spent"),
            "variance": st.column_config.NumberColumn("variance (actual - planned)", disabled=True),
            "category": st.column_config.SelectboxColumn("category", options=CATEGORIES, default="Necessity"),
            "delete": st.column_config.CheckboxColumn("❌ Delete", help="Tick to remove this row on save"),
        },
        hide_index=True,
    )

    # remove helper column before later use
    edited_exp = edited_exp.drop(columns=["variance"], errors="ignore")

st.divider()

# ----------------- Compute Metrics -----------------
# filter out rows marked for deletion
inc_eff = edited_inc[edited_inc.get("delete") != True].drop(columns=["delete"], errors="ignore").copy()
exp_eff = edited_exp[edited_exp.get("delete") != True].drop(columns=["delete"], errors="ignore").copy()
ps_eff  = edited_ps[edited_ps.get("delete") != True].drop(columns=["delete"], errors="ignore").copy()

# Income / Expenses (planned vs actual)
total_income = safe_sum(inc_eff.get("amount", []))
planned_expense_total = safe_sum(exp_eff.get("planned", []))
actual_expense_total  = safe_sum(exp_eff.get("actual", []))

# Fallback for older rows with only 'amount'
if actual_expense_total == 0 and "amount" in exp_eff.columns:
    actual_expense_total = safe_sum(exp_eff.get("amount", []))
if planned_expense_total == 0 and "amount" in exp_eff.columns:
    planned_expense_total = safe_sum(exp_eff.get("amount", []))

expense_total = actual_expense_total  # rest of the app uses ACTUAL

# Category break-up by ACTUAL
edited_exp_num = exp_eff.copy()
if "actual" not in edited_exp_num.columns:
    edited_exp_num["actual"] = pd.to_numeric(edited_exp_num.get("amount", 0), errors="coerce").fillna(0)
else:
    edited_exp_num["actual"] = pd.to_numeric(edited_exp_num.get("actual", 0), errors="coerce").fillna(0)

cat_break = (
    edited_exp_num.groupby("category")["actual"].sum().reindex(CATEGORIES).fillna(0)
)

necessity_spend = float(cat_break.get("Necessity", 0.0))
need_spend      = float(cat_break.get("Need", 0.0))
want_spend      = float(cat_break.get("Want", 0.0))

# Left after necessities only (as per template)
left_after_necessity = total_income - necessity_spend

# Net cashflow after all expenses + planned savings
planned_savings_amt = safe_sum(ps_eff.get("amount", []))
net_left = total_income - (expense_total + planned_savings_amt)

# Total Savings = Planned Savings + positive leftover (if any)
total_savings = float(planned_savings_amt) + max(float(net_left), 0.0)

# ----------------- Persist current edits -----------------
if st.button("💾 Save Month", use_container_width=True):
    inc_clean = inc_eff.copy()
    exp_clean = exp_eff.copy()
    ps_clean  = ps_eff.copy()

    data[mkey] = {
        "incomes": inc_clean.to_dict(orient="records"),
        "expenses": exp_clean.to_dict(orient="records"),
        "planned_savings": ps_clean.to_dict(orient="records"),
        "one_off_inflows": month_obj.get("one_off_inflows", []),
    }
    save_data(data)
    st.success(f"Saved for {mkey} (deleted rows removed)")

# ----------------- Metrics Row -----------------
category_totals = {
    "Necessity": necessity_spend,
    "Need": need_spend,
    "Want": want_spend,
    "Savings": planned_savings_amt,
}

st.markdown("## 🧾 Monthly Summary")
st.markdown("<br>", unsafe_allow_html=True)

# Planned vs Actual Expenses snapshot
rp1, rp2, rp3 = st.columns([1,1,1])
rp1.metric("Planned Expenses", inr(planned_expense_total))
rp2.metric("Actual Expenses",  inr(actual_expense_total))
rp3.metric("Variance",         inr(actual_expense_total - planned_expense_total))

st.markdown("<br>", unsafe_allow_html=True)

# Top row — Income / Expense / Savings
r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
r1c1.metric("Total Income", inr(total_income))
r1c2.metric("Total Expenses (Actual)", inr(expense_total))
r1c3.metric("Planned Savings", inr(planned_savings_amt))

st.markdown("<br>", unsafe_allow_html=True)

# Second row — Net values
r2c1, r2c2, r2c3 = st.columns([1, 1, 1])
r2c1.metric("Left after Necessity (Actual)", inr(left_after_necessity))
r2c2.metric("Net Left (month)", inr(net_left))
r2c3.metric("Total Savings", inr(total_savings))

st.markdown("<br>", unsafe_allow_html=True)

# Third row — Category totals
r3c1, r3c2, r3c3  = st.columns([1, 1, 1])
r3c1.metric("Necessity Spend (Actual)", inr(category_totals["Necessity"]))
r3c2.metric("Need Spend (Actual)", inr(category_totals["Need"]))
r3c3.metric("Want Spend (Actual)", inr(category_totals["Want"]))

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- Visuals -----------------
st.subheader("Expense Mix (by Category)")
fig_mix, axm = plt.subplots(figsize=(6, 4), dpi=120)
fig_mix.patch.set_alpha(0)
axm.set_facecolor("none")

vals = [necessity_spend, need_spend, want_spend]
vals = list(np.nan_to_num(np.array(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
labels = ["Necessity", "Need", "Want"]

if sum(vals) > 0:
    axm.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90)
    axm.axis("equal")
else:
    axm.text(0.5, 0.5, "No expense data for this month", ha="center", va="center", transform=axm.transAxes)
    axm.set_axis_off()

st.pyplot(fig_mix, transparent=True)

st.subheader("Month at a Glance")
summary_df = pd.DataFrame(
    {
        "Metric": [
            "Total Income",
            "Planned Expenses",
            "Actual Expenses",
            "Planned Savings",
            "Left after Necessity (Actual)",
            "Net Left (month)",
            "Total Savings",
        ],
        "Amount": [
            total_income,
            planned_expense_total,
            actual_expense_total,
            planned_savings_amt,
            left_after_necessity,
            net_left,
            total_savings,
        ],
    }
)
st.dataframe(summary_df, use_container_width=True)

# ----------------- Roll-up Analytics (last 6 months) -----------------
st.divider()
st.header("📊 Month-over-Month (last 6 months)")

cur_first = start_of_month(msel)
months = []
net_left_series = []
exp_series = []
inc_series = []

for i in range(5, -1, -1):
    mnum = (cur_first.year * 12 + cur_first.month - 1) - i
    y = mnum // 12
    m = (mnum % 12) + 1
    d = date(y, m, 1)
    k = month_key(d)
    months.append(d.strftime("%b"))

    mo = data.get(k)
    if not mo:
        inc = 0.0
        ex = 0.0
        ps = 0.0
    else:
        inc = safe_sum(pd.DataFrame(mo.get("incomes", []))["amount"]) if mo.get("incomes") else 0.0

        exp_df = pd.DataFrame(mo.get("expenses", [])) if mo.get("expenses") else pd.DataFrame()
        if not exp_df.empty:
            if "actual" in exp_df.columns:
                ex = safe_sum(exp_df["actual"])
            elif "amount" in exp_df.columns:
                ex = safe_sum(exp_df["amount"])
            else:
                ex = 0.0
        else:
            ex = 0.0

        ps  = safe_sum(pd.DataFrame(mo.get("planned_savings", []))["amount"]) if mo.get("planned_savings") else 0.0

    inc_series.append(inc)
    exp_series.append(ex)
    net_left_series.append(inc - (ex + ps))

# ---- SANITIZE + GUARDS ----
def _clean(arr):
    a = np.array(arr, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

inc_arr = _clean(inc_series)
exp_arr = _clean(exp_series)
net_arr = _clean(net_left_series)

if len(months) == 0:
    st.info("No months to display yet.")
else:
    all_zero = (np.all(inc_arr == 0) and np.all(exp_arr == 0) and np.all(net_arr == 0))
    if all_zero:
        st.info("No data for the last 6 months to plot yet.")
    else:
        fig_bar, axb = plt.subplots(figsize=(8, 4), dpi=120)
        fig_bar.patch.set_alpha(0)
        axb.set_facecolor("none")

        x = np.arange(len(months))
        barw = 0.25
        axb.bar(x - barw, inc_arr, width=barw, label="Income")
        axb.bar(x,         exp_arr, width=barw, label="Expenses (Actual)")
        axb.bar(x + barw,  net_arr, width=barw, label="Net Left")
        axb.set_xticks(x)
        axb.set_xticklabels(months)
        axb.set_ylabel("Amount")
        axb.legend()
        axb.grid(True, linestyle="--", alpha=0.4)

        peak = float(np.max(np.abs(net_arr)))
        if peak > 0:
            bump = peak * 0.02
            for idx, val in enumerate(net_arr):
                ypos = float(val) + (bump if val >= 0 else -bump)
                axb.text(idx + barw, ypos, inr(val), ha="center", va="bottom" if val>=0 else "top", fontsize=8)

        st.pyplot(fig_bar, use_container_width=True)

# ----------------- One-off inflows -----------------
st.divider()
st.subheader("📥 One-off / Future Inflows")
df_one = pd.DataFrame(month_obj.get("one_off_inflows", []))
if df_one.empty:
    df_one = pd.DataFrame([{"name": "", "amount": 0.0, "month": month_key(msel), "note": ""}])
edited_one = st.data_editor(
    df_one,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "amount": st.column_config.NumberColumn("amount", min_value=0.0, step=100.0),
        "month": st.column_config.TextColumn("month (YYYY-MM)")
    },
)

if st.button("💾 Save One-off Inflows", use_container_width=True):
    data.setdefault(mkey, {
        "incomes": [],
        "expenses": [],
        "planned_savings": [],
        "one_off_inflows": [],
    })
    data[mkey]["one_off_inflows"] = edited_one.to_dict(orient="records")
    save_data(data)
    st.success("One-off inflows saved.")

upcoming = []
for k_arch, mo in data.items():
    for row in mo.get("one_off_inflows", []):
        if row.get("month") == mkey:
            upcoming.append(row)

if upcoming:
    st.info("Upcoming inflows this month detected. They are **not** added to income automatically; add to Income if received.")
    st.table(pd.DataFrame(upcoming))

# ----------------- Current Month Sheet & Saved History -----------------
st.divider()
st.header("🧾 Current Month Sheet & Saved History")

cur_sheet_rows = []
for r in inc_eff.to_dict(orient="records"):
    cur_sheet_rows.append({"Type": "Income", "Name": r.get("source", ""), "Category": "Income", "Amount": r.get("amount", 0.0), "Note": r.get("note", "")})

for r in exp_eff.to_dict(orient="records"):
    amt_for_sheet = r.get("actual", r.get("amount", 0.0))
    cur_sheet_rows.append({"Type": "Expense", "Name": r.get("item", ""), "Category": r.get("category", ""), "Amount": amt_for_sheet, "Note": r.get("note", "")})

for r in ps_eff.to_dict(orient="records"):
    cur_sheet_rows.append({"Type": "Planned Saving", "Name": r.get("name", ""), "Category": r.get("category", "Savings"), "Amount": r.get("amount", 0.0), "Note": r.get("note", "")})

st.subheader(f"Sheet — {mkey}")

cur_sheet_df = pd.DataFrame(cur_sheet_rows)
if cur_sheet_df.empty:
    cur_sheet_df = pd.DataFrame([
        {"Type": "Income", "Name": "", "Category": "Income", "Amount": 0.0, "Note": ""}
    ])

edited_sheet = st.data_editor(
    cur_sheet_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Type": st.column_config.SelectboxColumn("Type", options=["Income", "Expense", "Planned Saving"], required=True),
        "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES + ["Income"], help="Income rows keep 'Income'. Expenses choose Necessity/Need/Want. Savings optional."),
        "Amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=100.0),
    },
)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("💾 Save Sheet Edits (sync)", use_container_width=True):
        tmp = edited_sheet.fillna("")
        incomes_new, expenses_new, ps_new = [], [], []

        for _, r in tmp[tmp["Type"] == "Income"].iterrows():
            amt = float(r.get("Amount", 0) or 0)
            incomes_new.append({"source": str(r.get("Name","")), "amount": amt, "note": str(r.get("Note",""))})

        for _, r in tmp[tmp["Type"] == "Expense"].iterrows():
            amt = float(r.get("Amount", 0) or 0)
            cat = str(r.get("Category", "") or "Necessity")
            expenses_new.append({
                "item": str(r.get("Name","")),
                "planned": amt if "planned" not in exp_eff.columns else None,  # optional carry
                "actual": amt,
                "category": cat if cat in CATEGORIES else "Necessity",
                "note": str(r.get("Note","")),
            })

        for _, r in tmp[tmp["Type"] == "Planned Saving"].iterrows():
            amt = float(r.get("Amount", 0) or 0)
            cat = str(r.get("Category", "") or "Savings")
            ps_new.append({"name": str(r.get("Name","")), "amount": amt, "note": str(r.get("Note","")), "category": "Savings" if cat not in CATEGORIES else cat})

        # Merge: keep planned if exists in original; else planned from Amount (for fresh rows)
        merged_exp = []
        for e in expenses_new:
            base = {"item": e["item"], "category": e["category"], "note": e["note"]}
            # try to find existing planned for same item
            match = next((x for _, x in exp_eff.iterrows() if str(x.get("item","")) == e["item"]), None)
            planned_val = e["planned"] if e["planned"] is not None else (float(match.get("planned")) if match is not None and "planned" in match and pd.notna(match.get("planned")) else 0.0)
            merged_exp.append({**base, "planned": planned_val, "actual": e["actual"]})

        data[mkey] = {
            "incomes": incomes_new,
            "expenses": merged_exp,
            "planned_savings": ps_new,
            "one_off_inflows": month_obj.get("one_off_inflows", []),
        }
        save_data(data)
        st.success("Sheet synced with month and saved. Add/remove rows freely anytime.")

with col_b:
    csv_cur = edited_sheet.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Current Sheet (CSV)",
        data=csv_cur,
        file_name=f"budget_sheet_{mkey}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Archive/save the sheet for this month (to analyze later)
if st.button("📦 Save/Archive Current Month Sheet", use_container_width=True):
    snapshot = {
        "month": mkey,
        "snapshot_at": datetime.now().isoformat(timespec="seconds"),
        "incomes": inc_eff.to_dict(orient="records"),
        "expenses": exp_eff.to_dict(orient="records"),
        "planned_savings": ps_eff.to_dict(orient="records"),
        "metrics": {
            "total_income": total_income,
            "total_expenses": expense_total,
            "planned_expenses": planned_expense_total,
            "planned_savings": planned_savings_amt,
            "left_after_necessity": left_after_necessity,
            "net_left": net_left,
            "total_savings": total_savings,
        },
    }
    data.setdefault("_archives", {})
    data["_archives"][mkey] = snapshot
    save_data(data)
    st.success("Archived current month sheet.")

st.subheader("Saved Sheets / History")
archives = data.get("_archives", {})
if not archives:
    st.info("No archived sheets yet. Use the button above to save the current month.")
else:
    keys_sorted = sorted(archives.keys(), reverse=True)
    for mk in keys_sorted:
        snap = archives[mk]
        met = snap.get("metrics", {})
        with st.expander(f"{mk} — Savings: {inr(met.get('total_savings', 0.0))} | Net Left: {inr(met.get('net_left', 0.0))}"):
            sum_df = pd.DataFrame({
                "Metric": [
                    "Total Income", "Planned Expenses", "Total Expenses (Actual)",
                    "Planned Savings", "Left after Necessity (Actual)", "Net Left", "Total Savings"
                ],
                "Amount": [
                    met.get("total_income", 0.0),
                    met.get("planned_expenses", 0.0),
                    met.get("total_expenses", 0.0),
                    met.get("planned_savings", 0.0),
                    met.get("left_after_necessity", 0.0),
                    met.get("net_left", 0.0),
                    met.get("total_savings", 0.0),
                ],
            })
            st.table(sum_df)

            arch_rows = []
            for r in snap.get("incomes", []):
                arch_rows.append({"Type": "Income", "Name": r.get("source", ""), "Category": "Income", "Amount": r.get("amount", 0.0), "Note": r.get("note", "")})
            for r in snap.get("expenses", []):
                amt_for_sheet = r.get("actual", r.get("amount", 0.0))
                arch_rows.append({"Type": "Expense", "Name": r.get("item", ""), "Category": r.get("category", ""), "Amount": amt_for_sheet, "Note": r.get("note", "")})
            for r in snap.get("planned_savings", []):
                arch_rows.append({"Type": "Planned Saving", "Name": r.get("name", ""), "Category": r.get("category", "Savings"), "Amount": r.get("amount", 0.0), "Note": r.get("note", "")})
            arch_df = pd.DataFrame(arch_rows)
            st.dataframe(arch_df, use_container_width=True)
            csv_arch = arch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download {mk} Sheet (CSV)",
                data=csv_arch,
                file_name=f"budget_sheet_{mk}.csv",
                mime="text/csv",
                key=f"dl_{mk}",
            )

st.caption("Tip: Treat 'Planned' ko target aur 'Actual' ko reality samjho—variance pe focus karke overshoot ko jaldi pakdo.")
