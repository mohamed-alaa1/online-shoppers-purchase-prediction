import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, precision_recall_curve)

import features  # joblib needs this module to load the pipeline (add_features lives here)
from features import INPUT_COLS, NUM_INPUTS, CAT_COLS

BASE = Path(__file__).parent
CODE_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]
COUNT_COLS = ["Administrative", "Informational", "ProductRelated"]
ALL_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Online Shoppers Purchase Prediction", layout="wide")


@st.cache_resource
def load_models():
    model = joblib.load(BASE / "models" / "purchase_model.joblib")
    seg = joblib.load(BASE / "models" / "session_kmeans.joblib")
    with open(BASE / "models" / "model_info.json") as f:
        info = json.load(f)
    return model, seg, info


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "online_shoppers_intention.csv")
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_data
def load_test_preds():
    return pd.read_csv(BASE / "models" / "test_predictions.csv")


model, seg, info = load_models()
df = load_data()
THR = info["threshold"]
km_cols = info["segments"]["columns"]
seg_names = info["segments"]["names"]
seg_conv = info["segments"]["conversion"]


def predict(data):
    probs = model.predict_proba(data[INPUT_COLS])[:, 1]
    segs = seg.predict(data[km_cols])
    return probs, segs


def clean_batch(raw):
    # returns the valid rows and a list of problems found
    missing = [c for c in INPUT_COLS if c not in raw.columns]
    if missing:
        raise ValueError("The file is missing these columns: " + ", ".join(missing))
    data = raw.copy()
    bad = pd.Series(False, index=data.index)
    for c in NUM_INPUTS + CODE_COLS:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        bad |= data[c].isna()
    bad |= (data[NUM_INPUTS] < 0).any(axis=1)
    bad |= (data[["BounceRates", "ExitRates", "SpecialDay"]] > 1).any(axis=1)
    bad |= (data[COUNT_COLS + CODE_COLS] % 1 != 0).any(axis=1)
    data["Month"] = data["Month"].astype(str).str.strip().replace({"Jun": "June"})
    bad |= ~data["Month"].isin(ALL_MONTHS)
    data["VisitorType"] = data["VisitorType"].astype(str).str.strip()
    bad |= ~data["VisitorType"].isin(info["categories"]["VisitorType"])
    wk = data["Weekend"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    bad |= wk.isna()
    data["Weekend"] = wk
    good = data[~bad].copy()
    for c in COUNT_COLS + CODE_COLS:
        good[c] = good[c].astype(int)
    good["Weekend"] = good["Weekend"].astype(bool)
    return good, data.index[bad].tolist()


st.sidebar.title("Online Shoppers")
st.sidebar.write(
    "This app uses the models we trained on the UCI Online Shoppers Purchasing Intention data. "
    "It predicts how likely a session is to end with a purchase, puts the session into one of our "
    "K-Means browsing segments, and shows the patterns we found in the data.")
st.sidebar.write("Model: " + info["final_model"] + ", decision threshold " + str(round(THR, 3)))
st.sidebar.info(
    "The predictions are estimates based on past sessions from one shop. They are not certain and "
    "should not be used as the only reason for a decision.")

st.title("Online Shoppers Purchase Prediction")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Predict a session", "Batch CSV scoring", "Visitor segments", "Behavior patterns", "Model performance"])

# tab: single prediction
with tab1:
    d = info["defaults"]
    rng = info["numeric_ranges"]
    st.write("Enter the details of one session. Default values are the medians from our training data.")
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        with c1:
            admin = st.number_input("Administrative pages", min_value=0, value=int(d["Administrative"]), step=1)
            admin_d = st.number_input("Administrative duration (seconds)", min_value=0.0, value=float(d["Administrative_Duration"]))
            info_p = st.number_input("Informational pages", min_value=0, value=int(d["Informational"]), step=1)
            info_d = st.number_input("Informational duration (seconds)", min_value=0.0, value=float(d["Informational_Duration"]))
            prod = st.number_input("Product pages", min_value=0, value=int(d["ProductRelated"]), step=1)
            prod_d = st.number_input("Product pages duration (seconds)", min_value=0.0, value=float(d["ProductRelated_Duration"]))
        with c2:
            bounce = st.number_input("Bounce rate", min_value=0.0, max_value=rng["BounceRates"][1],
                                     value=float(d["BounceRates"]), step=0.001, format="%.4f")
            exit_r = st.number_input("Exit rate", min_value=0.0, max_value=rng["ExitRates"][1],
                                     value=float(d["ExitRates"]), step=0.001, format="%.4f")
            pv = st.number_input("Page value", min_value=0.0, value=float(d["PageValues"]))
            sday = st.selectbox("Special day closeness", info["special_day_values"], index=0)
            month = st.selectbox("Month", info["month_order"], index=info["month_order"].index(d["Month"]))
            weekend = st.checkbox("Weekend session")
        with c3:
            vtype = st.selectbox("Visitor type", info["categories"]["VisitorType"],
                                 index=info["categories"]["VisitorType"].index(d["VisitorType"]))
            cats = {}
            for c in CODE_COLS:
                opts = info["categories"][c]
                cats[c] = st.selectbox(c + " (code)", opts, index=opts.index(d[c]))
        submitted = st.form_submit_button("Predict")

    if submitted:
        warn = []
        if prod == 0 and prod_d > 0:
            warn.append("product duration is above 0 but there are 0 product pages")
        if admin == 0 and admin_d > 0:
            warn.append("administrative duration is above 0 but there are 0 administrative pages")
        if info_p == 0 and info_d > 0:
            warn.append("informational duration is above 0 but there are 0 informational pages")
        if pv > rng["PageValues"][1]:
            warn.append("page value is higher than anything in the training data, the result is less reliable")
        for w in warn:
            st.warning("Check the input: " + w + ".")

        row = pd.DataFrame([{
            "Administrative": admin, "Administrative_Duration": admin_d, "Informational": info_p,
            "Informational_Duration": info_d, "ProductRelated": prod, "ProductRelated_Duration": prod_d,
            "BounceRates": bounce, "ExitRates": exit_r, "PageValues": pv, "SpecialDay": sday, "Month": month,
            "VisitorType": vtype, "Weekend": weekend, **cats}])
        try:
            prob, s = predict(row)
            prob, s = float(prob[0]), str(s[0])
        except Exception as e:
            st.error("The model could not score this input: " + str(e))
            st.stop()

        a, b, c = st.columns(3)
        a.metric("Purchase probability", f"{prob:.1%}")
        b.metric("Decision", "likely to buy" if prob >= THR else "unlikely to buy")
        c.metric("Segment", seg_names[s], f"segment buy rate {seg_conv[s]:.1%}", delta_color="off")
        st.write(
            f"The model gives this session a {prob:.1%} chance of ending with a purchase. We call a session "
            f"'likely to buy' when the probability is at least {THR:.1%}. This threshold was picked on the "
            f"training data to get the best balance between precision and recall. Around {df['Revenue'].mean():.1%} "
            "of all sessions in the data ended with a purchase.")

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("This session compared to buyers and non-buyers")
            comp = pd.DataFrame({"this session": row[NUM_INPUTS].iloc[0],
                                 "median buyer": pd.Series(info["buyer_medians"]),
                                 "median non-buyer": pd.Series(info["nonbuyer_medians"])})
            st.dataframe(comp.round(4), use_container_width=True)
        with g2:
            st.subheader("Which inputs matter most overall")
            imp = pd.Series(info["permutation_importance"]).sort_values()
            fig = px.bar(imp, orientation="h", labels={"value": "drop in PR-AUC when shuffled", "index": ""})
            fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Permutation importance on the test set. It is for all sessions together, not just this one.")

        st.subheader("What if the page value was different")
        top = max(100.0, pv * 2)
        grid = np.linspace(0, top, 60)
        what = pd.concat([row] * len(grid), ignore_index=True)
        what["PageValues"] = grid
        wp = model.predict_proba(what[INPUT_COLS])[:, 1]
        fig = px.line(x=grid, y=wp, labels={"x": "PageValues", "y": "purchase probability"})
        fig.add_hline(y=THR, line_dash="dash", annotation_text="threshold")
        fig.add_vline(x=pv, line_dash="dot", annotation_text="your input")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Everything else is kept the same as your input, only PageValues changes.")

# tab: batch scoring
with tab2:
    st.write("Upload a CSV with the same columns as the original dataset. A Revenue column is optional, "
             "if it is there we also show how well the model did on your file.")
    t1, t2 = st.columns(2)
    t1.download_button("Download empty template", ",".join(INPUT_COLS) + "\n", "template.csv", "text/csv")
    t2.download_button("Download sample file (40 test sessions)",
                       (BASE / "data" / "sample_sessions.csv").read_bytes(), "sample_sessions.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error("Could not read the file as CSV: " + str(e))
            st.stop()
        if raw.empty:
            st.error("The file has no rows.")
            st.stop()
        try:
            good, bad_rows = clean_batch(raw)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        if bad_rows:
            st.warning(f"{len(bad_rows)} row(s) had missing or invalid values and were skipped "
                       f"(row numbers: {bad_rows[:20]}{' ...' if len(bad_rows) > 20 else ''}).")
        if good.empty:
            st.error("No valid rows left to score.")
            st.stop()
        unseen = ~good["Month"].isin(info["month_order"])
        if unseen.any():
            st.info(f"{unseen.sum()} row(s) are from Jan or Apr. These months are not in the training data, "
                    "so the model treats the month as unknown.")

        probs, segs = predict(good)
        res = good.copy()
        res["purchase_probability"] = probs.round(4)
        res["predicted_purchase"] = probs >= THR
        res["segment"] = [seg_names[str(s)] for s in segs]
        st.success(f"Scored {len(res)} sessions. {res['predicted_purchase'].sum()} are predicted as likely buyers.")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results", res.to_csv(index=False), "scored_sessions.csv", "text/csv")

        h1, h2 = st.columns(2)
        fig = px.histogram(res, x="purchase_probability", nbins=30, title="Predicted probabilities")
        fig.add_vline(x=THR, line_dash="dash", annotation_text="threshold")
        h1.plotly_chart(fig, use_container_width=True)
        h2.plotly_chart(px.histogram(res, x="segment", title="Sessions per segment"), use_container_width=True)

        if "Revenue" in raw.columns:
            yt = raw.loc[good.index, "Revenue"].astype(str).str.strip().str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0})
            ok = yt.notna()
            if ok.sum() and yt[ok].nunique() == 2:
                yt, pr = yt[ok].astype(int), res.loc[ok, "purchase_probability"]
                pred = (pr >= THR).astype(int)
                st.subheader("Performance on your file")
                m = st.columns(5)
                m[0].metric("ROC-AUC", f"{roc_auc_score(yt, pr):.3f}")
                m[1].metric("PR-AUC", f"{average_precision_score(yt, pr):.3f}")
                m[2].metric("Precision", f"{precision_score(yt, pred, zero_division=0):.3f}")
                m[3].metric("Recall", f"{recall_score(yt, pred):.3f}")
                m[4].metric("F1", f"{f1_score(yt, pred):.3f}")
                cm = confusion_matrix(yt, pred, labels=[0, 1])
                st.dataframe(pd.DataFrame(cm, index=["actual: no", "actual: yes"],
                                          columns=["predicted: no", "predicted: yes"]))
            else:
                st.info("The Revenue column needs both True and False values to compute metrics.")

# tab: segments
with tab3:
    st.write("We grouped the training sessions with K-Means using page counts, durations, bounce rate, exit rate "
             "and page value. Revenue was not used to build the groups, only to describe them afterwards.")
    prof = pd.DataFrame(info["segments"]["profile"]).T
    prof.index = [seg_names[i] for i in prof.index]
    prof["sessions"] = [info["segments"]["sessions"][i] for i in info["segments"]["sessions"]]
    prof["buy rate"] = [seg_conv[i] for i in seg_conv]
    st.subheader("Segment profiles (medians)")
    st.dataframe(prof.round(3), use_container_width=True)

    sample = df.sample(min(4000, len(df)), random_state=42)
    Z = seg[:-1].transform(sample[km_cols])
    P = PCA(n_components=2, random_state=42).fit_transform(Z)
    plot_df = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1],
                            "segment": [seg_names[str(s)] for s in seg.predict(sample[km_cols])],
                            "purchase": sample["Revenue"].values})
    s1, s2 = st.columns([3, 2])
    s1.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="segment", hover_data=["purchase"], opacity=0.6,
                               title="Sessions on the first two principal components"), use_container_width=True)
    bar = prof.reset_index().rename(columns={"index": "segment"})
    s2.plotly_chart(px.bar(bar, x="segment", y="buy rate", title="Buy rate per segment"), use_container_width=True)
    s2.plotly_chart(px.bar(bar, x="segment", y="sessions", title="Training sessions per segment"), use_container_width=True)

    pick = st.selectbox("Look at one segment", list(prof.index))
    overall = df[km_cols].median()
    st.dataframe(pd.DataFrame({"segment median": prof.loc[pick, km_cols], "all sessions median": overall}).round(3))

# tab: patterns
with tab4:
    f1c, f2c = st.columns(2)
    months = f1c.multiselect("Months", info["month_order"], default=info["month_order"])
    vtypes = f2c.multiselect("Visitor types", info["categories"]["VisitorType"],
                             default=info["categories"]["VisitorType"])
    sub = df[df["Month"].isin(months) & df["VisitorType"].isin(vtypes)]
    if sub.empty:
        st.warning("No sessions match these filters.")
    else:
        st.write(f"{len(sub)} sessions selected, {sub['Revenue'].mean():.1%} of them ended with a purchase.")
        p1, p2 = st.columns(2)
        bym = sub.groupby("Month")["Revenue"].agg(["count", "mean"]).reindex(
            [m for m in info["month_order"] if m in months]).reset_index()
        p1.plotly_chart(px.bar(bym, x="Month", y="mean", hover_data=["count"], labels={"mean": "buy rate"},
                               title="Buy rate by month"), use_container_width=True)
        byv = sub.groupby("VisitorType")["Revenue"].agg(["count", "mean"]).reset_index()
        p2.plotly_chart(px.bar(byv, x="VisitorType", y="mean", hover_data=["count"], labels={"mean": "buy rate"},
                               title="Buy rate by visitor type"), use_container_width=True)
        p3, p4 = st.columns(2)
        byt = sub.groupby("TrafficType")["Revenue"].agg(["count", "mean"]).sort_values("count", ascending=False).head(10)
        byt = byt.reset_index().assign(TrafficType=lambda x: x["TrafficType"].astype(str))
        p3.plotly_chart(px.bar(byt, x="TrafficType", y="mean", hover_data=["count"], labels={"mean": "buy rate"},
                               title="Buy rate for the 10 most common traffic types"), use_container_width=True)
        bins = pd.cut(sub["PageValues"], [-0.01, 0, 10, 25, 50, 100, 1000], labels=["0", "0-10", "10-25", "25-50", "50-100", "100+"])
        byp = sub.groupby(bins, observed=False)["Revenue"].agg(["count", "mean"]).reset_index()
        p4.plotly_chart(px.bar(byp, x="PageValues", y="mean", hover_data=["count"], labels={"mean": "buy rate"},
                               title="Buy rate by page value range"), use_container_width=True)
        sc = sub.sample(min(3000, len(sub)), random_state=42)
        st.plotly_chart(px.scatter(sc, x="BounceRates", y="ExitRates", color="Revenue", opacity=0.5,
                                   title="Bounce rate vs exit rate"), use_container_width=True)

# tab: performance
with tab5:
    st.write("All numbers below are from the test set (20% of the data that was kept aside and never used in training).")
    mt = pd.DataFrame(info["test_metrics"]).T
    st.dataframe(mt.round(3), use_container_width=True)
    tp = load_test_preds()
    fpr, tpr, _ = roc_curve(tp["y_true"], tp["y_prob"])
    pr, rc, _ = precision_recall_curve(tp["y_true"], tp["y_prob"])
    q1, q2 = st.columns(2)
    fig = px.line(x=fpr, y=tpr, labels={"x": "false positive rate", "y": "true positive rate"},
                  title=f"ROC curve, AUC {roc_auc_score(tp['y_true'], tp['y_prob']):.3f}")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    q1.plotly_chart(fig, use_container_width=True)
    q2.plotly_chart(px.line(x=rc, y=pr, labels={"x": "recall", "y": "precision"},
                            title=f"Precision-recall curve, AP {average_precision_score(tp['y_true'], tp['y_prob']):.3f}"),
                    use_container_width=True)

    st.subheader("Try a different threshold")
    th = st.slider("Threshold", 0.05, 0.95, float(round(THR, 2)), 0.01)
    pred = (tp["y_prob"] >= th).astype(int)
    tn, fp, fn, tpp = confusion_matrix(tp["y_true"], pred, labels=[0, 1]).ravel()
    m = st.columns(4)
    m[0].metric("Precision", f"{precision_score(tp['y_true'], pred, zero_division=0):.3f}")
    m[1].metric("Recall", f"{recall_score(tp['y_true'], pred):.3f}")
    m[2].metric("F1", f"{f1_score(tp['y_true'], pred):.3f}")
    m[3].metric("False alarm rate", f"{fp / (fp + tn):.3f}")
    fig = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tpp]], x=["predicted no", "predicted yes"],
                               y=["actual no", "actual yes"], text=[[tn, fp], [fn, tpp]], texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix", height=350, yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A lower threshold finds more buyers (higher recall) but also flags more non-buyers (more false alarms).")
